# Feature Request: FSDP training example for Monarch actors on Kubernetes

## Summary

Please add an example showing how to run FSDP (Fully Sharded Data Parallel) training inside a regular Monarch actor on Kubernetes, using the MonarchMesh CRD operator.

## Context

The existing examples cover:
- **DDP with SPMDActor** (`examples/ddp/spmd_ddp.html`) — uses SPMDActor to set torchrun-style env vars, then runs a standalone training script
- **GRPO actor** (`examples/grpo_actor.html`) — single-GPU learner with RDMA weight sync, no FSDP

There is no example that combines FSDP with regular Monarch actors. This matters because:
- SPMDActor is a black-box script launcher — you lose `@endpoint` methods, RDMA weight sync, and per-step orchestration from the controller
- The GRPO example's single-GPU learner doesn't scale beyond models that fit on one GPU
- Users who need both FSDP (for large models) and actor endpoints (for weight sync, checkpointing, metrics) have no reference pattern

## What we tried (and what works)

After significant trial and error on torchmonarch 0.4.0, we found a working pattern:

### 1. Use `setup_torch_elastic_env()` on the proc mesh before spawning the actor

```python
from monarch.spmd import setup_torch_elastic_env

learner_procs = learner_host.spawn_procs(per_host={"gpus": 8})
setup_torch_elastic_env(learner_procs)  # Sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

learner = learner_procs.spawn("learner", LearnerActor, ...)
await learner.initialize.call()  # Post-spawn init for FSDP
```

### 2. In the actor, call `dist.init_process_group("nccl")` in a post-spawn `@endpoint`

```python
class LearnerActor(Actor):
    def __init__(self, model_name, ...):
        # Lightweight -- just save config. No distributed setup here.
        self.model_name = model_name

    @endpoint
    async def initialize(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # Actor model (trainable, FSDP on GPU)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, ...)
        for layer in model.model.layers:
            fully_shard(layer, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=False)
        self.model = model

        # Reference model (frozen, FSDP with CPU param offload)
        ref_model = AutoModelForCausalLM.from_pretrained(self.model_name, ...)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        offload = CPUOffloadPolicy()
        for layer in ref_model.model.layers:
            fully_shard(layer, reshard_after_forward=True, offload_policy=offload)
        fully_shard(ref_model, reshard_after_forward=True, offload_policy=offload)
        self.ref_model = ref_model
```

### 3. Use `.call()` (broadcast) for training steps so all FSDP ranks participate

```python
train_results = await learner.train_step.call(batch)  # All 8 ranks run together
```

## Why `__init__` doesn't work for FSDP setup

Calling `dist.init_process_group()` or `proc_mesh.activate()` inside `__init__` fails because the actor instance (`context().actor_instance`) is not fully constructed during `__init__` — the Monarch runtime sets `Instance.instance = Class(*args, **kwargs)` and registers the actor AFTER `__init__` returns. APIs that depend on the fully-registered actor (like `spawn_tensor_engine` via `proc_mesh.activate()`) fail with "Actor object is missing."

## What would be helpful

An official example showing:
1. FSDP inside a regular Monarch actor (not SPMDActor)
2. The correct init sequence (`setup_torch_elastic_env` → spawn → post-spawn endpoint for `dist.init_process_group`)
3. How `proc_mesh.activate()` (tensor engine) relates to `dist.init_process_group` (NCCL) — these are different things but easy to conflate
4. Data sharding pattern (each FSDP rank processes batch_size/world_size)
5. Weight sync from FSDP actor to a single-GPU generator actor (ideally via RDMA)

## Weight sync challenge: FSDP + RDMA

The official GRPO example uses RDMA for weight sync (single-GPU learner → generator). With FSDP across multiple processes, this doesn't work for two reasons:

### Problem 1: full_tensor() collectives are slow (but don't always timeout)

Each `DTensor.full_tensor()` is a separate all-gather collective. A 1.5B model has 339 parameters → 339 sequential collectives (~30s total). This does NOT timeout when used in a simple endpoint like `get_weights()` (returns bytes). However, when combined with RDMA buffer creation (`expose_weights`), the total time exceeded Monarch's 120s supervision watchdog. The timeout was from RDMA registration overhead, not `full_tensor()` alone.

### Problem 2: Cross-process RDMA buffer isolation

We tried a shard-based approach: each learner rank exposes its local FSDP shard (~444 MB) as an RDMABuffer (no all-gather needed). But RDMA buffers created inside a Monarch actor process are not accessible from actors in other processes or nodes.

**What happens**: The learner's rank 0 process creates an `RDMABuffer`. The generator on Node 1 calls `read_into()`. The generator's `IbvManagerActor` sends a `RequestBuffer` message to the learner node's worker loop RDMA manager — but the buffer was registered in rank 0's child process, not the worker loop. The worker loop's RDMA manager doesn't know about it. Result: `delivery timeout`.

This works in the single-GPU GRPO example because the learner actor and its RDMA manager share the same process. With multi-process FSDP, each rank runs in a separate process spawned by the worker loop.

### Problem 3: EFA doesn't support ibverbs RC queue pairs (root cause of RDMA failure)

Even with a ParameterServerActor in the worker process (correct RDMA manager visibility), `read_into()` fails. Testing revealed:

```
$ ibv_rc_pingpong -d rdmap16s27
Couldn't create QP

$ ibv_ud_pingpong -d rdmap16s27 localhost
Failed to create AH, GID ::
```

AWS EFA exposes `/dev/infiniband/uverbs*` for compatibility but doesn't support RC or standard UD queue pairs. EFA only works through **libfabric** (`FI_PROTO_EFA`). Monarch's `RDMABuffer` uses ibverbs directly → always fails on EFA.

### Problem 4: vLLM reload_weights corrupts model state

We tried saving weights to FSx Lustre (shared filesystem) and using vLLM's `reload_weights(weights_path=)`. The weights load correctly, but `reload_weights` internally sets `self.model_config.model = weights_path`, which corrupts vLLM's state for subsequent generation. Reward declined from 0.14 → 0.09.

### Problem 5: apply_model incompatible with TP>1

vLLM's `apply_model(fn)` sends the function to worker processes via `collective_rpc`. With TP>1, the closure can't be pickled across workers. `VLLM_ALLOW_INSECURE_SERIALIZATION=1` doesn't help.

### Problem 6: FSDP named_parameters() ≠ state_dict() keys

When reconstructing state_dict from NCCL-gathered flat buffer, using `named_parameters()` keys produces FSDP-namespace keys. vLLM expects HF-namespace keys (from `state_dict()`). With `strict=False`, all keys silently fail to match — zero weights loaded.

### Current workaround

Serialized weight sync via `torch.save/load` over Monarch actor RPC (~47s per sync for 3GB model with TP=4 vLLM). This is the **only** approach that consistently produces correct weights across 10+ training runs and hundreds of weight syncs. Every optimization attempt (RDMA, gloo, FSx, NCCL gather reconstruction) introduced subtle correctness issues.

### Why fast weight sync matters — training results prove it

With the correct hyperparameters (`kl_coef=0.001`, matching veRL), we found that **weight sync every step is critical**:

| Sync interval | Best reward (1 epoch) | Notes |
|---|---|---|
| Every 3 steps | 0.24 | Stale old_log_probs corrupt PPO ratio 2/3 of steps |
| Every 1 step | **0.53+ (still climbing)** | Fresh weights = correct PPO ratio every step |

The 47s per-sync overhead (serialized RPC) adds ~3 hours to a 233-step epoch. This is the single biggest performance bottleneck in our Monarch GRPO implementation. Faster weight sync (via RDMA, shared memory, or colocated placement) would directly translate to faster training without sacrificing accuracy.

With split placement on Monarch, weight sync is unavoidable. veRL avoids this entirely via colocated placement (zero-cost weight resharding). **This is the strongest argument for Monarch to support colocated/role-switching placement, or to provide a fast weight transfer primitive (e.g., TorchStore, or RDMA over libfabric).**

### Suggestions

1. **Monarch RDMA over libfabric**: Instead of ibverbs, use libfabric like NCCL does via aws-ofi-nccl. This would make RDMABuffer work on AWS EFA.

2. **TorchStore on K8s**: Can TorchStore be used as a standalone package in Monarch on Kubernetes, without requiring TorchForge/Slurm? This would be the cleanest solution for DTensor state transfer between actor meshes.

3. **Cross-process RDMA buffer visibility**: Allow RDMA buffers registered in child actor processes to be accessible by the parent worker loop's RDMA manager.

4. **vLLM weight loading for TP>1**: A `load_state_dict_distributed(state_dict)` API that works across TP workers without pickling closures or mutating model_config would solve the TP>1 weight sync problem.

### Debugging notes

We also tried a ParameterServerActor approach (flat CPU buffer in its own process, learner ranks push shards via messages, generator reads via RDMA). The push step works, but the cross-node RDMA `read_into` from generator (Node 1) to param server (Node 0) fails with a delivery timeout. This suggests RDMA buffer negotiation between processes on different nodes may have connectivity issues even when EFA/ibverbs is correctly configured (`ibverbs` backend confirmed, 4 EFA devices per node).

## Reference implementation

https://github.com/harishvs/rl_grpo_qwen_math (feat/monarch-grpo branch)

Working FSDP + Monarch actor implementation with:
- `setup_torch_elastic_env` + `dist.init_process_group` for NCCL setup
- Composable `fully_shard()` with CPU-offloaded frozen reference model
- Micro-batching with `set_requires_gradient_sync` for gradient accumulation
- TP=4 vLLM generation, tested with both 256 and 2048 completions/step
- Reward: 14.5% → 53%+ in 14 steps (with correct hyperparameters)
- 20+ issues documented across 11 training runs
- Weight sync is the primary bottleneck: 47s/sync × 233 steps = ~3 hours overhead per epoch

## Environment

- torchmonarch 0.4.0
- PyTorch 2.10.0+cu128
- Kubernetes (EKS 1.30) with MonarchMesh CRD operator via Helm
- 2x p4d.24xlarge (8x A100 40GB each)
