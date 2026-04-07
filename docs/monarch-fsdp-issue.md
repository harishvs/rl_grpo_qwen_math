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

        model = AutoModelForCausalLM.from_pretrained(self.model_name, ...)
        for layer in model.model.layers:
            fully_shard(layer, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=False)
        self.model = model
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

### Problem 1: full_tensor() timeout

Each `DTensor.full_tensor()` is a separate all-gather collective. A 1.5B model has 339 parameters → 339 sequential collectives → exceeds Monarch's 120s supervision watchdog timeout. No PyTorch API exists to gather an entire FSDP2 state_dict in one collective.

### Problem 2: Cross-process RDMA buffer isolation

We tried a shard-based approach: each learner rank exposes its local FSDP shard (~444 MB) as an RDMABuffer (no all-gather needed). But RDMA buffers created inside a Monarch actor process are not accessible from actors in other processes or nodes.

**What happens**: The learner's rank 0 process creates an `RDMABuffer`. The generator on Node 1 calls `read_into()`. The generator's `IbvManagerActor` sends a `RequestBuffer` message to the learner node's worker loop RDMA manager — but the buffer was registered in rank 0's child process, not the worker loop. The worker loop's RDMA manager doesn't know about it. Result: `delivery timeout`.

This works in the single-GPU GRPO example because the learner actor and its RDMA manager share the same process. With multi-process FSDP, each rank runs in a separate process spawned by the worker loop.

### Current workaround

Serialized weight sync via `torch.save/load` over Monarch actor RPC (~65s per sync for 3GB model). Works but slow.

### Suggestions

1. **Cross-process RDMA buffer visibility**: Allow RDMA buffers registered in child actor processes to be accessible by the parent worker loop's RDMA manager, or provide an API to register buffers at the worker loop level from within actor endpoints.

2. **TorchStore on K8s**: Learner publishes DTensor state to TorchStore, generator subscribes. TorchStore not currently available in torchmonarch 0.4.0 on Kubernetes.

3. **Batched all-gather API**: An `all_gather_flat()` that concatenates all local FSDP shards and gathers in one NCCL call would make the full_tensor approach viable within the 120s timeout.

## Reference implementation

https://github.com/harishvs/rl_grpo_qwen_math (feat/monarch-grpo branch)

Working FSDP + Monarch actor implementation with:
- `setup_torch_elastic_env` + `dist.init_process_group` for NCCL setup
- Composable `fully_shard()` with CPU-offloaded frozen reference model
- Micro-batching with `set_requires_gradient_sync` for gradient accumulation
- 18 issues documented across 4 training runs

## Environment

- torchmonarch 0.4.0
- PyTorch 2.10.0+cu128
- Kubernetes (EKS 1.30) with MonarchMesh CRD operator via Helm
- 2x p4d.24xlarge (8x A100 40GB each)
