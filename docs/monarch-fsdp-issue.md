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

## Environment

- torchmonarch 0.4.0
- PyTorch 2.10.0+cu128
- Kubernetes (EKS 1.30) with MonarchMesh CRD operator via Helm
- 2x p4d.24xlarge (8x A100 40GB each)
