# Monarch GRPO Training Run -- 2026-04-05

## Cluster
- **EKS**: rl-code-llm-training-dev, EKS 1.30, us-west-2
- **GPU Nodes**: 2x p4d.24xlarge (8x A100 40GB each) in us-west-2d, capacity block cr-0e35a6085ded0488f
- **CPU Nodes**: 2x c7i.large (us-west-2a, us-west-2c)
- **Monarch**: torchmonarch 0.4.0, MonarchMesh CRD operator via Helm
- **Model**: Qwen/Qwen2.5-1.5B
- **Dataset**: GSM8K (7,473 train problems)

## What Worked

### Infrastructure
- MonarchMesh operator installed via Helm, CRD created successfully
- 2 GPU pods running `run_worker_loop_forever()` on port 26600
- Pod discovery via K8s DNS: `grpo-monarch-0.grpo-monarch-svc.default.svc.cluster.local`
- Lightweight reward service (python:3.12-slim, ~150MB) on CPU pod via FastAPI
- Reward service reachable from GPU pods at `http://monarch-reward:8080`

### Actors
- **DatasetActor**: GSM8K loaded from HuggingFace, batches served correctly
- **GeneratorActor**: vLLM v0.19.0 loaded Qwen2.5-1.5B in-process, generated 256 completions (32 prompts × 8 group size) at ~8,400 tokens/sec using Flash Attention 2
- **ReplayBufferActor**: Episode storage working
- **RewardActor**: Binary scoring via HTTP to FastAPI service

### Generation Output
```
Processed prompts: 100% | 256/256 [00:06, 36.62it/s, input: 2651 toks/s, output: 8424 toks/s]
```

## What Failed

### FSDP Training -- Cannot Form Distributed Process Group

**Root cause**: Monarch actors are isolated processes with separate mailboxes. When we spawn 8 LearnerActors (one per GPU for FSDP), each calls `dist.init_process_group("nccl")` independently, but they cannot rendezvous with each other because:

1. No shared MASTER_ADDR/MASTER_PORT between actor processes
2. `setup_torch_elastic_env()` sets env vars but each actor's init_process_group creates a separate group
3. Monarch's process isolation prevents the NCCL rendezvous that FSDP requires

**This is the fundamental tension**: Monarch's actor model (isolated processes, message passing) and PyTorch's FSDP (shared NCCL process group, tightly coupled ranks) are incompatible paradigms. TorchForge solves this with `TitanTrainer` + `TorchStore`, but TorchForge doesn't support K8s.

## Attempts Made

### Attempt 1: Regular Actor + `dist.init_process_group`
- **Error**: `ValueError: RANK env var not set`
- **Why**: Monarch doesn't set torchrun-style env vars (RANK, WORLD_SIZE, MASTER_ADDR)

### Attempt 2: `setup_torch_elastic_env(learner_procs)`
- **Error**: FSDP hangs during model initialization
- **Why**: Env vars are set on the mesh level, but each actor process still initializes its own separate process group -- they don't find each other for NCCL rendezvous

### Attempt 3: SPMDActor (considered, not implemented)
- SPMDActor runs a standalone torchrun-compatible script
- Would solve FSDP, but the Learner becomes a black-box -- controller loses per-step orchestration
- Without TorchForge's TorchStore, no way to exchange weights/batches between the script and other actors

## Other Issues Encountered and Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Pods crash-looping | No entrypoint command | Added `run_worker_loop_forever()` bootstrap |
| Pod label mismatch | Operator uses `monarch.pytorch.org/mesh-name`, not `app=monarch-grpo` | Updated label selectors in script |
| CRD API group wrong | `monarch.meta.com` → `monarch.pytorch.org` | Fixed manifest |
| CRD spec structure | `template.spec` → `podTemplate` (flat pod spec) | Rewrote manifest |
| 19GB image on CPU node | Disk pressure eviction on 50GB c7i.large | Created lightweight python:3.12-slim reward image |
| `KubernetesJob` import | Not exported from `monarch.job`, lives at `monarch._src.job.kubernetes` | Fixed import path |
| `ReplayBufferActor.size` | Collides with Monarch's `ActorMesh.size` attribute | Renamed to `buffer_size` |
| `current_rank()` returns Point | Not an int, has `.rank` property | Used `point.rank` |
| Flash Attention broken | torchmonarch upgraded PyTorch 2.6→2.10, pre-compiled flash_attn incompatible | Reinstalled flash_attn from source |
| vLLM broken | Same PyTorch upgrade broke vLLM's `_C.abi3.so` | Reinstalled vLLM |
| NumPy 2.x incompatible | NumPy 2.2.6 breaks modules compiled against NumPy 1.x | Downgraded to numpy<2 |
| `call_one()` on 8-actor mesh | FSDP needs all ranks, `call_one` targets one | Changed to `call()` broadcast |
| Batch key mismatch | Generator returns `log_probs`, Learner expects `old_log_probs` | Added fallback key lookup |
| Mesh name invalid | `grpo-monarch` has hyphen, KubernetesJob requires alphanumeric only | Used label_selector instead |

## Remaining Options

### Option 1: SPMDActor with standalone training script
- Learner runs as a torchrun-compatible script via SPMDActor
- FSDP works because SPMDActor sets up proper rendezvous
- Downside: Controller loses fine-grained per-step orchestration
- Communication between Learner script and other actors via files, HTTP, or shared filesystem

### Option 2: Single-GPU Learner (no FSDP)
- One LearnerActor on one GPU, gradient accumulation for effective batch size
- Qwen 1.5B fits on one A100 40GB (~3GB in bf16)
- Simplest fix, doesn't scale to 7B
- Keeps the current architecture intact

### Option 3: Hybrid approach
- Monarch for Generator/Reward/Dataset actors
- Separate torchrun Job for the Learner (like the custom trainer)
- Weight sync via HTTP or shared storage between the two systems
- Mixes paradigms but both parts work independently

### Option 4: Wait for Monarch tensor engine / FSDP support
- Monarch's distributed tensor engine may handle this in future versions
- Not available in v0.4.0

## Key Learnings

1. **Monarch actors ≠ torch.distributed ranks**: They're fundamentally different parallelism models. Actors are isolated with message passing. FSDP ranks are tightly coupled with NCCL collectives. You can't simply spawn FSDP ranks as actors.

2. **TorchForge exists for a reason**: It bridges Monarch actors and PyTorch training with `TitanTrainer` and `TorchStore`. Without it, this bridge must be built manually.

3. **Monarch is excellent for heterogeneous workloads**: Generator, Reward, Dataset actors worked perfectly. The actor model shines when components have different resource needs and can run independently.

4. **Monarch K8s support is functional but rough**: MonarchMesh CRD works, but the API has sharp edges (no hyphens in names, private import paths, undocumented pod spec format).

5. **NGC base image + torchmonarch = version hell**: torchmonarch pulls PyTorch 2.10 which breaks all pre-compiled CUDA extensions (flash_attn, vLLM). Must reinstall from source after pip install.

## Files Created
- `src/shared/` -- reward, grpo, dataset, reward_service (4 files)
- `src/monarch/` -- 5 actors, config, main, metrics (9 files)
- `config/monarch/` -- qwen-1.5b.yaml, qwen-7b.yaml
- `docker/monarch/` -- Dockerfile, Dockerfile.reward, requirements.txt
- `k8s/monarch/` -- monarchmesh.yaml, reward-deployment.yaml, configmap.yaml, serviceaccount.yaml
- `scripts/monarch/` -- run-training.sh, build-and-push-images.sh, collect-logs.sh
- `scripts/infra/` -- teardown-eks.sh
- `docs/diagrams/monarch/` -- 4 Mermaid diagrams + PNGs
- `docs/monarch-concepts.md`, `docs/monarch-code-walkthrough.md`
- `tests/test_monarch.py` -- 37 unit tests
