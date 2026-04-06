# Monarch GRPO Training Run #3 -- 2026-04-06 00:30 PT

## Cluster
Same as Run #2: rl-code-llm-training-dev, 2x p4d.24xlarge, us-west-2

## Changes from Run #2

### Key fix: setup_torch_elastic_env + dist.init_process_group
- **Previous approach (Run #2)**: `proc_mesh.activate()` inside actor `__init__` — failed because `proc_mesh.activate()` is for Monarch's tensor engine, not FSDP/NCCL
- **New approach**: 
  1. `setup_torch_elastic_env(learner_procs)` in controller — sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT on all 8 learner processes (same env vars torchrun sets)
  2. `dist.init_process_group("nccl")` inside `@endpoint initialize()` — standard PyTorch NCCL rendezvous
  3. Composable `fully_shard()` applied after process group init
- **Why this works**: `current_rank()` and `current_size()` are base Monarch actor APIs, not SPMDActor-specific. We replicate what SPMDActor does but stay as a regular Actor (keeping @endpoint methods and future RDMA support)

### Learner refactored: lightweight __init__ + initialize endpoint
- `__init__` just saves config params (no model loading, no distributed setup)
- New `@endpoint initialize()` does: `dist.init_process_group("nccl")` → load model → `fully_shard()` → optimizer
- Reason: `context().actor_instance` must be fully constructed before calling `proc_mesh`/distributed APIs. Monarch calls `Class(*args, **kwargs)` during init but the actor instance isn't fully registered until after `__init__` returns.

### Removed proc_mesh.activate() from all endpoints
- `train_step`, `get_weights`, `save_checkpoint` no longer wrap code in `with self.proc_mesh.activate():`
- FSDP collectives work automatically once `dist.init_process_group` is initialized — no per-call context manager needed

### Data sharding across FSDP ranks
- Each rank processes `batch_size / world_size` samples (data parallelism)
- FSDP handles weight sharding and gradient all-reduce during backward

### Docker image fixes
- Removed flash-attn CUDA compilation step from Dockerfile (not using flash attention)
- `pip uninstall flash-attn` on both pods — broken flash_attn_2_cuda.so was crashing vLLM's rotary embedding import
- vLLM falls back to non-flash-attn rotary path

### FSDP2 fixes (composable fully_shard)
- **reshard_after_forward**: Added `reshard_after_forward=True` per layer, `False` on root — frees gathered weight shards after each layer's forward
- **Model loaded on CPU, not GPU**: `fully_shard()` handles CPU→CUDA movement internally. Moving to GPU first prevented proper sharding.
- **HF gradient_checkpointing_enable() doesn't work with FSDP2**: Has zero effect on memory. Replaced with PyTorch's `apply_activation_checkpointing` targeting Qwen2DecoderLayer.
- **use_cache=False**: Disabled KV cache in forward pass — useless during training, wastes memory.
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True**: Reduces CUDA memory fragmentation.

### Micro-batching for activation memory
- Forward + backward in micro-batches of 4 samples per rank
- Gradients accumulate across micro-batches, one optimizer step at the end
- Limits peak activation memory to ~4 samples worth instead of 32

### ValueMesh API
- `.call()` returns `ValueMesh`, not a list — use `.values()[0]` to get rank 0's result

## What Worked

### FSDP initialization -- SOLVED (the Run #1/Run #2 blocker)
```
Learner: rank 0/8 initialized on cuda:0
Learner: rank 1/8 initialized on cuda:1
...
Learner: rank 7/8 initialized on cuda:7
```
All 8 FSDP ranks initialized successfully with NCCL process groups.

### Weight sharding confirmed
After init, each GPU used only 1.4 GiB (3 GiB model / 8 ranks ≈ 375 MB per shard + overhead). Correct FSDP2 behavior.

### Host slicing for split placement
- `hosts.slice(hosts=slice(0, 1))` for learner, `hosts.slice(hosts=slice(1, 2))` for generator
- Dimension name is `hosts` (plural), discovered via `hosts.sizes` → `{'hosts': 2}`

### Generator (vLLM) — worked after flash-attn fix
- vLLM 0.19.0 engine created successfully on pod 1 after uninstalling broken flash-attn
- Generation completed in ~60s (includes CUDA graph warmup)

### Training steps completed (2 steps before OOM)
```
step:0 | loss=-0.0003 | kl=0.0005 | reward=0.1523 | grad_norm=4.91 | 56.6s
step:1 | loss=0.0082 | kl=0.0203 | reward=0.1484 | grad_norm=39.5 | 11.1s | 23.2 samples/sec
```
- Full GRPO loop working: dataset → generate → score → train → log
- Step 0 slow (56s) due to vLLM warmup. Step 1 at 11s = 23 samples/sec
- Reward ~15% correct — expected for untrained Qwen 1.5B on GSM8K

## What Failed

### OOM during backward on step 2
```
torch.OutOfMemoryError: Tried to allocate 10.09 GiB. GPU 0 has 39.49 GiB total, 2.52 GiB free.
25.33 GiB allocated by PyTorch, 10.54 GiB reserved but unallocated.
```
- Activation memory consumed ~37 GiB per rank during forward+backward
- FSDP only shards weights/gradients/optimizer — activations are NOT sharded
- HF's `gradient_checkpointing_enable()` had zero effect with FSDP2
- 32 samples per rank with variable-length sequences → huge activation footprint
- Fix: micro-batching (4 samples per forward+backward, accumulate gradients) + PyTorch activation checkpointing + expandable_segments

### Previous OOMs (all fixed before reaching training)
- OOM on train_step (all ranks processing full 256-sample batch) → data sharding
- OOM on initialize (stale GPU memory from previous run) → pod restart
- OOM on `.to(cuda)` before `fully_shard()` → load model on CPU

## Complete Issue Log (Run #3)

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `KubernetesJob` import error | Stale image had top-level import from wrong path | Local file already fixed; image had old code |
| 2 | `HostMesh not subscriptable` | HostMesh uses `.slice()` not `[0:1]` | `hosts.slice(hosts=slice(0,1))` |
| 3 | `call_one requires 1 actor, mesh has 2` | Generator mesh spanned both hosts | Split hosts with `.slice()` |
| 4 | `proc_mesh.activate()` crash in `__init__` | Actor instance not fully constructed during `__init__` | Move to `@endpoint initialize()` |
| 5 | `proc_mesh.activate()` wrong for FSDP | It's for Monarch tensor engine, not NCCL | `setup_torch_elastic_env` + `dist.init_process_group` |
| 6 | vLLM engine core init failed | Broken `flash_attn_2_cuda.so` imported by vLLM rotary embedding | `pip uninstall flash-attn` |
| 7 | vLLM "Free memory less than desired" | Stale GPU memory from previous vLLM test runs | Restart pod 1 |
| 8 | OOM: all ranks process full batch | No data sharding across FSDP ranks | Each rank gets `batch_size / world_size` |
| 9 | OOM: `.to(cuda)` before `fully_shard()` | Full model on GPU before sharding | Load on CPU, let `fully_shard()` handle device placement |
| 10 | HF gradient checkpointing no effect | Doesn't compose with FSDP2 | PyTorch `apply_activation_checkpointing` |
| 11 | `ValueMesh` not subscriptable | `.call()` returns ValueMesh, not list | Use `.values()[0]` |
| 12 | OOM during backward (step 2) | 32 samples per rank → huge activation memory | Micro-batching: 4 samples per forward+backward |

## Key Learnings

1. **`setup_torch_elastic_env()` is the bridge** between Monarch actors and PyTorch distributed. It uses `current_rank()`/`current_size()` (base actor APIs) to set torchrun-style env vars. Then standard `dist.init_process_group("nccl")` works.

2. **Regular actors can do FSDP** — no need for SPMDActor. This preserves @endpoint methods (per-step control) and future RDMA weight sync capability.

3. **`proc_mesh.activate()` ≠ NCCL process groups**. It's for Monarch's distributed tensor engine. FSDP needs real NCCL process groups via `dist.init_process_group`.

4. **Actor `__init__` timing matters**. The actor instance (`context().actor_instance`) is set DURING `__init__`, but methods that depend on the fully-constructed instance (like `spawn_tensor_engine`) can fail. Use a post-spawn `@endpoint` for heavy initialization.

5. **flash-attn is deeply embedded** — even vLLM imports it for rotary embeddings. A broken flash-attn breaks everything, not just attention.

6. **FSDP2 only shards weights, not activations**. For memory savings during forward/backward, you need activation checkpointing AND micro-batching. HuggingFace's `gradient_checkpointing_enable()` does NOT work with FSDP2 — must use PyTorch's `apply_activation_checkpointing`.

7. **ValueMesh, not list**. Monarch's `.call()` returns a `ValueMesh` object. Use `.values()` to extract results, `.item()` for single-value meshes.

8. **GPU memory cleanup requires pod restart**. Killing Monarch bootstrap processes frees CPU memory but not GPU memory — CUDA contexts persist until the process (PID 1 worker loop) exits.

## Timeline

| Time (PT) | Event |
|-----------|-------|
| ~00:30 | Run #3 started. FSDP init works (8/8 ranks). Generator fails: flash-attn crash |
| ~00:35 | Uninstall flash-attn on both pods. Generator starts. OOM on train_step (no data sharding) |
| ~00:45 | Add data sharding. OOM on init (stale GPU memory). Restart pod 0 |
| ~00:50 | Relaunch. GPU memory 1.4 GiB/rank after init (FSDP sharding works!). Hangs on step |
| ~01:00 | Discover model was `.to(cuda)` before `fully_shard()` — 40 GiB/rank during forward |
| ~01:10 | Fix: load on CPU, let fully_shard handle device placement. Restart both pods |
| ~01:20 | Relaunch. FSDP shards correctly (1.4 GiB). Training step runs! |
| ~01:25 | TypeError: ValueMesh not subscriptable. Fix: `.values()[0]` |
| ~01:35 | 2 training steps complete! step:0 56s, step:1 11s (23 samples/sec). reward=0.15 |
| ~01:40 | OOM on backward (step 2). Activation memory ~37 GiB. HF grad ckpt has no effect with FSDP2 |
| ~02:00 | Fix: PyTorch activation checkpointing + micro-batching (4 samples) + expandable_segments |
