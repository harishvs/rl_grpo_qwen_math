# Monarch GRPO — Observations So Far (2026-04-06)

## Status: Preparing Run #4 — Adding Frozen Reference Model

## What Works (proven in Run #3)
- FSDP initialization: 8 ranks on Node 0, NCCL via setup_torch_elastic_env
- Composable fully_shard() with reshard_after_forward, CPU model loading
- PyTorch activation checkpointing (not HF's — doesn't work with FSDP2)
- Micro-batching with set_requires_gradient_sync for correct gradient accumulation
- vLLM generation on Node 1: ~8400 tok/s
- Weight sync: FSDP DTensor→plain tensor→serialize→apply_model on vLLM
- Full training loop: 233 steps, no crashes, 26 samples/sec steady state
- GPU memory: 7-9 GiB per learner rank (down from 40 GiB before fixes)

## What Didn't Work: Model Did Not Learn
- Reward flat at 10-22% across 233 steps (full epoch)
- KL divergence: periodic spikes up to 1400, unstable
- No upward reward trend despite correct GRPO loss computation

## Root Cause Analysis

### Investigation: Custom vs Monarch trainer comparison

Side-by-side code comparison revealed 3 key differences:

#### 1. KL gradient flow (CRITICAL)
- **Custom** (`trainer.py:220`): KL computed from **detached** ratio — contributes ZERO gradient. KL is monitoring only.
  ```python
  kl_tokens = (token_ratio.detach() - 1) - torch.log(token_ratio.detach())
  ```
- **Monarch** (`learner.py:216`): KL computed from **live** ratio — gradients flow through KL term. KL actively fights the policy update.
  ```python
  kl = (ratio - 1) - (new_logps - old_logps)  # NOT detached
  ```
- **Impact**: Monarch's KL penalty creates conflicting gradients. When KL spikes, the KL gradient dominates and destabilizes training. Custom avoids this entirely.

#### 2. No frozen reference model
- **veRL**: Uses frozen reference model with FSDP CPU offload (`ref.fsdp_param_offload: true`). KL computed against the original base policy — stable anchor.
- **Custom**: No ref model, but KL is detached so it doesn't matter — KL has no training effect.
- **Monarch**: No ref model AND KL has gradient flow — worst of both worlds. The "reference" is old_log_probs from the previous generation step, which drifts with every update.
- **Result**: veRL reached 77% accuracy. Custom reached 59% reward (no test accuracy gain). Monarch stayed flat.

#### 3. Custom's KL=0.0000 explained
- Custom had `KL=0.0000 for 99% of steps` because KL was detached — the model updated freely without resistance.
- At lr=1e-6 with small batch_size=4, updates were small enough that KL stayed near zero naturally.
- This "accidentally stable" training let custom improve reward from 0.41→0.59 over 911 steps.

### Other differences (less impactful)
- **Prompt format**: Custom uses `to_prompt()` ("Solve the following..."), Monarch uses `to_chat_prompt()` content ("Let's think step by step..."). Tested directly — both work with Qwen2.5-1.5B base model. Raw prompt actually produces better completions than chat template (model is base, not instruct).
- **Batch size**: Custom 32 rollouts/step, Monarch 256 rollouts/step (8x larger — should be more stable, not less).
- **FSDP gradient sync**: Custom has redundant all-reduces per micro-batch (no no_sync). Monarch correctly uses set_requires_gradient_sync. Custom's approach is wasteful but not broken.

## Plan: Run #4

Add a frozen reference model to the learner, matching veRL's approach:

1. **Load second copy of Qwen 1.5B** on each learner rank
2. **FSDP with CPUOffloadPolicy** — params on CPU, moved to GPU only during forward pass
3. **Compute ref log probs** during train_step — forward pass through frozen model for each micro-batch
4. **KL = (ratio_to_ref - 1) - log(ratio_to_ref)** where ratio_to_ref = exp(new_logps - ref_logps)
5. **KL stays detached** (monitoring only) — PPO clipping controls update magnitude

This matches veRL's architecture:
- veRL: actor + ref on same GPUs, ref offloaded to CPU
- Monarch: actor + ref on same 8 learner ranks, ref offloaded to CPU

Memory impact: ~3 GB CPU RAM per rank for 1.5B model. Negligible on p4d (1.5 TB RAM).
Time impact: One extra forward pass per step (~3-5s for 1.5B with CPU offload).

## Changes Made

### learner.py
- `initialize`: Loads frozen reference model with FSDP2 + CPUOffloadPolicy (params on CPU)
- `_forward_log_probs`: New shared helper for forward pass through any model
- `_compute_ref_log_probs`: Forward pass through ref_model with torch.no_grad()
- `train_step`: Computes ref log probs per micro-batch, KL against frozen ref (detached)
- KL is now monitoring only (detached) — PPO clipping alone controls updates

### config
- Reverted to veRL's proven hyperparameters: lr=5e-6, kl_coef=0.1, clip_range=0.2
- Previous conservative settings (lr=1e-6, kl_coef=0.2, clip_range=0.1) were compensating for the missing ref model

## 18 Issues Fixed Across Runs #1-3

| # | Issue | Fix |
|---|-------|-----|
| 1 | KubernetesJob top-level import | Lazy import inside try/except |
| 2 | HostMesh not subscriptable | hosts.slice(hosts=slice(0,1)) |
| 3 | call_one on 2-actor mesh | Split hosts for learner vs generator |
| 4 | proc_mesh.activate() crash in __init__ | Move to @endpoint initialize() |
| 5 | proc_mesh.activate() wrong for FSDP | setup_torch_elastic_env + dist.init_process_group |
| 6 | vLLM flash_attn crash | pip uninstall flash-attn, remove from Dockerfile |
| 7 | vLLM stale GPU memory | Restart pods |
| 8 | OOM: no data sharding | Each rank gets batch/world_size |
| 9 | OOM: model .to(cuda) before fully_shard | Load on CPU, let FSDP handle |
| 10 | HF gradient checkpointing no-op with FSDP2 | PyTorch apply_activation_checkpointing |
| 11 | ValueMesh not subscriptable | .values()[0] |
| 12 | OOM during backward | Micro-batching (4 samples per fwd+bwd) |
| 13 | vLLM model_executor removed in v0.19 | engine.apply_model() public API |
| 14 | apply_model closure not serializable | VLLM_ALLOW_INSECURE_SERIALIZATION=1 |
| 15 | reload_weights treats path as HF repo | Use apply_model instead |
| 16 | FSDP DTensor in state_dict | full_tensor().cpu() conversion |
| 17 | FSDP gradient sync during accumulation | model.set_requires_gradient_sync(False) |
| 18 | set_requires_gradient_sync import error | It's a method on the model, not a function |

## Run #4a: Reference Model Results (30 steps captured)

KL completely stabilized with frozen reference model:
- KL range: 0.0000 - 0.0090 (vs 0.0005 - 1400 without ref model)
- Reward trending up slightly: 0.14 → 0.20-0.25
- Step time unchanged (~10s normal, ~65s weight sync)

## Run #4b: RDMA Weight Sync

### EFA Fix
- EFA device plugin was CrashLoopBackOff — missing /dev/infiniband volume mount
- Replaced manual DaemonSet with official Helm chart (`aws-efa-k8s-device-plugin`)
- Ref: https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html
- MonarchMesh pods now request `vpc.amazonaws.com/efa: 4`
- RDMABuffer reports `ibverbs` backend (real RDMA, not TCP fallback)

### RDMA Attempts
1. **339 individual RDMA buffers**: Crashed — supervision timeout (120s) exceeded. Too many ibverbs memory registrations + full_tensor collectives.
2. **Only rank 0 creates buffers, others exit early**: Deadlocked — full_tensor() is a collective, all ranks must call it.
3. **All ranks call full_tensor, only rank 0 creates buffers**: Still timed out — 339 separate registrations too slow.
4. **Single flat buffer**: Concatenate all params into one 3GB tensor, one RDMA buffer, one read_into call. Plus `refresh_weights` endpoint to update the flat buffer in-place after training steps.

### Architecture (attempt 4)
- `expose_weights()`: All ranks gather shards → rank 0 concatenates into flat tensor → one RDMABuffer. Called once at init.
- `refresh_weights()`: All ranks re-gather shards → rank 0 overwrites flat tensor in-place. Called before each sync.
- `sync_weights_rdma()`: Generator does one `read_into` for 3GB → slices back into params → `load_state_dict`.
- RDMA buffer handle stays valid across refreshes (same memory address).

### Root cause of RDMA crashes
- Monarch supervision watchdog timeout (120s) — `full_tensor()` calls 339 FSDP all-gather collectives sequentially, exceeding timeout
- Not OOM, not stale buffers — purely a time limit issue

### Research findings
- **No PyTorch API** exists to gather an entire FSDP2 state_dict in one collective. Each DTensor's `full_tensor()` is a separate all-gather.
- **veRL** doesn't use RDMA either — it uses in-process weight passing or serialized transfers.
- **Expert recommendation**: Use Monarch TorchStore to publish FSDP DTensor state between meshes. However, TorchStore is not available in torchmonarch 0.4.0 — it requires TorchForge (Slurm/MAST only).
- **Potential optimization**: Access local shards via `dtensor._local_tensor`, concatenate, then one `all_gather_into_tensor()` — replaces 339 collectives with 1. Not yet implemented.

### Attempt 5: Parameter Server design
Root cause of RDMA failures: RDMABuffers created in child actor processes are invisible to the worker process's IbvManagerActor. The generator's RDMA manager can't negotiate with a buffer in a different process.

Solution: ParameterServerActor spawned in its own CPU process on the learner host.
- Owns a flat CPU buffer + RDMABuffer (in same process as RDMA manager)
- Learner ranks push their local shards via Monarch messages (intra-node, no FSDP collective)
- Generator reads the flat buffer via one RDMA read_into call
- No all-gather, no cross-process RDMA, no timeout risk
