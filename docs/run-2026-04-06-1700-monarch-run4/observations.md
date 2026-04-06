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
