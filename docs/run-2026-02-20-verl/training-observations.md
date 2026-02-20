# Run 2026-02-20: veRL GRPO Training on 2x p4d.24xlarge

## Overview

Trained Qwen2.5-1.5B on GSM8K using veRL framework with GRPO. Completed 1 full epoch (29 steps) in 35 minutes. Model accuracy went from 14.5% → 77.0% on held-out test set.

## Infrastructure

- **Cluster**: rl-code-llm-training-dev, EKS 1.29, us-east-1
- **GPU Nodes**: 2x p4d.24xlarge (8x A100 40GB each, 16 GPUs total)
- **Orchestration**: KubeRay v1.3.0 (Terraform-managed)
- **Framework**: veRL 0.6.1 + vLLM 0.8.4
- **Image**: `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0`
- **Capacity Block**: cr-REDACTED (active until 2026-02-21 11:30 UTC)

## Training Config

```
algorithm: GRPO (adv_estimator=grpo)
model: Qwen/Qwen2.5-1.5B
train_batch_size: 256 prompts/step
rollout.n: 8 completions/prompt (2,048 sequences/step)
lr: 5e-6
kl_loss_coef: 0.1
kl_loss_type: low_var_kl
clip_ratio: 0.2
rollout.gpu_memory_utilization: 0.5
ref.fsdp_config.param_offload: True (CPU offload)
actor.fsdp_config.param_offload: False (stays on GPU)
total_epochs: 1
save_freq: 20
test_freq: 10
```

## Training Progress

![Training Progress](training_progress.png)

| Step | Reward (%) | KL | Step Time (s) |
|------|-----------|-----|---------------|
| 0 (val) | 2.3 | — | — |
| 1 | 1.1 | 0.0001 | 76 |
| 3 | 10.3 | 0.0007 | 73 |
| 5 | 18.4 | — | 71 |
| 7 | 30.5 | — | 70 |
| 10 | 44.5 | — | 69 |
| 14 | 59.8 | — | 69 |
| 16 | 66.3 | — | 71 |
| 19 | 68.0 | 0.0002 | 70 |
| 20 (val) | 72.3 | — | 69 |
| 25 | 65.0 | — | 70 |
| 29 (val) | 72.7 | — | 70 |

Total training time: 35 minutes (72.83s/step average).

## Step Timing Breakdown (per step)

| Phase | Time | Description |
|-------|------|-------------|
| Generation | ~19s | All 16 GPUs run vLLM engines in parallel, 128 sequences/GPU |
| Actor update | ~39s | FSDP training forward + backward + optimizer step |
| Old log probs | ~5.5s | Forward pass with current policy |
| Ref log probs | ~5s | FSDP ref model loaded from CPU, forward pass, offloaded back |
| Reward | ~0.6s | CPU-side string matching via compute_score() |
| Advantage | ~0.05s | GRPO group normalization |
| **Total** | **~70s** | |

## Eval Results

200 random GSM8K test problems, greedy decoding, chat template prompt matching training format:

| Model | Accuracy |
|-------|----------|
| Base Qwen2.5-1.5B | 29/200 = 14.5% |
| GRPO-trained (step 29) | 154/200 = **77.0%** |
| **Improvement** | **+62.5%** |

Note: Using a non-chat prompt format, the base model scores 56% (it can do math, just not in chat format). The GRPO training taught the model both the chat format and improved reasoning, scoring 77% with the chat prompt.

## Final Step Metrics (Step 29)

```
actor/entropy: 1.716
actor/pg_clipfrac: 0.0006
actor/kl_loss: 0.009
actor/grad_norm: 0.251
perf/max_memory_allocated_gb: 39.81
perf/max_memory_reserved_gb: 43.06
perf/cpu_memory_used_gb: 65.85
perf/throughput: 856.5 tokens/s
perf/total_num_tokens: 970,663
critic/score/mean: 0.650 (65% of sequences got correct answer)
response_length/mean: 381 tokens
response_length/max: 1024 tokens
response_length/clip_ratio: 0.171 (17% hit max length)
prompt_length/mean: 93 tokens
```

## Checkpoints

Saved to S3 via `kubectl exec tar | aws s3 cp -` (pods have no IRSA):

```
s3://rl-code-llm-training-dev-checkpoints/verl/run_2026_02_19/global_step_20.tar.gz (~9GB)
s3://rl-code-llm-training-dev-checkpoints/verl/run_2026_02_19/global_step_29.tar.gz (~9GB)
```

Checkpoints are FSDP-sharded as DTensors. Each checkpoint has 16 rank files (`model_world_size_16_rank_{0..15}.pt`), split across head pod (ranks 0-7) and worker pod (ranks 8-15). To reconstruct a HuggingFace model:

```python
import torch

shards = []
for rank in range(16):
    s = torch.load(f"actor/model_world_size_16_rank_{rank}.pt", map_location="cpu", weights_only=False)
    shards.append({k: v._local_tensor.clone() for k, v in s.items()})

# Get shard dimension from DTensor metadata
s0 = torch.load("actor/model_world_size_16_rank_0.pt", map_location="cpu", weights_only=False)
shard_dims = {k: v.placements[0].dim if hasattr(v.placements[0], 'dim') else None for k, v in s0.items()}

# Concatenate
full = {k: torch.cat([s[k] for s in shards], dim=shard_dims[k]) if shard_dims[k] is not None else shards[0][k] for k in shards[0]}
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in full.items()})
```

## Issues Encountered

### 1. veRL 0.7.0 incompatible with vLLM 0.8.4
`pip install verl` installed 0.7.0 which imports `vllm.v1.engine.utils` — doesn't exist in vLLM 0.8.4. Downgraded to `verl==0.6.1` on both head and worker pods.

### 2. Disk pressure from large image
The veRL image is ~20GB. Both GPU nodes hit disk pressure after pulling it. Resolved by pruning old trainer/vllm images via SSM: `crictl rmi --prune`.

### 3. FSDP2 not available
FSDP2 requires veRL 0.7+. Stuck on FSDP1 due to the vLLM compatibility constraint above.

### 4. Checkpoint reconstruction requires all 16 shards
Head pod only has ranks 0-7, worker has 8-15. Must copy shards between pods via `kubectl exec tar` before reconstruction. Direct `cat` pipe corrupts binary files — must use tar.

## Comparison vs Custom Trainer

| Metric | Custom Trainer (best) | veRL |
|--------|----------------------|------|
| Sequences/step | 32 | 2,048 |
| Step time | 50s | 70s |
| Throughput | 2,304 seq/hr | 104,448 seq/hr |
| Time for 1 epoch | ~26 hours | 35 minutes |
| Debug time | 12 hours (5 attempts) | Worked first try |
| Final accuracy | N/A (didn't finish) | 77.0% |

The 45x throughput improvement comes from:
- **Colocated vLLM** on all 16 GPUs (vs separate pod with HTTP sync)
- **256 prompts/step** (vs 4, limited by OOM in custom code)
- **Proper memory management** (phase-based GPU sharing, CPU offload for ref model)
- **No weight sync overhead** (zero-copy resharding vs 22s HTTP transfer)
