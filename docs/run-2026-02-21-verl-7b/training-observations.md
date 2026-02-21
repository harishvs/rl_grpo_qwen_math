# veRL GRPO Training — Qwen2.5-7B on GSM8K

## Setup
- Model: Qwen/Qwen2.5-7B
- Dataset: GSM8K (7,473 train, 1,319 test)
- Infrastructure: 2x p4d.24xlarge (16x A100 40GB)
- Config: `config/verl/qwen-7b.yaml`
- Capacity block expires: 2026-02-21 11:30 UTC

## Config (key differences from 1.5B)
| Parameter | 1.5B | 7B |
|-----------|------|-----|
| train_batch_size | 256 | 128 |
| ppo_mini_batch_size | 128 | 64 |
| ppo_micro_batch_size_per_gpu | 16 | 4 |
| lr | 5e-6 | 1e-6 |
| gpu_memory_utilization | 0.5 | 0.4 |
| rollout log_prob micro_batch | 32 | 8 |
| ref log_prob micro_batch | 32 | 8 |

## Timeline
- **17:17 PST** — Launched via `./scripts/verl/run-training.sh --config qwen-7b.yaml`
- **17:20** — Worker pod stuck Pending (disk pressure taint on ip-10-0-1-161)
- **17:32** — Fixed: kubelet restart cleared stale disk pressure, but lost pause image
- **17:40** — Fixed: re-pulled pause image via SSM with instance role creds
- **17:42** — Worker running, veRL installed on both nodes
- **17:49** — Relaunched training (first attempt failed: only 8/16 GPUs visible)
- **17:52** — 7B model loaded (7.62B params, 4 shards), FSDP wrapped, vLLM initializing
- **17:55** — Initial validation running, all GPUs at 100%, ~28.7GB/40GB memory
- Head pod: `verl-grpo-head-zjv8s`

## Training Progress

58 total steps, ~505s/step → estimated ~8.1 hours total

| Step | Step Time | Notes |
|------|-----------|-------|
| 1 | 508s | First step |
| 2 | 505s | Consistent |

## GPU Memory
- All 8 GPUs at 100% utilization during generation
- ~28.7GB / 40GB used per GPU (FSDP actor + vLLM KV cache with TP=2)
- Detailed per-step metrics (reward, KL) not visible in stdout — swallowed by tqdm progress bar

## Observations

- 7B is ~7x slower per step than 1.5B (505s vs 70s)
- With batch_size=128 (half of 1.5B's 256), sequences/step = 1,024 vs 2,048
- Effective throughput: 1,024 seq / 505s ≈ 2 seq/s vs 1.5B's 29 seq/s
- Initial vLLM cache error with gpu_memory_utilization=0.4 — bumped to 0.5 with TP=2
- Disk pressure taint on worker node was stale (8% disk used) — kubelet restart fixed it but lost pause image
