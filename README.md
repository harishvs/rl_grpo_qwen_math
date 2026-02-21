# GRPO Training for Math Reasoning LLMs

This project trains language models on mathematical reasoning (GSM8K) using **Group Relative Policy Optimization (GRPO)** — the RL algorithm from DeepSeek-R1. It contains two implementations:

1. **Custom GRPO trainer** (`src/trainer/`) — hand-rolled training loop, useful for learning how GRPO works end-to-end
2. **veRL-based trainer** (`k8s/verl/`) — production framework by ByteDance, 40x faster throughput

Both run on EKS with multi-node GPU clusters.

## Results

Qwen2.5-1.5B trained on GSM8K (7,473 problems, 1 epoch) using veRL on 2x p4d.24xlarge (16x A100 40GB):

| Metric | Value |
|--------|-------|
| Training time | 35 minutes |
| Base model accuracy | 14.5% |
| Trained model accuracy | **77.0%** |
| Improvement | **+62.5%** |

## What is GRPO?

GRPO (Group Relative Policy Optimization) is an RL algorithm from the DeepSeek-R1 paper. It simplifies PPO by removing the need for a value/critic network.

**The RL loop:**

```
┌─────────┐   action (completion)   ┌─────────────┐
│  Agent   │ ─────────────────────► │ Environment │
│ (policy) │ ◄───────────────────── │  (scorer)   │
└─────────┘   reward (0.0 or 1.0)   └─────────────┘
```

- **Agent/Policy (π)** — The language model. Given a prompt, it produces a completion. π is the probability distribution over tokens.
- **Environment** — Evaluates the completion and returns a reward. For math: extract the number, check if it matches the answer.
- **Reward** — 1.0 if correct, 0.0 if wrong. No learned reward model needed for well-defined tasks like math.

**Why RL instead of supervised fine-tuning?** SFT requires human-written solutions and the model learns to imitate them. With RL, the model discovers its own reasoning strategies — it only needs a reward signal, not worked-out solutions. It optimizes for *correctness*, not *similarity to a reference*.

**How GRPO works:**

1. For each problem, generate G completions (a "group")
2. Score each completion (correct = 1.0, wrong = 0.0)
3. Compute advantages relative to the group mean — no value network needed:
   ```
   advantage_i = (reward_i - mean(rewards)) / std(rewards)
   ```
4. Update the policy using clipped gradients to prevent too-large updates:
   ```python
   ratio = exp(log_prob_new - log_prob_old)  # π_new(a|s) / π_old(a|s)
   clipped_ratio = clamp(ratio, 1 - ε, 1 + ε)  # ε = 0.2
   loss = -min(ratio * advantage, clipped_ratio * advantage)
   ```

If 3 out of 8 completions are correct, those 3 get positive advantage and the other 5 get negative. The model learns by comparing against itself.

**KL penalty** keeps the model from drifting too far from the original:

```
total_loss = policy_loss + kl_coef * KL(π_new || π_old)
```

We use the Schulman estimator for KL, which is always non-negative: `((ratio - 1) - log(ratio)).mean()`. Without this, the model can "hack" the reward signal — finding degenerate outputs that score high but are nonsensical.

## GSM8K Dataset

GSM8K (Grade School Math 8K) is 8,792 grade-school math word problems by OpenAI — the standard benchmark for math reasoning.

- 7,473 training problems, 1,319 test problems
- Each problem requires 2-8 steps of basic arithmetic
- Answer format: step-by-step reasoning ending with `#### <number>`

**Example:**
```
Q: A craft store makes a third of its sales in the fabric section, a quarter
   in jewelry, and the rest in stationery. They made 36 sales today.
   How many sales were in the stationery section?

A: 36 / 3 = 12 fabric sales. 36 / 4 = 9 jewelry sales.
   36 - 12 - 9 = 15 stationery sales.
   #### 15
```

**Why GSM8K for RL:** Problems are simple enough that a 1.5B model sometimes gets them right (giving a non-zero reward signal), but hard enough that the base model gets ~85% wrong, leaving room for improvement. Binary correctness is trivial to verify.

## FSDP (Fully Sharded Data Parallelism)

FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds 1/N of the model.

- `FULL_SHARD` — shards params, grads, and optimizer states. Minimum memory, maximum communication.
- Our 1.5B model in bf16: ~3GB total → ~200MB per GPU with 16-way sharding

## Training Loop

```
1. Sample batch of math problems from GSM8K
2. Generate G completions per problem (group_size=8)
3. Score completions (binary: correct=1.0, wrong=0.0)
4. Compute GRPO advantages: (reward - group_mean) / group_std
5. Compute policy gradient with KL penalty against reference model
6. Update actor model weights with clipped gradients
```

---

## Approach 1: Custom GRPO Trainer (from scratch)

A hand-rolled implementation to understand every piece of the GRPO pipeline. This was the first approach attempted and went through 5 iterations of debugging and optimization.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       EKS Cluster                             │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Trainer Job (torchrun, 16 GPUs across 2 nodes)         │ │
│  │                                                          │ │
│  │  Actor Model (FSDP-sharded)    Generation Model (rank 0) │ │
│  │  - Policy training             - HuggingFace generate    │ │
│  │  - Gradient updates            - Synced from actor       │ │
│  │  - 16-way sharded              - Full model ~3GB         │ │
│  └──────────────────────┬───────────────────────────────────┘ │
│                         │                                     │
│  ┌──────────────────────▼───────────────────────────────────┐ │
│  │  Environment Service (CPU node)                           │ │
│  │  - Evaluates math answers via FastAPI                     │ │
│  │  - Returns binary reward                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Components

- `src/custom/trainer/trainer.py` — GRPO training loop, FSDP actor model, rollout engine
- `src/custom/trainer/grpo.py` — Group-normalized advantage computation
- `src/custom/trainer/main.py` — Entry point, dataset loading, distributed setup
- `src/custom/environment/service.py` — FastAPI reward service

### What We Learned (5 attempts)

**Attempt 1** — batch_size=8, OOM immediately. Rank 0 holds FSDP shard + full generation model.

**Attempt 2** — batch_size=4, KL exploded to 194.8 on step 2. Root cause: computing importance ratio on sequence-level summed log probs. `exp(sum of 200 token shifts)` → exponential blowup.

**Attempt 3** — Per-token importance ratio fix. KL stable at 0.0004–0.0009. But 5 min/step (sequential forward passes), 12 steps/hr. At 1,868 total steps, would take 156 hours.

**Attempt 4** — Batched HF generate with `num_return_sequences`. 1.5 min/step, 40 steps/hr. 3x speedup. But still 47 hours for 1 epoch.

**Attempt 5** — vLLM as separate server pod + HTTP weight sync. 63s/step, 57 steps/hr. Hit checkpoint save crash at step 50 (local_rank vs global rank bug). After batched log_probs optimization: 50s/step, 72 steps/hr. Still 26 hours for 1 epoch.

### Key Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Sequence-level importance ratio | KL explosion (0.4 → 194.8 in 1 step) | Per-token ratio, then average |
| `log_probs.mean()` | Wrong gradient signal | `log_probs.sum()` per sequence |
| Checkpoint `local_rank == 0` | Crash on multi-node (rank 7 got empty dict) | `rank == 0` (global) |
| vLLM inside torchrun | NCCL/CUDA deadlock | Separate vLLM server pod |
| Sequential forward passes | 32 passes × 16-rank all-gather per step | Batched with padding |

### Performance

| Config | Step Time | Steps/hr | Time for 1 Epoch | Sequences/Step |
|--------|-----------|----------|-------------------|----------------|
| Sequential generate | 5 min | 12 | 156 hours | 32 |
| Batched HF generate | 1.5 min | 40 | 47 hours | 32 |
| vLLM server | 63s | 57 | 33 hours | 32 |
| vLLM + batched log_probs | 50s | 72 | 26 hours | 32 |

After 49 steps (before crash), reward was oscillating 0.31–0.56 with no clear upward trend. The small batch size (4 prompts × 8 completions = 32 sequences/step) meant high variance and slow learning.

### Running the Custom Trainer

```bash
# Build and push images
./scripts/custom/build-and-push-images.sh all

# Deploy
kubectl apply -f k8s/custom/config/
kubectl apply -f k8s/custom/environment/
kubectl apply -f k8s/custom/trainer/

# Monitor
kubectl logs -f -l job-name=grpo-trainer
```

---

## Approach 2: veRL Framework (production)

After spending ~12 hours debugging the custom trainer to reach 72 steps/hr with 32 sequences/step, we switched to [veRL](https://github.com/volcengine/verl) — a production RL post-training framework by ByteDance. The difference was dramatic.

### Why veRL?

The custom trainer had fundamental architectural limitations:

| Problem | Custom Trainer | veRL |
|---------|---------------|------|
| Generation | Separate vLLM pod, HTTP weight sync (22s overhead) | Colocated vLLM on all GPUs, zero-copy weight resharding |
| Batch size | 4 prompts/step (OOM with more) | 256 prompts/step (proper memory management) |
| Sequences/step | 32 | 2,048 (64x more) |
| Reference model | Skipped (used old log probs) | Full ref model with CPU offload |
| Step time | 50s for 32 sequences | 70s for 2,048 sequences |
| Effective throughput | 32 seq × 72 steps/hr = 2,304 seq/hr | 2,048 seq × 51 steps/hr = **104,448 seq/hr** |

veRL processes **45x more data per unit time** than the custom trainer.

### Architecture

veRL uses a colocated architecture — every GPU runs all roles (actor, rollout, ref) by switching between phases:

```
┌──────────────────────────────────────────────────────────────┐
│  RayCluster on EKS (2x p4d.24xlarge, 16x A100 40GB)         │
│                                                               │
│  Each GPU runs a WorkerDict that cycles through:              │
│                                                               │
│  1. Generation (~19s)  — vLLM engine, 128 sequences/GPU      │
│  2. Ref log probs (~5s) — FSDP ref model loaded from CPU     │
│  3. Reward (~0.6s)     — CPU-side string matching             │
│  4. Advantage (~0.05s) — GRPO group normalization             │
│  5. Actor update (~39s) — FSDP training forward+backward      │
│                                                               │
│  Total: ~70s/step for 2,048 sequences                         │
└──────────────────────────────────────────────────────────────┘
```

### Training Config

```bash
# Model
model: Qwen/Qwen2.5-1.5B

# Data
train_batch_size: 256        # prompts per step
rollout.n: 8                 # completions per prompt (256 × 8 = 2,048 sequences/step)

# Optimization
lr: 5e-6
kl_loss_coef: 0.1
clip_ratio: 0.2

# Infrastructure
n_gpus_per_node: 8
nnodes: 2                   # 16 GPUs total
rollout.gpu_memory_utilization: 0.5
ref.fsdp_config.param_offload: True   # ref model on CPU
```

### Training Progression

29 steps, 35 minutes, 1 full epoch:

| Step | Reward | KL | Step Time |
|------|--------|-----|-----------|
| 0 (val) | 2.3% | — | — |
| 1 | 1.1% | 0.0001 | 76s |
| 3 | 10.3% | 0.0007 | 73s |
| 14 | 59.8% | — | 69s |
| 20 (val) | 72.3% | — | 69s |
| 29 (val) | **72.7%** | — | 70s |

### Eval Results (200 GSM8K test problems, greedy decoding)

| Model | Accuracy |
|-------|----------|
| Base Qwen2.5-1.5B | 14.5% |
| GRPO-trained (step 29) | **77.0%** |

### Running veRL

```bash
# Train Qwen2.5-1.5B (default)
./scripts/verl/run-training.sh

# Train Qwen2.5-7B
./scripts/verl/run-training.sh --config qwen-7b.yaml

# With log following
./scripts/verl/run-training.sh --config qwen-7b.yaml --follow
```

### Checkpoints

Saved to S3 (pods have no IRSA, streamed via local pipe):
```
s3://rl-code-llm-training-dev-checkpoints/verl/run_2026_02_19/global_step_20.tar.gz
s3://rl-code-llm-training-dev-checkpoints/verl/run_2026_02_19/global_step_29.tar.gz
```

Checkpoints are FSDP-sharded (16 rank files as DTensors). To reconstruct a HuggingFace model, load all 16 `model_world_size_16_rank_*.pt` files and concatenate along `Shard(dim=0)`.

---

## Side-by-Side Comparison

| | Custom Trainer | veRL |
|---|---|---|
| Lines of training code | ~800 (trainer.py + grpo.py) | ~50 (config + reward function) |
| Debug iterations | 5 attempts over 12 hours | Worked on first try |
| Sequences per step | 32 | 2,048 |
| Step time | 50s | 70s |
| Effective throughput | 2,304 seq/hr | 104,448 seq/hr |
| Time for 1 epoch (GSM8K) | ~26 hours | **35 minutes** |
| Final reward after 1 epoch | N/A (didn't finish) | 72.7% |
| Weight sync overhead | 22s HTTP transfer every 3 steps | 0s (colocated) |
| Reference model | Skipped | Full, CPU-offloaded |
| Generation engine | vLLM (separate pod) | vLLM (colocated, all GPUs) |

The custom trainer was valuable for understanding GRPO internals — per-token importance ratios, KL estimation, FSDP weight management, vLLM integration challenges. But for actual training, veRL is the right tool.

## Observability

Training metrics are exported to Prometheus and visualized in Grafana, both running on CPU nodes so they persist independently of GPU training pods.

**Stack:** kube-prometheus-stack (Helm) → Prometheus scrapes veRL head pod → Grafana dashboards

**Metrics exported:** Reward, KL divergence, policy loss, entropy, gradient norm, step time, GPU memory, throughput, validation reward.

**How it works:**
- A sidecar script (`cloudwatch_metrics.py` in the configmap) runs on the head pod, tails the veRL worker log, and exposes metrics via `prometheus_client` on port 9090
- A `PodMonitor` (`k8s/verl/podmonitor.yaml`) tells Prometheus to scrape the head pod
- A Grafana dashboard (`k8s/verl/grafana-dashboard.yaml`) is auto-provisioned via ConfigMap

**Access Grafana:**
```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Open http://localhost:3000 — admin / grpo-training
```

## Infrastructure

```
EKS Cluster: rl-code-llm-training-dev (us-east-1, EKS 1.29)
GPU Nodes:   2x p4d.24xlarge (8x A100 40GB each)
CPU Nodes:   2x AL2023_x86_64_STANDARD
KubeRay:     v1.3.0 (Terraform-managed)
FSx Lustre:  1.2TB SCRATCH_2, version 2.15
Storage:     S3 for checkpoints
```

All infrastructure managed via Terraform (`terraform/environments/dev/`).

## Project Structure

```
src/
├── custom/                     # Approach 1: From-scratch GRPO trainer
│   ├── trainer/
│   │   ├── trainer.py          # Training loop, FSDP actor, rollout engine
│   │   ├── grpo.py             # Advantage computation
│   │   ├── config.py           # Hyperparameters
│   │   ├── main.py             # Entry point
│   │   └── environment_client.py
│   ├── environment/
│   │   └── service.py          # FastAPI reward service
│   └── vllm_server/
│       └── server.py           # Standalone vLLM server for generation
└── verl/                       # Approach 2: veRL framework
    ├── prep_data.py            # GSM8K → parquet
    ├── reward.py               # Binary reward function
    └── run_grpo.sh             # Training launch script

scripts/
├── custom/                     # Scripts for from-scratch trainer
│   ├── build-and-push-images.sh
│   ├── start-training.sh
│   └── ...
└── verl/
    └── run-training.sh         # Deploy RayCluster + launch veRL training

k8s/
├── custom/                     # K8s manifests for from-scratch trainer
│   ├── trainer/                # GPU training job (torchrun IndexedJob)
│   ├── environment/            # Reward service deployment
│   ├── vllm-server/            # Standalone vLLM server
│   ├── config/                 # Training hyperparameters
│   └── monitoring/             # CloudWatch agent
└── verl/                       # K8s manifests for veRL
    ├── raycluster.yaml         # RayCluster for 2x p4d.24xlarge
    ├── configmap.yaml          # Data prep, reward function, training script
    ├── podmonitor.yaml         # Prometheus scrape target
    ├── grafana-dashboard.yaml  # Auto-provisioned Grafana dashboard
    ├── storageclass.yaml       # gp3 EBS storage class
    └── pause-image-daemonset.yaml  # Auto-fix pause image on GPU nodes

docker/
├── custom/                     # Dockerfiles for from-scratch trainer
│   ├── trainer/
│   ├── environment/
│   └── vllm-server/
└── verl/                       # Dockerfile for veRL (unused, using base image)

terraform/                      # All infrastructure
├── environments/dev/
│   └── main.tf                 # EKS, VPC, FSx, KubeRay, IAM
└── modules/

docs/
├── run-2026-02-19/             # Custom trainer observations (5 attempts)
└── run-2026-02-20-verl/        # veRL training observations + results
```

## References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) — GRPO algorithm
- [veRL](https://github.com/volcengine/verl) — Production RL post-training framework
- [GSM8K Dataset](https://github.com/openai/grade-school-math) — Math problems
- [vLLM](https://github.com/vllm-project/vllm) — Fast inference engine
