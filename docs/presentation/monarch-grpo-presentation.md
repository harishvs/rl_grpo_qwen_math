---
title: "Teaching LLMs to Solve Math with Reinforcement Learning"
subtitle: "From RL Basics to Training Qwen on GSM8K"
author: "Harish Rao, SA"
date: "April 2026"
---

# Why RL for LLMs?

**Supervised fine-tuning**: Give the model question + correct answer → learns to copy

**RL training**: Give the model a question → it tries, you say right or wrong → it figures it out

Why this matters:
- No human-written solutions needed — just a way to check the answer
- The model discovers its own reasoning strategies
- Works for any task with a verifiable answer: math, code, logic puzzles

Our experiment: Qwen2.5-1.5B goes from **14.5% → 87%** on grade-school math using only "right or wrong" feedback

::: notes
The key insight of RL for LLMs: you don't need to show the model how to solve problems. You just need to tell it whether it got the right answer. For math, that's trivial — extract the number, check it. The model figures out its own chain-of-thought reasoning. This is the approach behind DeepSeek-R1 and similar recent breakthroughs. We replicated it on a small model to understand how it works end-to-end.
:::

# The RL Training Loop

```
For each batch of math problems:

  1. GENERATE — model writes 8 attempts per problem
     "Janet has 16 eggs..." → 8 different solutions

  2. SCORE — check each answer: correct = 1, wrong = 0
     3 out of 8 got the right number ✓

  3. COMPARE — within each group of 8:
     "These 3 were better than average → reinforce them"
     "These 5 were worse → discourage them"

  4. UPDATE — adjust model weights slightly
     Make the good reasoning patterns more likely
```

Repeat 233 times (one pass through the dataset). That's it.

::: notes
This is the complete algorithm at a high level. The model generates multiple attempts at each problem — we use 8. We score them with a binary reward. Then we compute advantages: how much better or worse was each attempt compared to the group average? The correct answers get positive advantages, wrong ones get negative. We use these advantages to update the model — increase the probability of token sequences that led to correct answers, decrease the ones that didn't. The clipping mechanism prevents any single update from changing the model too dramatically. After 233 steps through the 7,473 GSM8K training problems, the model has seen every problem once and improved from 14.5% to 87%.
:::

# Key Concepts: What Makes RL Work

**Policy** — the model's current strategy. Training changes it to produce better answers.

**Reward** — did you get the right number? 1 = yes, 0 = no. Only feedback the model gets.

**Advantage** — "how much better was this attempt vs the group average?"
- If 1 out of 8 is correct: that one gets a large positive signal
- If 7 out of 8 are correct: the wrong one gets a large negative signal
- If all 8 are the same: no learning signal (nothing to compare)

**Clipping** — safety rail. Prevents the model from changing too much in one step.

**KL Penalty** — rubber band pulling the model back toward its starting point. Prevents collapse.

::: notes
These five concepts are everything you need to understand RL training for LLMs. The policy is just the model's behavior. The reward is binary — beautifully simple for math. The advantage is the clever part of GRPO (the algorithm from DeepSeek-R1): instead of needing a separate "critic" network to estimate how good a state is, you just compare attempts within a group. The correct ones get reinforced, wrong ones get discouraged, and if all attempts have the same outcome there's nothing to learn from that problem. Clipping limits how much any single update can change the model — without it, one lucky answer could swing the model so hard it forgets everything else. The KL penalty measures total drift from the starting model — too much drift and the model collapses into repetitive outputs.
:::

# GRPO vs PPO: Why We Use GRPO

| | PPO (classic) | GRPO (DeepSeek-R1) |
|---|---|---|
| How it knows "better than expected" | Trains a separate critic network | Compares attempts within a group |
| Extra model needed | Yes (doubles GPU memory) | No |
| Advantage formula | Complex: GAE from critic | Simple: `(reward - mean) / std` |
| Attempts per problem | 1 | 8 (need a group to compare) |

GRPO trades more generation compute for simpler, more stable training. For tasks where checking answers is free (math, code), it's a clear win.

::: notes
PPO — Proximal Policy Optimization — is the classic RL algorithm. It trains a separate neural network called a critic that learns to predict "how good is this situation?" That prediction becomes the baseline for computing advantages. The problem: the critic is itself a large model that needs training, doubles your GPU memory, and can be a source of instability.

GRPO from DeepSeek throws out the critic entirely. Instead, for each question you generate a group of 8 answers, score them all, and compute each one's advantage relative to the group average. No critic to train, half the memory, simpler code. The tradeoff is you need 8 attempts per problem instead of 1, which costs more inference compute. But for math where checking the answer is just comparing two numbers, that's a great tradeoff.
:::

# The Dataset: GSM8K

**Grade School Math 8K**: 7,473 train / 1,319 test problems

**Example:**
> Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 each. How much does she make daily?

> 16 - 3 - 4 = 9 eggs. 9 × $2 = **$18** ← we only check this final number

- 2-8 step arithmetic reasoning
- The dataset has worked-out solutions — we throw them away
- The model only gets "right" or "wrong" feedback on the final number
- Base model accuracy: **14.5%** (mostly guessing)

::: notes
GSM8K is the standard benchmark for mathematical reasoning in LLMs. The problems are grade-school level — addition, subtraction, multiplication, division — but they require multi-step reasoning. The dataset comes with step-by-step solutions, but we don't use them at all. We only extract the final number after the #### marker and use it as ground truth for the binary reward. The model has to discover its own reasoning process. The base Qwen 1.5B model gets about 14.5% of these right — essentially random guessing for problems it hasn't memorized from pretraining.
:::

# What We Built: Three Approaches

All trained on the same model (Qwen2.5-1.5B), same data (GSM8K), same cluster (16× A100 GPUs):

**1. Custom Trainer** (from scratch)
- Hand-rolled GRPO loop: ~800 lines of Python
- Learned how every piece works. 5 attempts, 12 hours debugging.

**2. veRL** (ByteDance production framework)
- ~50 lines of config. Worked first try. 35 minutes to 77% accuracy.

**3. Monarch** (Meta's actor framework)
- New territory: no existing FSDP example. 12 training runs, 20+ issues fixed.
- 100 minutes to **87% accuracy**.

::: notes
We deliberately tried three approaches to understand the tradeoffs. The custom trainer was invaluable for learning — we hit every possible bug (KL explosion from sequence-level ratios, OOM from batch sizes, NCCL deadlocks between vLLM and FSDP). veRL by ByteDance is production-grade — it handles all the hard infrastructure problems and just works. Monarch from Meta was the research frontier — nobody had run FSDP training inside Monarch actors before. We filed detailed issues with the Monarch team and contributed back findings about EFA RDMA compatibility.
:::

# Results

| Approach | Final Accuracy | Training Time | Key Insight |
|---|---|---|---|
| Custom Trainer | ~60% reward | 26 hours | Learning exercise — every RL bug imaginable |
| veRL | **77%** | **35 min** | Production-ready, colocated architecture |
| Monarch | **87%** | 100 min | Matches quality, weight sync is the bottleneck |

![](monarch_run12_overview.png){ width=80% }

The reward curve shows steady learning: 14% → 50% in 10 steps, plateau at 80-90%.

::: notes
The custom trainer never finished a full epoch but we learned every detail of GRPO implementation. veRL reached 77% in 35 minutes — the gold standard. Monarch reached 87% in 100 minutes — actually higher accuracy, but 3x slower because of weight sync overhead between nodes. The reward curve is classic RL: rapid initial improvement as the model discovers basic arithmetic patterns, then a gradual climb as it learns more complex multi-step reasoning, and finally a plateau where most problems it can solve are solved. The 7B model reaches 88% in just 83 minutes on veRL — larger models learn faster and reach higher ceilings.
:::

# Why the Speed Difference?

The core infrastructure challenge: getting updated model weights from the **trainer** to the **generator**.

| | veRL | Monarch |
|---|---|---|
| Architecture | All GPUs switch roles | Separate nodes for each role |
| Weight sync | **Zero** — same GPUs, just reshard | 47 seconds per step (3GB over network) |
| GPU utilization | 16/16 | 5/16 |
| Steps per epoch | 29 | 233 |
| Time per step | 70s | 10s train + 47s sync |

veRL's trick: no data ever moves. The same GPU memory is viewed differently for training vs generation.

Monarch's advantage: generation and training can scale independently (matters for 70B+ models).

::: notes
The speed difference comes entirely from weight sync. veRL colocates everything on every GPU — after training, the weights are already there for generation, just resharded via a metadata operation. Monarch puts the trainer on one set of GPUs and the generator on another — clean separation, but 3GB of weights must transfer across the network every step. On our p4d instances, that's 47 seconds via serialized RPC. We tried RDMA to make this near-instant, but discovered p4d's EFA hardware (Nitro v3) doesn't support RDMA read/write — only newer P5 instances do. For small models, veRL's colocated approach wins. For very large models (70B+), Monarch's split placement lets you scale generation and training clusters independently, which is more cost-efficient at scale.
:::

# What We Learned

**About RL training:**
- Hyperparameters matter enormously — `kl_coef` being 100× wrong (0.1 vs 0.001) was the difference between 24% and 87%
- Weight sync every step is critical — stale weights corrupt the PPO importance ratio
- A frozen reference model stabilizes KL divergence

**About infrastructure:**
- EFA on p4d (Nitro v3) doesn't support RDMA read/write — only messaging
- NCCL works over EFA because it uses `fi_send/fi_recv`, not `fi_read`
- vLLM's `reload_weights` mutates internal state — breaks subsequent generation
- FSDP2's `named_parameters()` returns different keys than `state_dict()` — silent weight sync failures

**The takeaway:**
RL with a binary reward signal is enough to teach small models to reason. The algorithm is simple. The infrastructure is hard.

::: notes
Two categories of lessons. On the RL side: getting the hyperparameters right matters more than getting the infrastructure right. We spent days optimizing weight sync speed, but the 10x improvement came from changing one number in the config. On the infrastructure side: we discovered real limitations in AWS EFA on p4d instances, subtle API incompatibilities between FSDP2 and vLLM, and a behavior difference between named_parameters and state_dict that caused silent weight sync failures. Every one of these is documented in our repo with reproduction steps. The meta-lesson: RL training for LLMs is accessible — a 1.5B model on 16 GPUs for 35 minutes gets you from 14% to 77% accuracy on grade-school math. The hard part isn't the algorithm, it's the plumbing.
:::

# Resources

**Code**: `github.com/harishvs/rl_grpo_qwen_math`

Three implementations, all training logs, Terraform infrastructure, 12 run observations

**Branches:**
- `main` — Custom trainer + veRL
- `feat/monarch-grpo` — Monarch implementation (20+ issues documented)
- `feat/monarch-ofi-rdma` — EFA RDMA debugging

**Key docs in repo:**
- `docs/monarch-fsdp-issue.md` — FSDP + Monarch example request
- `docs/monarch-rdma-issue.md` — EFA RDMA root cause analysis
- `docs/run-2026-04-09-monarch-run12/` — Best training run (87% reward)

::: notes
Everything is open source. The custom trainer is about 800 lines of Python — great for understanding GRPO at every level. The veRL config is 50 lines. The Monarch implementation is the most interesting — we documented every issue across 12 training runs. The FSDP issue doc has been shared with the Monarch team at Meta as a feature request for an official FSDP training example. The RDMA issue doc traces the root cause from "it doesn't work" through 6 wheel versions to "p4d is Nitro v3 which doesn't support fi_read." Feel free to use any of this for your own RL training experiments.
:::
