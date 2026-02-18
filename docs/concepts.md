# Key Concepts

## KL Divergence (Kullback-Leibler Divergence)

KL divergence measures how much one probability distribution differs from another. In RL fine-tuning, it measures how far the policy has drifted from the original model.

**Formula:**

```
KL(π_new || π_old) = Σ π_new(x) * log(π_new(x) / π_old(x))
```

π (pi) denotes a "policy" — the model's probability distribution over tokens. `π(token)` means the probability the model assigns to that token given the input (the softmax output). The Greek letter π is used because it's the standard notation in RL literature for a policy. Nothing to do with 3.14159.

**Properties:**
- Always ≥ 0
- KL = 0 means the distributions are identical
- Not symmetric: KL(A||B) ≠ KL(B||A)

**Why it matters in GRPO:**

Without a KL penalty, the model can "hack" the reward signal — finding degenerate outputs that score high but are nonsensical. The KL term keeps the model close to its original behavior:

```
total_loss = policy_loss + kl_coef * KL(π_new || π_old)
```

- `kl_coef` too low → model diverges, outputs degrade
- `kl_coef` too high → model barely changes, doesn't learn
- Typical range: 0.01 - 0.2

**In practice**, we approximate KL using only the sampled tokens rather than summing over the full vocabulary:

```python
approx_kl = (actor_log_probs - old_log_probs).mean()
```

This is a single-sample Monte Carlo estimate. It's noisy but unbiased over many steps.

## GRPO (Group Relative Policy Optimization)

GRPO is a reinforcement learning algorithm from the DeepSeek-R1 paper. It's a simplification of PPO that removes the need for a value/critic network.

**How it works:**

1. For each problem, generate G completions (a "group")
2. Score each completion (correct = 1.0, wrong = 0.0)
3. Compute advantages relative to the group mean:
   ```
   advantage_i = (reward_i - mean(rewards)) / std(rewards)
   ```
4. Update the policy to increase probability of above-average completions and decrease below-average ones

**Why "group relative":**

Instead of needing a learned value function to estimate "how good is this state?", GRPO uses the group average as the baseline. If 3 out of 8 completions are correct, those 3 get positive advantage and the other 5 get negative. The model learns by comparing against itself.

**Advantage over PPO:**
- No value network to train (saves memory, simpler code)
- Works well for tasks with clear binary/scalar rewards (math, code)
- Fewer hyperparameters to tune

## FSDP (Fully Sharded Data Parallelism)

FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU only holds 1/N of the model.

**Sharding strategies:**
- `FULL_SHARD` — shards params, grads, and optimizer states. Minimum memory, maximum communication.
- `SHARD_GRAD_OP` — shards grads and optimizer states only. More memory, less communication.
- `NO_SHARD` — equivalent to DDP. Each GPU holds full model.

**For our 1.5B model on 8x A10s:**
- Full model in bf16: ~3GB
- FSDP FULL_SHARD: ~400MB per GPU
- Leaves room for activations, gradients, and the generation model on rank 0

**Key gotcha:** FSDP flattens parameters internally into `_flat_param` tensors. Saving/loading checkpoints requires using `FSDP.state_dict_type(FULL_STATE_DICT)` to get standard parameter names back. See [lessons-learned.md](lessons-learned.md) for details on this bug.

## Policy Gradient with Clipping

The core update rule from PPO, also used in GRPO:

```python
ratio = exp(log_prob_new - log_prob_old)  # π_new(a|s) / π_old(a|s)
clipped_ratio = clamp(ratio, 1 - ε, 1 + ε)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

The clipping (ε = 0.2 typically) prevents the policy from changing too much in a single step. If the ratio goes beyond [0.8, 1.2], the gradient is zeroed out for that sample.

**Why clipping matters:**
- Without it, a single high-advantage sample could cause a huge policy shift
- Acts as a trust region — keeps updates conservative
- More stable than raw policy gradient, especially with noisy rewards

## Reward Model vs Environment Service

Two approaches to scoring completions:

**Reward model** — A learned neural network that predicts human preference. Used for open-ended tasks (chat, writing). Expensive to train, can be gamed.

**Environment service** (what we use) — Deterministic evaluation. For math: extract the numerical answer, compare to ground truth. Binary reward (1.0 or 0.0). Cheap, accurate, ungameable for well-defined tasks.

Our environment service:
1. Receives model completion
2. Extracts the final numerical answer
3. Compares to ground truth
4. Returns 1.0 (correct) or 0.0 (wrong)
