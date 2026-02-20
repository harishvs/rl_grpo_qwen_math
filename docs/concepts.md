# Key Concepts

## The Reinforcement Learning Loop

RL is a framework where an **agent** learns by interacting with an **environment**. Unlike supervised learning where you have labeled examples, in RL the model discovers what's good through trial and error.

**Core components:**

```
┌─────────┐   action (completion)   ┌─────────────┐
│  Agent   │ ─────────────────────► │ Environment │
│ (policy) │ ◄───────────────────── │  (scorer)   │
└─────────┘   reward (0.0 or 1.0)   └─────────────┘
```

- **Agent / Policy (π)** — The language model. Given a prompt (state), it produces a completion (action). The policy is the probability distribution over tokens that the model uses to generate text.
- **Environment** — Anything that evaluates the action and returns a reward. For us, it's the environment service that checks if the math answer is correct.
- **State** — The input to the agent. In our case, the math problem prompt.
- **Action** — The output from the agent. The full completion (reasoning chain + answer).
- **Reward** — A scalar signal from the environment. 1.0 if correct, 0.0 if wrong.

**The loop, step by step:**

```
1. Agent sees state         →  "Janet's ducks lay 16 eggs per day..."
2. Agent takes action       →  "Step 1: 16 - 3 - 4 = 9 eggs... Answer: 18"
3. Environment gives reward →  1.0 (correct!)
4. Agent updates policy     →  Increase probability of similar reasoning
5. Repeat from step 1 with new problem
```

**The goal:** Find a policy that maximizes expected reward over time. The agent doesn't memorize answers — it learns *how to reason* so it can solve new problems it hasn't seen.

**Why RL instead of supervised fine-tuning?**

In supervised fine-tuning (SFT), you need human-written solutions to train on. The model learns to imitate those specific solutions. In RL:
- The model discovers its own reasoning strategies
- It only needs a reward signal, not worked-out solutions
- It can find approaches that humans wouldn't write
- It optimizes for *correctness*, not for *similarity to a reference*

The tradeoff: RL is noisier and harder to train. The reward signal is sparse (just a number), so the model needs many attempts to figure out what works. That's why techniques like GRPO (grouping multiple attempts and comparing them) help — they give the model a richer learning signal from each batch.

**How our training loop maps to this:**

| RL Concept | Our Implementation |
|---|---|
| Agent | Qwen2.5-1.5B (actor model, FSDP-sharded) |
| Policy | The model's token probability distribution |
| State | GSM8K math problem prompt |
| Action | Generated completion (up to 512 tokens) |
| Environment | Environment service at `http://environment-service:8080` |
| Reward | 1.0 (correct answer) or 0.0 (wrong) |
| Policy update | GRPO with clipped gradients + KL penalty |

## GSM8K Dataset

GSM8K (Grade School Math 8K) is a dataset of 8,792 grade-school-level math word problems created by OpenAI. It's the standard benchmark for testing whether RL training improves a model's math reasoning.

**Structure:**
- 7,473 training problems, 1,319 test problems
- Each problem has a `question` (natural language word problem) and an `answer` (step-by-step solution ending with `#### <number>`)
- Problems require 2-8 steps of basic arithmetic (addition, subtraction, multiplication, division)
- No algebra, geometry, or advanced math — just multi-step reasoning with real-world scenarios

**Example 1 — Simple (2 steps):**
```
Q: Natalia sold clips to 48 of her friends in April, and then she sold
   half as many clips in May. How many clips did Natalia sell altogether
   in April and May?

A: Natalia sold 48/2 = 24 clips in May.
   Natalia sold 48+24 = 72 clips altogether in April and May.
   #### 72
```

**Example 2 — Medium (4 steps):**
```
Q: A craft store makes a third of its sales in the fabric section, a
   quarter of its sales in the jewelry section, and the rest in the
   stationery section. They made 36 sales today. How many sales were
   in the stationery section?

A: The craft store made 36 / 3 = 12 sales in the fabric section.
   It made 36 / 4 = 9 sales in the jewelry section.
   Thus, there were 36 - 12 - 9 = 15 sales in the stationery section.
   #### 15
```

**Example 3 — Harder (5 steps):**
```
Q: A family of 12 monkeys collected 10 piles of bananas. 6 piles had
   9 hands, with each hand having 14 bananas, while the remaining piles
   had 12 hands, with each hand having 9 bananas. How many bananas would
   each monkey get if they divide the bananas equally amongst themselves?

A: The first 6 bunches had 6 x 9 x 14 = 756 bananas.
   There were 10 - 6 = 4 remaining bunches.
   The 4 remaining bunches had 4 x 12 x 9 = 432 bananas.
   All together, there were 756 + 432 = 1188 bananas.
   Each monkey would get 1188/12 = 99 bananas.
   #### 99
```

**Why GSM8K for RL training:**
- Problems are simple enough that a 1.5B model can sometimes get them right, giving a non-zero reward signal to learn from
- But hard enough that the base model gets many wrong (~60-70% error rate), leaving room for improvement
- Binary correctness is easy to verify — just check if the extracted number matches `#### <answer>`
- Multi-step reasoning means the model must learn to chain operations, not just pattern-match

**In our setup**, we use 3,736 training problems (half the training set) for faster iteration. The environment service extracts the final number from the model's completion and compares it to the ground truth answer.

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

**In practice**, we approximate KL using the Schulman estimator, which is always non-negative:

```python
ratio = exp(new_log_probs - old_log_probs)  # π_new / π_old
approx_kl = ((ratio - 1) - log(ratio)).mean()
```

This works because `(x - 1) - log(x) ≥ 0` for all `x > 0`, with equality only at `x = 1` (identical distributions). Unlike the naive estimate `(new - old).mean()` which can go negative and accidentally reward divergence, the Schulman estimator is a proper non-negative KL approximation.

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
