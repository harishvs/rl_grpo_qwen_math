# GRPO Training for Math Reasoning LLMs

This project implements **Group Relative Policy Optimization (GRPO)** to train language models on mathematical reasoning tasks using reinforcement learning. It's based on the DeepSeek-R1 approach for training reasoning models.

## What is GRPO?

GRPO is a reinforcement learning algorithm that improves upon PPO (Proximal Policy Optimization) for language model training:

1. **Generate multiple responses** (a "group") for each math problem
2. **Score each response** using an environment service that checks correctness
3. **Compute advantages** by comparing each response to the group average (no value network needed)
4. **Update the policy** using clipped gradients to prevent too-large updates

This is simpler than PPO because it doesn't require training a separate value/critic network.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EKS Cluster                                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    GPU Node (g5.48xlarge)                     │   │
│  │                         8x A10 GPUs                           │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │              GRPO Trainer Job (torchrun 8 procs)         │ │   │
│  │  │                                                          │ │   │
│  │  │  ┌─────────────────────┐  ┌─────────────────────────┐   │ │   │
│  │  │  │   Actor Model       │  │   Generation Model      │   │ │   │
│  │  │  │   (FSDP sharded)    │  │   (rank 0 only)         │   │ │   │
│  │  │  │                     │  │                         │   │ │   │
│  │  │  │  - Policy training  │  │  - HuggingFace generate │   │ │   │
│  │  │  │  - Gradient updates │  │  - Synced from actor    │   │ │   │
│  │  │  │  - 8-way sharded    │  │  - Full model ~3GB      │   │ │   │
│  │  │  └─────────────────────┘  └─────────────────────────┘   │ │   │
│  │  │                                                          │ │   │
│  │  └──────────────────────────┬───────────────────────────────┘ │   │
│  │                             │                                  │   │
│  └─────────────────────────────┼──────────────────────────────────┘   │
│                                │                                      │
│  ┌─────────────────────────────┼──────────────────────────────────┐   │
│  │                    CPU Node │                                   │   │
│  │                             ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              Environment Service                         │   │   │
│  │  │                                                          │   │   │
│  │  │  - Receives model completions                            │   │   │
│  │  │  - Extracts final answers                                │   │   │
│  │  │  - Compares to ground truth                              │   │   │
│  │  │  - Returns reward (1.0 = correct, 0.0 = wrong)          │   │   │
│  │  │                                                          │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Loop

Each training step:

```
1. Sample batch of math problems from GSM8K dataset
   └── "Janet's ducks lay 16 eggs per day..."

2. Generate G completions per problem using vLLM (group_size=8)
   └── 8 different reasoning chains for each problem

3. Send completions to Environment Service for scoring
   └── Returns rewards: [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

4. Compute GRPO advantages (group-normalized)
   └── advantage = (reward - group_mean) / group_std
   └── Good answers get positive advantage, bad ones negative

5. Compute policy gradient with KL penalty
   └── loss = -advantage * log_prob + kl_coef * KL(actor || reference)

6. Update actor model weights
   └── Gradient descent step with clipping
```

## Components

### Trainer (`src/trainer/`)

- **`trainer.py`** - Main GRPO training orchestration
  - `GRPOTrainer` - Coordinates the training loop
  - `ActorModel` - The policy being trained (Qwen2.5-1.5B), FSDP-sharded across 8 GPUs
  - `RolloutEngine` - HuggingFace generate-based text generation (separate model on rank 0)

- **`grpo.py`** - GRPO advantage computation
  - Group-normalized advantages without value network

- **`environment_client.py`** - HTTP client for reward service

- **`main.py`** - Entry point, loads GSM8K dataset, distributed training setup

### Environment Service (`src/environment/`)

- FastAPI service that evaluates math answers
- Extracts numerical answers from model completions
- Compares to ground truth, returns binary reward

## Key Files

```
src/
├── trainer/
│   ├── trainer.py      # GRPO training logic
│   ├── grpo.py         # Advantage computation
│   ├── config.py       # Training hyperparameters
│   ├── main.py         # Entry point
│   └── environment_client.py
└── environment/
    └── service.py      # Reward computation service

k8s/
├── trainer/
│   ├── job.yaml        # GPU training job
│   └── serviceaccount.yaml
├── environment/
│   └── deployment.yaml # Reward service deployment
└── config/
    └── training-config.yaml  # Hyperparameters

docker/
├── trainer/
│   ├── Dockerfile
│   └── requirements-trainer.txt
└── environment/
    └── Dockerfile
```

## Running Training

### Prerequisites

- EKS cluster with GPU nodes (g5.48xlarge recommended - 8x A10 GPUs, 24GB each)
- ECR repositories for Docker images
- Configured kubectl access

### Deploy

```bash
# Build and push images
./scripts/build-and-push-images.sh all

# Apply Kubernetes manifests
kubectl apply -f k8s/config/
kubectl apply -f k8s/environment/
kubectl apply -f k8s/trainer/

# Monitor training
kubectl logs -f -l job-name=grpo-trainer
```

### Configuration

Edit `k8s/config/training-config.yaml`:

```yaml
data:
  model_name: "Qwen/Qwen2.5-1.5B"
  batch_size: "4"         # See memory notes below
  num_epochs: "1"
  group_size: "2"         # Completions per problem
  learning_rate: "1e-6"
  kl_coef: "0.1"          # KL penalty weight
  clip_range: "0.2"       # PPO-style clipping
  max_samples: "3736"     # Half of GSM8K for faster iteration
```

## Memory Management & OOM Issues

Training GRPO with FSDP across 8 GPUs required careful memory tuning. Here's what we learned:

### GPU Memory Layout (per A10 24GB)

With our current setup, each GPU holds:
- **FSDP-sharded actor model**: ~400MB per GPU (1.5B params / 8 GPUs, bf16)
- **Generation model** (rank 0 only): ~3GB (full model for inference)
- **Activations & gradients**: Variable based on batch size and sequence length

### OOM Journey

| Batch Size | Group Size | Result |
|------------|------------|--------|
| 32 | 8 | OOM immediately |
| 8 | 4 | OOM during backward pass |
| 8 | 2 | OOM during backward pass |
| 4 | 2 | ✅ Works stable |
| 2 | 2 | ✅ Works (slower) |

### Why batch_size=4 is the sweet spot

1. **Rank 0 has extra load**: It holds both the FSDP shard AND the full generation model (~3GB extra)
2. **Gradient checkpointing helps but isn't enough**: We enable it, but large batches still OOM
3. **Generation is memory-hungry**: Each completion stores scores for log prob computation
4. **FSDP summon_full_params**: During weight sync, temporarily materializes full model

### Memory Optimization Techniques Used

```python
# 1. Gradient checkpointing
self._model.gradient_checkpointing_enable()

# 2. bf16 mixed precision
torch_dtype=torch.bfloat16

# 3. FSDP with FULL_SHARD strategy
sharding_strategy=ShardingStrategy.FULL_SHARD

# 4. Separate generation model (avoids summon_full_params during generate)
# Only rank 0 loads it, other ranks just receive broadcast rollouts

# 5. No reference model - use old_log_probs from generation time instead
```

### If You Hit OOM

1. Reduce `batch_size` first (most impact)
2. Reduce `group_size` (fewer completions per problem)
3. Reduce `max_new_tokens` (shorter generations)
4. Try `SHARD_GRAD_OP` instead of `FULL_SHARD` (trades memory for communication)

### Training Time Estimates

With batch_size=4, group_size=2, 3736 samples, 1 epoch:
- Steps: 934
- Time per step: ~85 seconds
- Total: ~22 hours on 8x A10 GPUs

## Why This Architecture?

1. **Separate Environment Service**: Decouples reward computation from training. Can scale independently, easier to test, and allows swapping reward functions.

2. **FSDP for Distributed Training**: Shards model parameters across 8 GPUs, enabling training of larger models than would fit on a single GPU.

3. **Separate Generation Model**: Avoids FSDP complexity during generation. Only rank 0 loads a full model copy for inference, then broadcasts rollouts to all ranks.

4. **No Reference Model**: Instead of maintaining a frozen reference model for KL divergence, we use the log probs computed during generation as "old_log_probs". Saves ~3GB GPU memory.

5. **GRPO over PPO**: No value network to train = simpler, fewer hyperparameters, works well for reasoning tasks.

## Metrics

Training logs show:
- `loss` - Policy gradient loss + KL penalty
- `reward` - Mean reward across batch (0-1 for binary)
- `kl` - KL divergence from reference policy

Good training shows:
- Increasing mean reward over time
- Stable/slowly increasing KL (not exploding)
- Decreasing loss

## How to Know if RL Training Worked

### During Training: Watch the Metrics

```bash
kubectl logs -f -l job-name=grpo-trainer
```

Look for:
```
Step 1: loss=0.0012, reward=0.15, kl=-0.01
Step 100: loss=-0.0023, reward=0.35, kl=0.02
Step 500: loss=-0.0045, reward=0.52, kl=0.05
```

**Signs of successful training:**
- `reward` trending upward (model getting more answers correct)
- `loss` becoming more negative (policy improving)
- `kl` staying small and stable (not diverging too far from initial policy)

**Red flags:**
- `reward` stuck at 0 or not improving → model not learning
- `kl` exploding (>1.0) → policy diverging, reduce learning rate
- `loss` = 0 constantly → gradient flow issue

### After Training: Evaluate on Test Set

The GSM8K dataset has a held-out test set (1,319 problems). Compare before/after:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model (before training)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")

# Load fine-tuned model (after training)
trained_model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint/final")

# Evaluate both on test set
def evaluate_accuracy(model, tokenizer, test_problems):
    correct = 0
    for problem in test_problems:
        prompt = problem.to_prompt()
        output = model.generate(tokenizer(prompt, return_tensors="pt").input_ids, max_new_tokens=512)
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract answer and compare to ground truth
        predicted = extract_answer(completion)
        if predicted == problem.answer:
            correct += 1
    
    return correct / len(test_problems)

base_accuracy = evaluate_accuracy(base_model, tokenizer, test_set)
trained_accuracy = evaluate_accuracy(trained_model, tokenizer, test_set)

print(f"Base model: {base_accuracy:.1%}")
print(f"Trained model: {trained_accuracy:.1%}")
print(f"Improvement: {trained_accuracy - base_accuracy:+.1%}")
```

### Expected Results

For Qwen2.5-1.5B on GSM8K:
- **Base model**: ~30-40% accuracy (varies by prompting)
- **After GRPO training**: ~50-60% accuracy (with good hyperparameters)
- **State-of-the-art**: ~90%+ (larger models, more training)

### Quick Sanity Check

Test a few problems manually:

```python
prompt = """Solve this math problem step by step:

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Answer:"""

# Generate with trained model
output = trained_model.generate(...)
print(output)
# Should show step-by-step reasoning ending with "18" (correct answer)
```

If the trained model:
1. Shows clearer step-by-step reasoning
2. Gets more answers correct
3. Makes fewer arithmetic errors

Then RL training worked!

## References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - GRPO algorithm
- [GSM8K Dataset](https://github.com/openai/grade-school-math) - Math problems
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference engine
