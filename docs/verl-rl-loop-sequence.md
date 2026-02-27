# veRL RL Loop Sequence Diagram

```mermaid
sequenceDiagram
    participant Trainer
    participant Policy
    participant vLLM
    participant Ref
    participant Env
    participant GRPO

    Note over Trainer: Step begins (256 prompts)
    
    Note over vLLM: Phase 1 Generation (~19s)
    Note over vLLM: vLLM reshards Actor FSDP weights
    Trainer->>vLLM: Generate 8 completions per prompt
    Note over vLLM: 128 sequences per GPU, 2048 total
    vLLM-->>Trainer: Completions + old_log_probs

    Note over Ref: Phase 2 Reference Log Probs (~5s)
    Trainer->>Ref: Load from CPU
    Trainer->>Ref: Forward pass
    Ref-->>Trainer: ref_log_probs

    Note over Env: Phase 3 Reward Scoring (~0.6s)
    loop For each completion
        Trainer->>Env: Extract answer
        Env->>Env: Parse number after ####
        Env->>Env: Compare with ground truth
        Env-->>Trainer: reward (1.0 or 0.0)
    end

    Note over GRPO: Phase 4 Advantage Computation (~0.05s)
    Trainer->>GRPO: rewards (grouped by prompt)
    Note over GRPO: advantage = (reward - mean) / std
    GRPO-->>Trainer: advantages (2048 values)

    Note over Policy: Phase 5 Policy Update (~39s)
    Trainer->>Policy: Forward pass
    Policy-->>Trainer: new_log_probs
    Note over Trainer: Compute ratio and clipped loss
    Note over Trainer: Add KL penalty
    Trainer->>Policy: Backward pass
    Trainer->>Policy: Optimizer step (FSDP all-reduce)
    Note over Policy: Weights updated across 16 GPUs

    Note over Trainer: Step complete (~70s total)
```

## Phase Breakdown

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Generation | ~19s | vLLM generates 8 completions per prompt (2,048 sequences total) |
| 2. Reference Log Probs | ~5s | Reference model computes log probs for KL penalty |
| 3. Reward Scoring | ~0.6s | Binary correctness check (1.0 or 0.0) |
| 4. Advantage Computation | ~0.05s | Group-relative normalization (GRPO) |
| 5. Policy Update | ~39s | Forward + backward pass with clipped gradients |
| **Total** | **~70s** | **2,048 sequences processed** |

## Key Differences from Custom Trainer

- **Colocated architecture**: All components run on the same GPUs, switching phases
- **Zero-copy weight sync**: vLLM directly accesses Actor weights (no HTTP transfer)
- **Massive batch size**: 256 prompts × 8 completions = 2,048 sequences/step (vs 32 in custom)
- **Full reference model**: CPU-offloaded for proper KL computation (custom trainer skipped this)
- **Throughput**: 104,448 sequences/hour (45x faster than custom trainer)
