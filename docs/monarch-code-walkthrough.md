# Monarch GRPO Implementation -- Code Walkthrough

This doc explains every file in the Monarch implementation, the reasoning behind each design choice, and the non-obvious trade-offs.

---

## 1. Shared Modules (`src/shared/`)

These files contain code used by all three implementations (custom, veRL, Monarch). They were extracted from the custom and veRL implementations to eliminate duplication.

### `src/shared/reward.py`

**What it does**: Takes a model completion string and a ground truth answer, returns 1.0 (correct) or 0.0 (wrong).

**How it works**: Regex searches for `#### <number>` in the completion (GSM8K answer format). Strips commas, compares as floats. Falls back to string comparison if float parsing fails.

**Why binary reward?** GRPO doesn't need a learned reward model for well-defined tasks like math. The answer is either right or wrong. Partial credit would complicate the advantage computation without improving training (tested by DeepSeek).

**Why regex and not just string matching?** The model might generate "#### 1,234" or "#### 1234.0" -- need to normalize before comparing.

### `src/shared/grpo.py`

**What it does**: Computes GRPO advantages -- how much better each completion was compared to others in its group.

**The formula**: For each group of G=8 completions from the same prompt:
```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

**Why group normalization?** This is the core GRPO insight from DeepSeek-R1. Instead of a value network (PPO), you compare completions against each other. A correct answer in a group where everything was correct gets advantage ≈ 0 (nothing special). A correct answer in a group where most were wrong gets a high advantage (reinforce this behavior).

**The zero-std edge case**: When all rewards in a group are identical (all correct or all wrong), std=0 would cause division by zero. We set advantages to 0 in this case -- there's nothing to learn when all outcomes are the same.

**Why two functions?** `compute_grpo_advantages()` works on flat tensors (used during training). `compute_grpo_advantages_from_list()` works on nested lists (used for debugging/testing).

### `src/shared/dataset.py`

**What it does**: Loads GSM8K math problems from HuggingFace or local JSON.

**`to_prompt()` vs `to_chat_prompt()`**: The custom trainer uses a simple prompt format ("Solve the following..."). The veRL implementation uses a chat format with the instruction to output the answer after "####". The chat format reached 77% accuracy, so we use that for Monarch too. Both formats are available since different implementations need different ones.

**Why `_extract_gsm8k_answer()`?** GSM8K's answer field contains the full solution with the final answer after "####". We need just the number for reward computation.

---

## 2. Monarch Config (`src/monarch/config.py`)

**What it does**: Defines all training hyperparameters as Python dataclasses, loadable from YAML.

**Why dataclasses not a dict?** Type safety and IDE autocomplete. `config.learner.lr` is clearer than `config["learner"]["lr"]` and catches typos at load time.

**Why separate sections (LearnerConfig, GeneratorConfig, etc.)?** Mirrors the architecture -- each actor gets its own config section. Easy to see what parameters affect what component.

**The type coercion fix**: YAML loads `1e-6` as a string, not a float. The loader checks the dataclass field type and coerces accordingly. This was caught by unit tests.

**Why not OmegaConf/Hydra?** Simpler. OmegaConf is what veRL uses, but it's a heavy dependency for what amounts to "load YAML into a dataclass". Plain PyYAML + dataclasses does the job.

---

## 3. Monarch Actors (`src/monarch/actors/`)

### `actors/generator.py` -- GeneratorActor

**What it does**: Runs vLLM in-process on dedicated GPUs. Generates G=8 completions per prompt with per-token log probabilities.

**Why in-process vLLM (not HTTP)?** The custom trainer runs vLLM as a separate pod and calls it via HTTP. This adds ~22 seconds per weight sync (serializing the entire model over HTTP). By running vLLM in-process inside a Monarch actor, we eliminate that overhead entirely.

**`generate()` endpoint**: Takes prompts, expands each G times, calls `engine.generate()` with `SamplingParams(n=group_size, logprobs=1)`. Returns completions + per-token log probs (needed for PPO ratio computation).

**`update_weights()` endpoint**: Receives a serialized state dict from the Learner, deserializes it, and loads it into vLLM's model. This is called every 3 training steps. The ~3GB transfer happens over EFA (400 Gbps), taking about 0.06 seconds theoretically (though serialization adds overhead).

**Why `trust_remote_code=True`?** Qwen models require custom code in their HuggingFace repo.

**Why `dtype="bfloat16"`?** Matches the training precision. Using float16 would cause numerical mismatches between generation and training log probs.

### `actors/learner.py` -- LearnerActor

**What it does**: FSDP-wrapped Qwen model that performs PPO-clipped training updates.

**FSDP setup**: Each actor instance is one FSDP rank. With 8 GPUs on Node 0, there are 8 LearnerActor instances. Monarch's `call()` broadcasts to all of them, so FSDP collectives (all-reduce, all-gather) work correctly.

**Why `Qwen2DecoderLayer` auto-wrap?** FSDP needs to know where to shard. Wrapping at the decoder layer level means each layer's parameters are sharded independently -- good balance between communication overhead and memory savings.

**`_compute_log_probs()`**: Forward pass on the full sequence (prompt + completion), but only extracts log probs for the response tokens. Uses the shift-by-one pattern: logits at position t predict token t+1. This matches how the model was trained and how vLLM computes log probs during generation.

**`train_step()` endpoint**: The core training logic:
1. Tokenize prompt + completion pairs
2. Forward pass → new log probs
3. Compute ratio: `exp(new_logprob - old_logprob)` (how much the policy changed)
4. PPO clipped loss: `min(ratio * advantage, clip(ratio, 1±0.2) * advantage)` (prevents too-large updates)
5. KL penalty: `(ratio - 1) - log(ratio)` approximate KL divergence (keeps policy near reference)
6. Backward pass + gradient clipping + optimizer step

**Why per-token, not per-sequence?** Each token in the response gets its own importance ratio and contributes to the loss individually. This gives finer-grained credit assignment than treating the whole response as one action.

**`get_weights()` endpoint**: Gathers FSDP-sharded weights onto rank 0 using `summon_full_params`, serializes with `torch.save`. The ~3GB bytes are sent to the Generator via Monarch messaging.

**`save_checkpoint()`**: Saves in HuggingFace format (not FSDP sharded format) so the checkpoint can be loaded directly with `AutoModelForCausalLM.from_pretrained()`.

### `actors/reward_actor.py` -- RewardActor

**What it does**: Wraps `compute_score()` as a Monarch actor running on a CPU pod.

**Why a separate CPU pod?** Reward computation is pure regex -- no GPU needed. Running it on a c7i.large CPU node keeps GPU memory free for training and generation. Same pattern as the custom trainer's FastAPI environment service, but as an actor instead of HTTP.

**Why not inline in the controller?** Keeps the pattern consistent (everything is an actor). Also allows future scaling -- if reward computation becomes expensive (e.g., code execution sandbox), it can be replicated independently.

### `actors/dataset_actor.py` -- DatasetActor

**What it does**: Serves batches of GSM8K problems. Tracks position, handles epoch cycling with reshuffling.

**Why an actor and not just a list?** State management. The dataset position, epoch counter, and shuffle order need to be tracked across training steps. Making it an actor isolates this state cleanly.

**Why reshuffle each epoch?** Standard practice -- prevents the model from memorizing the order of problems. Uses a fixed seed for reproducibility.

### `actors/replay_buffer.py` -- ReplayBufferActor

**What it does**: Stores scored episodes between generation and training. Tracks policy versions.

**Why have a replay buffer in synchronous mode?** Right now with `max_policy_age=0`, it's effectively a pass-through. But the architecture is ready for async training -- just increase `max_policy_age` and generation/training decouple automatically. No code changes needed.

**`max_policy_age`**: Controls how stale episodes can be. 0 = only current policy (synchronous). 1 = one step behind allowed. Higher = more async, better GPU utilization, but more off-policy bias.

**Why not just pass data directly?** The replay buffer is the decoupling point between the generation loop and the training loop. Even in sync mode, it cleanly separates "produce data" from "consume data".

---

## 4. Controller (`src/monarch/main.py`)

**What it does**: The orchestrator. Spawns all actors on the right GPU meshes and runs the synchronous GRPO training loop.

**The mesh setup**:
- `learner_procs`: 8 GPUs on Node 0 (FSDP training)
- `generator_procs`: 8 GPUs on Node 1 (vLLM generation)
- `cpu_procs`: CPU processes on Node 0 (dataset, reward, replay buffer)

**The training loop** (one step):
1. `dataset.next_batch()` → prompts + ground truths
2. `generator.generate()` → 256 completions (32 prompts × 8 per prompt)
3. `reward.score()` → 256 binary rewards
4. `compute_grpo_advantages()` → normalized advantages (runs locally, pure math)
5. `replay_buffer.add()` → store episodes
6. `replay_buffer.sample()` → get training batch
7. `learner.train_step()` → PPO update
8. Every 3 steps: `learner.get_weights()` → `generator.update_weights()` (weight sync)
9. Every 20 steps: `learner.save_checkpoint()` (HuggingFace format)

**Why `call_one()` everywhere?** Most operations target a single actor (or rank 0 of a mesh). `call()` would broadcast to all ranks -- only needed for the FSDP training step where all 8 ranks must participate.

**KubernetesJob fallback**: Tries `KubernetesJob` first (production), falls back to `this_host()` (local development). This means the same script works on your laptop with 2 GPUs and on EKS with 16 GPUs.

**Prompt formatting**: The chat-format prompts from `to_chat_prompt()` are converted to plain strings before sending to the Generator. The Generator's vLLM tokenizer handles the chat template.

---

## 5. Metrics (`src/monarch/metrics.py`)

**What it does**: Exposes training metrics via Prometheus on port 9090 and prints to stdout.

**Why Prometheus (not just stdout)?** The existing monitoring stack (Prometheus + Grafana) is already deployed on the cluster. By exposing metrics on :9090, we plug into the same dashboards used for veRL training. No new monitoring infrastructure needed.

**Why also stdout?** Stdout is captured by Fluent Bit → CloudWatch. This gives us a backup if Prometheus scraping fails, and makes it easy to `kubectl logs` during development.

**Gauge (not Counter/Histogram)?** Training metrics like loss and reward are point-in-time values, not cumulative. Gauges are the right Prometheus type for "current value" metrics.

---

## 6. Configuration (`config/monarch/qwen-1.5b.yaml`, `qwen-7b.yaml`)

**Why match veRL's hyperparameters?** veRL's config is proven (14.5% → 77% accuracy). Starting with the same hyperparameters gives a fair comparison -- any performance difference is due to the framework, not the hyperparameters.

**Key differences for 7B**:
- `train_batch_size: 128` (not 256) -- 7B model needs 4x more memory per sample
- `lr: 1e-6` (not 5e-6) -- larger models are more sensitive to learning rate
- `tensor_parallel_size: 2` -- 7B doesn't fit on a single A100 40GB for inference
- `micro_batch_size_per_gpu: 4` (not 16) -- memory constraint

---

## 7. Docker (`docker/monarch/Dockerfile`)

**Base image**: NVIDIA NGC PyTorch 25.01 -- includes PyTorch, CUDA, NCCL, cuDNN. Same base as the custom trainer.

**EFA installer**: Installs libfabric and aws-ofi-nccl so NCCL can use EFA for cross-node communication. `--skip-kmod` because the kernel module comes from the host (EFA device plugin DaemonSet). The EFA deps (pciutils, environment-modules, tcl) must be installed in the same `RUN` layer as the EFA installer -- if you clean the apt cache first, the installer can't find them.

**Why copy `src/shared/` and `src/monarch/` separately?** Docker layer caching. If you only change Monarch code, the shared layer is cached. If you only change shared code, both need to rebuild (but shared changes are rare).

---

## 8. Kubernetes (`k8s/monarch/`)

### `monarchmesh.yaml`
Replaces RayCluster (veRL) or Indexed Job (custom). The MonarchMesh CRD tells the operator to create 2 pods with 8 GPUs each. The operator handles pod discovery and service creation.

### `reward-deployment.yaml`
Separate CPU deployment for the RewardActor. Same pattern as the custom trainer's environment service. No GPU resources requested -- runs on c7i.large nodes.

### `configmap.yaml`
Embeds the training script and data prep code. Same pattern as veRL -- change training logic without rebuilding the Docker image. Contains two ConfigMaps: one for scripts, one for the YAML config.

### `serviceaccount.yaml`
IRSA (IAM Roles for Service Accounts) for S3 checkpoint access. The `ACCOUNT_ID` placeholder is substituted at deploy time by `run-training.sh`.

---

## 9. Deployment Scripts (`scripts/monarch/`)

### `run-training.sh`
The main entry point. Does everything in order:
1. Auto-detects AWS account ID (no hardcoding)
2. Installs MonarchMesh Helm operator if not present
3. Substitutes `ACCOUNT_ID` and region in manifests via `sed` at deploy time
4. Applies K8s manifests
5. Waits for pods
6. Pre-downloads the model
7. Launches training
8. Optionally follows logs and collects from CloudWatch when done

### `build-and-push-images.sh`
Builds for `linux/amd64` (EKS target) using `docker buildx`. Creates ECR repo if it doesn't exist.

### `collect-logs.sh`
Downloads training logs from CloudWatch (survives pod termination) plus live pod logs. Saves to `logs/monarch/run-<timestamp>/` (gitignored -- may contain sensitive info).

---

## 10. Tests (`tests/test_monarch.py`)

37 tests covering:
- **Shared reward**: correct/wrong/edge cases (negative, comma, decimal, empty, no marker)
- **Shared GRPO**: basic advantages, all-same rewards, zero-std, single group, output shape
- **Shared dataset**: prompt formats, batch loading, JSON loading, iteration, answer extraction
- **Config**: defaults, YAML loading (1.5B and 7B), type coercion, overrides
- **Metrics**: logger creation, stdout output
- **Replay buffer**: add/sample, staleness filtering, sync mode
- **Integration**: batch scoring, full reward→advantage pipeline

Tests that require Monarch runtime (actor spawning, GPU) are tested via the data structures and logic directly, not via Actor instantiation.
