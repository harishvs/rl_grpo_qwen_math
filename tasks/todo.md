# Monarch GRPO Implementation for Qwen 2.5 + GSM8K on EKS

## Goal
Create a third GRPO training implementation using raw Monarch actors (Meta's distributed actor framework), alongside the existing custom and veRL implementations. TorchForge was ruled out because it only supports Slurm/MAST launchers -- not Kubernetes. Raw Monarch has K8s support via MonarchMesh CRD + operator.

---

## Phase 0: Education & Planning
- [x] Research Monarch concepts and create `docs/monarch-concepts.md`
  - **Why**: Monarch is a new framework with unfamiliar concepts (actors, meshes, RDMA). Need a reference doc before writing any code.
- [x] Research TorchForge API -- determined it doesn't support K8s
  - **Why**: TorchForge would have been the simpler path (pre-built actors). Had to verify K8s support before committing to it.
- [x] Update concepts doc to be a pure Monarch guide
  - **Why**: Since we're using raw Monarch (not TorchForge), the concepts doc needs to reflect the actual framework we're building on.
- [x] User reviews concepts doc and provides guidance
  - **Why**: User needs to understand the Monarch actor model before reviewing implementation decisions. Prevents misalignment.
- [x] Create implementation plan (`tasks/todo.md`) with explanations
  - **Why**: Non-trivial multi-file implementation across 5 directories. Need a roadmap before writing code.
- [x] Research veRL GPU placement (colocated vs split)
  - **Why**: User asked how veRL handles GPU layout. Finding: `main_ppo.py` hardcodes colocated placement -- all roles (Actor, Rollout, Ref) share one GPU pool. No hydra flag to split. Split requires a custom entry point using veRL's lower-level APIs (`examples/split_placement/`). This validates Monarch's approach where split placement is the natural default.
- [x] Get user sign-off on todo.md before starting implementation
  - **Why**: CLAUDE.md requires plan verification before starting. User needs to approve the approach.

---

## Phase 1a: Shared Module (`src/shared/`) -- eliminate code duplication

- [x] `src/shared/__init__.py` -- package init
  - **Why**: Makes `src/shared/` importable by all three implementations.

- [x] `src/shared/reward.py` -- move from `src/verl/reward.py`
  - **Why**: The reward logic (extract `####`, compare to ground truth) is identical across all three implementations. One copy, three consumers. Currently duplicated between `src/verl/reward.py` and `src/custom/environment/reward.py`.

- [x] `src/shared/grpo.py` -- move from `src/custom/trainer/grpo.py`
  - **Why**: The advantage formula `A_i = (r_i - mean(R)) / std(R)` is pure PyTorch math with zero framework dependency. No reason for each implementation to have its own copy.

- [x] `src/shared/dataset.py` -- move from `src/custom/trainer/dataset.py`
  - **Why**: GSM8K loading and prompt formatting is the same regardless of training framework. Add veRL-style chat template as an option.

- [x] Update `src/verl/reward.py` to import from `src/shared/reward.py`
  - **Why**: veRL's ConfigMap embeds reward.py directly, so this file becomes a thin re-export. Keeps veRL's deployment path working without changes to K8s manifests.

- [x] Update `src/custom/trainer/grpo.py` to import from `src/shared/grpo.py`
  - **Why**: Custom trainer imports `from src.custom.trainer.grpo import ...`. Make it re-export from shared so existing imports still work.

- [x] Update `src/custom/trainer/dataset.py` to import from `src/shared/dataset.py`
  - **Why**: Same re-export pattern. Existing imports in custom trainer keep working.

## Phase 1b: Monarch-specific code (`src/monarch/`)

- [x] `src/monarch/__init__.py` -- empty package init
  - **Why**: Makes `src/monarch/` a proper Python package.

- [x] `src/monarch/config.py` -- training configuration dataclass + YAML loader
  - **Why**: Monarch-specific config (GPU mesh layout, actor placement, weight sync interval). Imports shared pieces but adds Monarch-specific fields. Can't be shared because each framework has different config needs.

### 1b. Monarch actor classes (new code)

- [x] `src/monarch/actors/__init__.py` -- empty package init
  - **Why**: Makes `src/monarch/actors/` a package for the actor class files.

- [x] `src/monarch/actors/generator.py` -- GeneratorActor (vLLM-powered)
  - **Why**: This actor runs vLLM **in-process** on its assigned GPUs. For each batch of prompts, it generates G=8 completions and returns them with per-token log_probs. Unlike the custom trainer (which calls vLLM over HTTP), this eliminates network serialization overhead. Also handles weight updates when the Learner pushes new weights after training steps.

- [x] `src/monarch/actors/learner.py` -- LearnerActor (FSDP training)
  - **Why**: This is the core training actor. Wraps Qwen2.5-1.5B in FSDP (same sharding strategy as custom trainer), computes PPO-clipped loss with KL penalty, runs optimizer step. Each actor instance is one FSDP rank. Monarch's `call()` broadcasts the train_step to all ranks so FSDP collectives work correctly. Also handles weight serialization for pushing to the Generator.

- [x] `src/monarch/actors/reward_actor.py` -- RewardActor
  - **Why**: Wraps the reward function as a Monarch actor. Runs on CPU (no GPU needed). Could call the reward function directly in the main loop, but making it an actor keeps the pattern consistent and allows it to run on a separate process (doesn't block the GPU actors).

- [x] `src/monarch/actors/dataset_actor.py` -- DatasetActor
  - **Why**: Serves GSM8K batches as a Monarch actor. Tracks current position in the dataset, handles epoch cycling with reshuffling. Runs on CPU. Keeps dataset state isolated from the training loop.

- [x] `src/monarch/actors/replay_buffer.py` -- ReplayBufferActor
  - **Why**: Stores scored episodes (completions + rewards + advantages) between generation and training. In synchronous mode, this is a simple pass-through. But having it as an actor enables future async training where generation and training run at different speeds. Tracks policy versions to control staleness.

### 1c. Orchestration

- [x] `src/monarch/main.py` -- controller / training loop
  - **Why**: This is the "brain" that spawns all actors on the right GPU meshes and runs the GRPO training loop. It's the equivalent of `src/custom/trainer/main.py` + `trainer.py` combined, but instead of doing everything in one process, it coordinates actors via messages. Runs inside the cluster (same as veRL and custom -- all implementations run inside pods).

- [x] `src/monarch/metrics.py` -- Prometheus metrics exporter (REQUIRED)
  - **Why**: Exposes training metrics (loss, KL, reward, clip fraction, grad norm, step time, throughput) via Prometheus on port 9090. Same pattern as veRL's `cloudwatch_metrics.py`. Not optional -- needed for the existing Grafana monitoring stack.

---

## Phase 2: Configuration (`config/monarch/`)

- [x] `config/monarch/qwen-1.5b.yaml` -- training hyperparameters
  - **Why**: Externalizes all tunable parameters (model, lr, batch size, GPU allocation, etc.) so you can change training settings without modifying code. Modeled after `config/verl/qwen-1.5b.yaml` with same proven hyperparameters (lr=5e-6, group_size=8, kl_coef=0.1, clip=0.2).

---

## Phase 3: Docker (`docker/monarch/`)

- [x] `docker/monarch/Dockerfile` -- container image with EFA support
  - **Why**: Packages the Monarch training code with all dependencies (torchmonarch, vLLM, PyTorch, transformers). Based on NVIDIA NGC PyTorch image. Uses AWS EFA installer (`aws-efa-installer-latest.tar.gz` with `--skip-kmod`) for EFA userspace libraries (libfabric, aws-ofi-nccl). Kernel module comes from host via EFA device plugin DaemonSet. Our p4d.24xlarge nodes have 4 EFA interfaces (400 Gbps RDMA) -- NCCL needs this for fast cross-node weight sync.

- [x] `docker/monarch/requirements.txt` -- Python dependencies
  - **Why**: Lists exact package versions (torchmonarch==0.4.0, vLLM, datasets, etc.) for reproducible builds. Separate from Dockerfile for clarity.

---

## Phase 4: Kubernetes (`k8s/monarch/`)

- [x] `k8s/monarch/monarchmesh.yaml` -- MonarchMesh CRD (GPU pods)
  - **Why**: Provisions 2 GPU pods (8 GPUs each, p4d.24xlarge) for Learner and Generator actors. The operator handles pod discovery, service creation, and mesh formation.

- [x] `k8s/monarch/reward-deployment.yaml` -- RewardActor on a CPU pod
  - **Why**: RewardActor is pure regex/string matching -- no GPU needed. Running it on a CPU pod (c7i.large) keeps GPU memory free for training and generation. Same pattern as the custom trainer's environment service (`k8s/custom/environment/deployment.yaml`).

- [x] `k8s/monarch/configmap.yaml` -- embedded scripts
  - **Why**: Mounts the training script, reward function, and data prep code into pods via ConfigMap. This is the same pattern veRL uses -- it means you can update training logic without rebuilding the Docker image. The scripts run inside the pod at `/scripts/`.

- [x] `k8s/monarch/config-configmap.yaml` -- training config YAML
  - **Why**: Mounts the training hyperparameter YAML into pods at `/config/`. Separate from the scripts ConfigMap so you can change hyperparameters independently.

- [x] `k8s/monarch/serviceaccount.yaml` -- IRSA service account
  - **Why**: Gives pods AWS IAM permissions (via IRSA) to access S3 for checkpoint storage. Reuses the same IAM role pattern as the existing implementations. Without this, checkpoint saves to S3 would fail with permission errors.

---

## Phase 5: Deployment Scripts (`scripts/monarch/`)

- [x] `scripts/monarch/run-training.sh` -- deploy and launch training
  - **Why**: Single command to go from zero to running training. Applies K8s manifests, installs the MonarchMesh Helm operator if needed, waits for pods to be ready, then launches the training script inside the controller pod. Follows the same pattern as `scripts/verl/run-training.sh`. Without this, deployment is a manual multi-step process.

- [x] `scripts/monarch/build-and-push-images.sh` -- build Docker image
  - **Why**: Builds the Docker image and pushes to ECR. Same pattern as `scripts/custom/build-and-push-images.sh`. Needed before the K8s pods can pull the image.

- [x] `scripts/monarch/collect-logs.sh` -- capture training logs and push to GitHub
  - **Why**: Captures full training execution logs (stdout/stderr from all pods) to `docs/runs/monarch/` and commits + pushes to GitHub. Preserves the complete training record for debugging, comparison with custom/veRL runs, and presentation. Follows existing pattern in `docs/run-2026-02-19/`, `docs/run-2026-02-20-verl/`.

---

## Phase 6: Documentation & Verification

- [x] Update `docs/monarch-concepts.md` with implementation-specific details
  - **Why**: Add the actual architecture diagram, GPU allocation, and "how to run" instructions specific to our implementation. Currently the doc covers Monarch concepts generally.

- [x] Validate Python syntax (`python -m py_compile` on all .py files)
  - **Why**: Catch syntax errors before attempting to build/deploy. Fast sanity check.

- [x] Validate K8s manifests (`kubectl apply --dry-run=client`)
  - **Why**: Catch YAML errors and invalid resource specs before deploying to the cluster.

- [x] Update `README.md` with Monarch implementation section
  - **Why**: Per CLAUDE.md, always update README when adding new features. Add architecture overview, how to run, comparison with custom and veRL approaches.

- [x] Create sequence diagrams from actual code (Mermaid → PNG)
  - **Why**: Diagrams based on real code are accurate. Shows the actual message flow between actors (DatasetActor → Generator → RewardActor → Learner → weight sync). Build first, diagram after. Use Mermaid `.mmd` files rendered to PNG images. No ASCII art.

- [x] Create layered components diagram (Mermaid → PNG)
  - **Why**: Shows the stack visually -- K8s/EKS at the bottom, MonarchMesh operator, Monarch actors, GRPO training loop, vLLM/FSDP engines, shared modules. Mermaid rendered to PNG for embedding in README and docs.

- [x] Install mermaid-cli (`npm install -g @mermaid-js/mermaid-cli`) for rendering
  - **Why**: Converts `.mmd` Mermaid source files to PNG images. Store both source and rendered images in `docs/diagrams/monarch/`.

- [x] Update `tasks/todo.md` with review notes
  - **Why**: Per CLAUDE.md task management rules, document results after completion.

---

## Architecture Summary

```
Node 0 (8x A100 GPUs)                    Node 1 (8x A100 GPUs)
┌───────────────────────────┐             ┌───────────────────────────┐
│  LearnerActor (FSDP)      │  weights    │  GeneratorActor (vLLM)    │
│  8 ranks, one per GPU     │ ─────────>  │  in-process vLLM engine   │
│                           │             │                           │
│  PPO clip + KL penalty    │             │  G=8 completions/prompt   │
│  AdamW optimizer          │             │  per-token log_probs      │
│  gradient checkpointing   │             │                           │
├───────────────────────────┤             │  Reference Model          │
│  CPU Actors:              │             │  (CPU param offload)      │
│  - DatasetActor (GSM8K)   │             └───────────────────────────┘
│  - ReplayBufferActor      │
│  - Controller (main.py)   │
└───────────────────────────┘
                                          CPU Node (c7i.large)
                                          ┌───────────────────────────┐
                                          │  RewardActor              │
                                          │  regex extraction only    │
                                          │  no GPU needed            │
                                          └───────────────────────────┘
```

**Training step flow**:
1. DatasetActor → prompts + ground_truths
2. GeneratorActor (GPU) → 8 completions per prompt + log_probs
3. RewardActor (CPU pod) → binary rewards (0.0 or 1.0)
4. compute_grpo_advantages → group-normalized advantages
5. LearnerActor (GPU) → PPO update (forward + backward + optimizer step)
6. LearnerActor → push weights to GeneratorActor via EFA (every 3 steps)

---

## Key Decisions Made
1. **Framework**: Raw Monarch (not TorchForge) -- TorchForge lacks K8s support
2. **Model**: Qwen/Qwen2.5-1.5B (same as other implementations, proven baseline)
3. **GPU layout**: Node 0 = Learner (8 GPU FSDP), Node 1 = Generator (8 GPU vLLM)
4. **RewardActor**: Separate CPU pod (c7i.large) -- no GPU needed for regex matching
5. **Weight sync**: Serialized state_dict via EFA (~3GB in bf16, 400 Gbps)
6. **EFA**: AWS EFA installer in Dockerfile (--skip-kmod), kernel module from host
7. **Training mode**: Synchronous (generation → scoring → training → repeat)
8. **K8s resource**: MonarchMesh CRD + Helm operator (replaces RayCluster)
9. **Monitoring**: Prometheus on port 9090 (required), Grafana dashboards
10. **Hyperparameters**: Match veRL's proven config (lr=5e-6, group_size=8, batch=256)

---

## Review

### Files Created
- `src/shared/` -- 4 files (reward, grpo, dataset, __init__)
- `src/monarch/` -- 9 files (5 actors, config, main, metrics, __init__)
- `config/monarch/` -- 1 file (qwen-1.5b.yaml)
- `docker/monarch/` -- 2 files (Dockerfile, requirements.txt)
- `k8s/monarch/` -- 4 files (monarchmesh, reward-deployment, configmap, serviceaccount)
- `scripts/monarch/` -- 3 files (run-training, build-and-push-images, collect-logs)
- `docs/diagrams/monarch/` -- 4 files (2 Mermaid source + 2 PNG renders)

### Files Modified
- `src/verl/reward.py` -- re-exports from shared
- `src/custom/trainer/grpo.py` -- re-exports from shared
- `src/custom/trainer/dataset.py` -- re-exports from shared
- `README.md` -- added Monarch section, updated project structure, updated comparison table
- `CLAUDE.md` -- added no-duplication rule, Mermaid diagram rule, update README rule
- `tasks/lessons.md` -- captured 4 lessons
- `docs/monarch-concepts.md` -- rewritten as pure Monarch guide

### Verification
- All Python files pass `py_compile`
- All YAML files pass `yaml.safe_load`
- Shared module imports verified (reward, grpo, dataset)
- Re-exports verified (verl, custom paths still work)
- Mermaid diagrams rendered to PNG successfully

---

## Phase 7: Deployment & Training (2026-04-05 to 2026-04-06)

### Run #1 (2026-04-05 ~09:12) — Generation works, FSDP fails
- [x] Deploy MonarchMesh, fix 16+ issues (CRD API, labels, flash-attn, numpy, etc.)
- [x] Generation working: 256 completions at 8400 tok/s via vLLM
- [x] FSDP fails: actors can't form NCCL process group (fundamental isolation problem)
- Observations: `docs/run-2026-04-05-0912-monarch/observations.md`

### Run #2 (2026-04-05 ~10:15) — proc_mesh.activate() attempt
- [x] Rewrite LearnerActor with `proc_mesh.activate()` + composable `fully_shard()`
- [x] Rebuild Docker image with all deps baked in
- [x] Discovered `proc_mesh.activate()` is for tensor engine, not FSDP/NCCL
- Observations: `docs/run-2026-04-05-1015-monarch/observations.md`

### Run #3 (2026-04-06 ~00:30) — FSDP working, training running
- [x] Research: `current_rank()`/`current_size()` are base actor APIs, not SPMDActor-specific
- [x] Use `setup_torch_elastic_env(learner_procs)` to set torchrun env vars on proc mesh
- [x] Refactor LearnerActor: lightweight `__init__`, `@endpoint initialize()` for FSDP setup
- [x] `dist.init_process_group("nccl")` + composable `fully_shard()` in initialize endpoint
- [x] Remove `proc_mesh.activate()` from all endpoints
- [x] Fix host slicing: `hosts.slice(hosts=slice(0,1))` for split placement
- [x] Fix flash-attn: uninstall broken package, vLLM falls back to non-flash rotary
- [x] Fix Dockerfile: remove flash-attn build step
- [x] Data sharding: each FSDP rank processes `batch_size / world_size` samples
- [x] Fix model device: load on CPU, let `fully_shard()` handle CPU→CUDA
- [x] Fix activation checkpointing: HF's doesn't work with FSDP2, use PyTorch's
- [x] Add `reshard_after_forward=True` per layer, `False` on root
- [x] Add `use_cache=False` in forward pass
- [x] Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [x] Fix ValueMesh: `.call()` returns ValueMesh, use `.values()[0]`
- [x] Micro-batching: 4 samples per forward+backward to limit activation memory
- [x] FSDP init confirmed: all 8 ranks, 1.4 GiB/GPU (correctly sharded)
- [x] Training confirmed: 3 steps, 26 samples/sec, reward=0.15
- Observations: `docs/run-2026-04-05-1015-monarch/observations-run3.md`

### Remaining — not started
- [ ] Fix vLLM weight sync API (v0.19.0 changed `model_executor` path)
- [ ] Fix KL divergence explosion (hyperparameter tuning — kl_coef, lr, clip_range)
- [ ] Rebuild Docker image with all fixes (no more manual pod patching)
- [ ] Run full training (233 steps, 1 epoch) end to end
- [ ] Collect final training logs and metrics
- [ ] Compare throughput and accuracy with veRL and custom implementations
- [ ] File GitHub issue on pytorch/monarch for FSDP example
