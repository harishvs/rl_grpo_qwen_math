# MoE + RL Training: Framework Survey (April 2026)

Research into which frameworks support Mixture of Experts (MoE) model training for reinforcement learning (GRPO, PPO, etc.), and how mature that support is.

---

## Summary Table

| Framework | MoE Support | Expert Parallelism | MoE Models Tested | GRPO/PPO for MoE | Production-Tested | Maturity |
|-----------|------------|--------------------|--------------------|-------------------|-------------------|----------|
| **NeMo-RL** (NVIDIA) | Yes | Yes (EP via Megatron-Core) | Qwen3.5 MoE (dense+MoE) | GRPO confirmed | Likely (NVIDIA internal) | **Highest** |
| **ChatLearn** (Alibaba) | Yes | Yes (Hyper EP) | Qwen2-MoE, Qwen3-MoE, DeepSeek V2 | Unclear (SFT/alignment focus) | Likely (Alibaba internal) | **High** |
| **veRL** (ByteDance) | Yes | Yes (EP/ETP since v0.6.1) | Qwen3-30B-A3B, Qwen3-VL-235B, Qwen3.5-35B-A3B | GRPO confirmed | Yes, but with significant issues | **Medium** (active bugs) |
| **OpenRLHF** | Partial | No (DeepSpeed ZeRO only) | Qwen-1.5-MoE, Qwen3-30B-A3B | PPO confirmed, GRPO likely | No evidence | **Low-Medium** |
| **TRL** (HuggingFace) | Incidental | No | Qwen3 MoE, Mixtral, GLM-4-MoE | GRPO/DPO (no MoE-specific logic) | No | **Low** |
| **DeepSpeed-Chat** | No MoE-specific | N/A | N/A | N/A | No | **None** |
| **Oat** (SAIL) | No | No | N/A | N/A | No | **None** |
| **rLLM** (Agentica) | No evidence | No | N/A | N/A | No | **None** |

---

## Detailed Analysis

### 1. NeMo-RL (NVIDIA) -- Most Mature MoE + RL

**Repo**: github.com/NVIDIA/NeMo-RL (successor to NeMo-Aligner, which was archived Nov 2025)

**MoE Support**: Strongest of any framework surveyed.
- Full expert parallelism (EP) via Megatron-Core integration
- Supported parallelism: TP/PP/CP/SP/EP/FSDP -- all composable
- Explicit GRPO training recipes for MoE models
- Qwen3.5 dense and MoE models (LLM and VLM) confirmed for GRPO training

**Known Issues** (from GitHub issues):
- MoE model fails with CPU offload enabled (gradient/parameter device mismatch) -- #2030
- DTensor backend fails with `expert_parallel_size=1` (`moe_mesh` is None) -- #2028
- Qwen3-MoE finetuning challenges reported -- #2087
- `moe_permute_fusion` default configuration questions -- #2255

**Production Evidence**:
- Blog post: "Journey of Optimizing Weight Transfer in Large MoE Models by 10x" (Sept 2025)
- Roadmap explicitly lists "Improved Large MoE Performance" as a priority
- NVIDIA likely uses this internally for alignment work

**Assessment**: NeMo-RL is the most complete option for MoE + GRPO. Megatron-Core provides battle-tested expert parallelism. The main drawback is complexity -- Megatron-Core requires specific model implementations and is not plug-and-play with arbitrary HuggingFace models. You need Megatron-format checkpoints.

---

### 2. ChatLearn / Pai-Align (Alibaba) -- Strong MoE, Less RL Focus

**Repo**: github.com/alibaba/ChatLearn

**MoE Support**: Extensive and actively maintained.
- Hyper expert parallel support with parameter synchronization (PR #136)
- GroupGEMM optimization for Qwen3-MoE (PR #296)
- FSDP2 support (PR #301)
- Megatron-Core backend for dense/MoE models
- VLLM checkpoint loading for MoE models

**Supported MoE Models**: Qwen2-MoE, Qwen3-MoE, DeepSeek V2

**Known Issues**:
- Convergence fixes needed for MoE in some cases (PR #293)
- MoE RL training tutorials listed as "planned but not yet available" on roadmap
- Documentation is sparse (Chinese-primary)

**Assessment**: ChatLearn has strong MoE infrastructure (likely used for Qwen's own alignment), but the RL training side (GRPO/PPO specifically for MoE) is less documented. The roadmap explicitly marks MoE RL tutorials as incomplete. Best suited if you're training Qwen-family MoE models and are comfortable with limited English documentation.

---

### 3. veRL (ByteDance) -- Has MoE GRPO, But Buggy

**Repo**: github.com/volcengine/verl

**MoE Support**: Present since v0.6.1 with EP/ETP selection.
- Expert parallelism support added
- `moe_aux_loss_coeff` and `moe_z_loss_coeff` exposed (PR #4103, Nov 2025)
- Megatron backend for MoE training
- vLLM integration for MoE rollouts

**Supported MoE Models**: Qwen3-30B-A3B, Qwen3-VL-235B, Qwen3.5-35B-A3B, Qwen3.5-122B

**Known Issues (significant)**:
- **OOM**: Qwen3-30B-A3B OOM with GRPO on 4x8xH200 (564GB total) -- #3364
- **Checkpoint saving hangs**: Distributed checkpointing stuck on NCCL communication with MoE models -- #2238 (UNRESOLVED, dense models save in seconds, MoE hangs indefinitely)
- **DP + EP conflicts**: Hanging with vLLM rollout when DP > 1 -- fixed in #5609
- **Parallelism mismatch**: Inconsistent parallelism between veRL and vLLM during MoE rollout -- #5568
- **Resume failures**: Qwen3.5-35B-A3B checkpoint resume errors -- #5892
- **Entropy collapse**: During Qwen3.5-35B-A3B training -- #5953
- **FSDP2 checkpoint offloading issues** with MoE -- #3258
- **FP8 quantization OOM**: Full-model materialization causes OOM -- #4641

**Assessment**: veRL has the broadest MoE model coverage and confirmed GRPO support, but the bug list is long and several issues remain unresolved. The checkpoint saving hang (#2238) is particularly concerning -- you cannot reliably save MoE training progress. This matches the issues you've already identified.

---

### 4. OpenRLHF -- Basic MoE, No Expert Parallelism

**Repo**: github.com/OpenRLHF/OpenRLHF

**MoE Support**: Basic, through DeepSpeed ZeRO.
- MoE training script exists (`train_sft_moe.sh`) with `--aux_loss_coef`
- Relies entirely on DeepSpeed for distribution -- no native expert parallelism
- FSDP2 backend added (PR #1115) but MoE not specifically tested with it

**Supported MoE Models**: Qwen-1.5-MoE, Qwen3-30B-A3B (with fixes)

**Known Issues**:
- **Qwen3-30B-A3B PPO training hangs** with NCCL timeout during MoE expert layer processing (DeepSpeed ZeRO-3) -- #1097 (fixed but fragile)
- **Qwen-MoE multi-GPU hang**: Training hangs during backward pass with DeepSpeed ZeRO-2; workaround is switching to ZeRO-3 -- #1063 (root cause was a DeepSpeed bug)
- **DeepSeek V2 236B**: No response from maintainers on whether PPO/GRPO works -- #738 (open, unanswered)
- No expert parallelism means all experts are sharded via ZeRO, not distributed to dedicated GPUs

**Assessment**: OpenRLHF's MoE support is fragile and relies on DeepSpeed ZeRO, which has known bugs with MoE architectures. There is no expert parallelism, limiting scalability for large MoE models. It works for smaller MoE models (Qwen-1.5-MoE) with workarounds, but is not ready for production MoE training at scale.

---

### 5. TRL (HuggingFace) -- No MoE-Specific Support

**Repo**: github.com/huggingface/trl

**MoE Support**: None intentionally -- MoE models work only because TRL is model-agnostic.
- No expert parallelism
- No MoE-specific memory optimization
- No aux loss handling
- Relies on DeepSpeed ZeRO or FSDP for distribution

**Known Issues**:
- Qwen3 MoE CUDA OOM in DDP settings -- #4961
- Mixtral 16-bit LoRA OOM with DeepSpeed ZeRO-3 on 4x A100-80GB -- #1268
- DPO crashes with PEFT `target_parameters` on MoE models (Transformers 5.x) -- reported 2026

**Assessment**: TRL is a non-starter for serious MoE training. It has no MoE-aware features. You can technically load an MoE model and call GRPOTrainer, but you'll hit OOM quickly on anything beyond small MoE models, and there's no expert parallelism to help. Fine for dense models, not designed for MoE.

---

### 6. DeepSpeed-Chat -- Dead End for MoE + RL

**Repo**: github.com/microsoft/DeepSpeed (blogs/deepspeed-chat)

DeepSpeed itself has excellent MoE support (expert parallelism, MoE-specific optimizations). However, DeepSpeed-Chat (the RLHF pipeline) has **no MoE integration**. The RLHF pipeline was designed around dense OPT models and hasn't been updated to leverage DeepSpeed's MoE capabilities. No issues or PRs exist combining the two.

---

### 7. Oat (SAIL-SG) -- No MoE

**Repo**: github.com/sail-sg/oat

Lightweight online alignment framework. Uses DeepSpeed ZeRO for the learner and vLLM for sampling. No MoE support, no expert parallelism, no evidence of MoE models being used. Focused on algorithmic research (SEA, APL, XPO), not scaling MoE.

---

### 8. rLLM (Agentica) -- No Evidence of MoE

**Repo**: github.com/agentica-project/rllm

Built on a modified fork of veRL. Focuses on agent training (SWE-bench, coding). Tested with dense models (DeepSeek-R1-Distill-Qwen-1.5B/14B, Qwen3-32B). No MoE-specific features documented.

---

## Key Findings

### The MoE + RL Problem is Genuinely Hard

Every framework that attempts MoE + RL training hits the same issues:
1. **NCCL timeouts** during all-gather/all-reduce in expert layers
2. **Checkpoint saving** fails or hangs for MoE models (communication overhead)
3. **OOM** even on high-end hardware (MoE parameter counts are deceptive -- all experts must be loaded)
4. **Parallelism mismatches** between training (FSDP/DeepSpeed) and inference (vLLM) for MoE

### Realistic Options (Ranked)

1. **NeMo-RL** -- Best bet if you can tolerate Megatron-Core complexity. Real EP support, GRPO recipes for MoE, NVIDIA backing. Downsides: steep learning curve, Megatron checkpoint format, less community support than HF ecosystem.

2. **ChatLearn** -- Strong MoE infrastructure if training Qwen-family models. Expert parallelism works. Downsides: RL training for MoE not yet documented, Chinese-primary docs, Alibaba-internal tooling assumptions.

3. **veRL** -- Broadest model coverage and active development, but significant unresolved bugs (checkpoint hangs, OOM, entropy collapse). Could work if you're willing to debug and contribute fixes. You already know the pain points.

4. **OpenRLHF** -- Possible for small MoE models with workarounds. No expert parallelism is a hard ceiling for larger models. Not recommended for production MoE training.

### What Nobody Has Solved Yet

- **DeepSeek V3/V3.1 RL training**: No framework has publicly demonstrated GRPO/PPO training on DeepSeek V3's full MoE architecture. The 671B model with 256 experts and MLA is beyond what any open-source framework has validated.
- **Reliable MoE checkpointing during RL**: This is broken or fragile in every framework surveyed.
- **Seamless EP + RL rollout**: The mismatch between training parallelism (EP during backprop) and inference parallelism (vLLM during rollout) is an unsolved systems problem that every framework handles differently and imperfectly.
