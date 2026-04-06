# Monarch GRPO Training Run #2 -- 2026-04-05 10:15 PT

## Cluster
- **EKS**: rl-code-llm-training-dev, EKS 1.30, us-west-2
- **GPU Nodes**: 2x p4d.24xlarge (8x A100 40GB each) in us-west-2d
- **CPU Nodes**: 2x c7i.large
- **Capacity Block**: cr-0e35a6085ded0488f (expires 2026-04-06 04:30 PT)
- **Monarch**: torchmonarch 0.4.0, MonarchMesh CRD operator via Helm

## Changes from Run #1 (0912)

### Key fix: proc_mesh.activate() for FSDP
- **Previous approach**: Manual `dist.init_process_group("nccl")` and `setup_torch_elastic_env()` -- both failed because Monarch actors are isolated processes that can't rendezvous
- **New approach**: Use `context().actor_instance.proc_mesh.activate()` which is Monarch's built-in distributed context. NCCL process groups and device assignment are handled by the proc mesh.
- **FSDP**: Switched from `FullyShardedDataParallel` wrapper to composable `fully_shard()` API (`torch.distributed._composable.fsdp`) which works inside proc_mesh.activate()

### Docker image rebuilt with all fixes baked in
Previous run had to pip install fixes into running containers (kubernetes, flash_attn, vllm, numpy) which were lost on pod restart. New Dockerfile:
1. `torchmonarch==0.4.0` installed first (upgrades PyTorch 2.6→2.10)
2. `vllm` force-reinstalled against PyTorch 2.10
3. `flash-attn` force-reinstalled from source (CUDA compilation against PyTorch 2.10)
4. `numpy<2` pinned (NumPy 2.x breaks modules compiled against 1.x)
5. `kubernetes>=28.0.0` for KubernetesJob

### Other fixes
- Lazy import of `KubernetesJob` (was failing at module level when kubernetes not installed)
- Proper `KubernetesJob` API: `add_mesh()` with label_selector, not `mesh_name` kwarg
- Removed `setup_torch_elastic_env` (no longer needed with proc_mesh.activate)

## Build Status

Docker image rebuilding with all fixes. Flash_attn CUDA compilation takes ~15 min.

## What we expect to happen
1. Pods pull new image with everything pre-installed
2. Workers start `run_worker_loop_forever()` 
3. Controller spawns actors:
   - GeneratorActor (1 process, vLLM on GPU) -- worked in run #1
   - LearnerActor (8 processes, FSDP via proc_mesh.activate) -- new approach
   - DatasetActor, ReplayBufferActor (CPU)
4. Training loop: dataset → generate → reward → advantages → train → weight sync
5. Generation already proven at ~8400 tok/s in run #1

## Risks
- `proc_mesh.activate()` + `fully_shard()` is the correct Monarch pattern (from their GRPO tensor engine example) but we haven't tested it yet with a real transformer model
- Composable FSDP (`fully_shard`) may behave differently than the wrapper FSDP with Qwen's architecture
- Weight gathering for sync (`model.state_dict()` under composable FSDP) may need different handling

---

## Timeline

| Time (PT) | Event |
|-----------|-------|
| ~09:12 | Run #1 started. Multiple issues: no entrypoint, label mismatch, CRD API group, numpy, flash_attn, vllm |
| ~09:55 | Generation working (256 completions at 8400 tok/s). FSDP fails (no NCCL rendezvous) |
| ~10:00 | Researched SPMDActor, setup_torch_elastic_env. Neither solves the core problem |
| ~10:10 | Researched proc_mesh.activate() + torchtitan. Found the correct Monarch pattern |
| ~10:15 | Rewrote LearnerActor with proc_mesh.activate() + composable fully_shard() |
| ~10:20 | Rebuilding Docker image with all fixes baked in |
| | *waiting for build...* |
