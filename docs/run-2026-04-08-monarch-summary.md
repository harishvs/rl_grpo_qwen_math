# Monarch GRPO — Weight Sync Attempts Summary

## The Problem

Every 3 training steps, 3 GB of model weights must transfer from learner (Node 0, 8 GPUs) to generator (Node 1, 4 GPUs with TP=4 vLLM).

## Attempts

### 1. Serialized RPC (WORKING — 47s)
- Learner: `state_dict()` → `full_tensor()` per param → `torch.save` → bytes over Monarch RPC
- Generator: `torch.load` → save as safetensors → `reload_weights(weights_path=)`
- **Why it works**: `full_tensor()` returns correct HF-namespace keys. `reload_weights` handles TP distribution.
- **Why it's slow**: 3 GB serialized twice (torch.save + safetensors), transferred over TCP.

### 2. Monarch RDMA (BROKEN)
- `RDMABuffer.read_into()` fails between any two actors, even on same node
- **Root cause**: AWS EFA doesn't support ibverbs RC queue pairs. Monarch RDMA uses ibverbs directly. EFA only works through libfabric.
- Verified: `ibv_rc_pingpong` fails with "Couldn't create QP", `ibv_ud_pingpong` fails with "GID ::"

### 3. Gloo ProcessGroupGloo send/recv (BROKEN — wrong keys)
- Learner: NCCL all_gather → gloo send to generator (27s total)
- **Root cause**: Reconstructed state_dict using `named_parameters()` keys (FSDP namespace) instead of `state_dict()` keys (HF namespace). `load_state_dict(strict=False)` silently ignored all keys — zero weights loaded.
- Fixed the keys to use `state_dict()`, but then NCCL gather reconstruction also had key ordering issues.

### 4. FSx shared filesystem with reload_weights (BROKEN — vLLM state corruption)
- Learner: `full_tensor()` → `save_file` (safetensors) to `/checkpoints/latest_weights/`
- Generator: `reload_weights(weights_path="/checkpoints/latest_weights/")`
- Correct keys in safetensors (verified). But `reload_weights` internally sets `self.model_config.model = weights_path`, which corrupts vLLM's state for subsequent operations.
- Reward declined from 0.14 → 0.09 (model getting worse, not better).

### 5. FSx with apply_model (INCOMPATIBLE with TP>1)
- Would work with TP=1: `torch.load` from FSx → `apply_model(lambda model: model.load_state_dict(...))`
- Fails with TP>1: vLLM can't pickle the closure across worker processes. `VLLM_ALLOW_INSECURE_SERIALIZATION=1` doesn't help with TP=4.

### 6. Parameter Server + RDMA (BROKEN — same as #2)
- ParameterServerActor in worker process (correct RDMA manager visibility)
- Learner ranks push shards via messages (works)
- Generator RDMA read from param server (fails — ibverbs doesn't work on EFA)

## Why TP=4 + RPC wins despite being slow

| | TP=4 + RPC (47s sync) | TP=1 + FSx (5s sync) |
|---|---|---|
| Generation (2048 completions) | ~58s | ~230s |
| Weight sync overhead | ~47s | ~5s |
| Normal step | ~58s | ~230s |
| Sync step (every 3rd) | ~105s | ~235s |
| Full epoch (29 steps) | ~32 min | ~112 min |

Generation dominates step time. TP=4 is 3.5x faster overall despite slow weight sync.

## What would actually fix this

1. **vLLM `reload_weights` without state corruption**: Don't change `model_config.model`. Or provide a `load_weights_from_state_dict()` API that works with TP>1.

2. **Monarch RDMA over libfabric**: Instead of ibverbs (broken on EFA), use libfabric like NCCL does via aws-ofi-nccl.

3. **TorchStore on K8s**: Monarch's TorchStore handles DTensor transfer between meshes natively. Not available in torchmonarch 0.4.0 on Kubernetes.

4. **Colocated placement**: Like veRL — all GPUs switch between training and generation roles. Zero weight transfer. But requires Monarch to support role-switching on the same mesh.
