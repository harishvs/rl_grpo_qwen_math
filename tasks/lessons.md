# Lessons Learned

## 1. NEVER use plan mode -- tasks/todo.md IS the plan
- **Pattern**: Repeatedly entered plan mode and edited the plan file instead of tasks/todo.md. Did this 3+ times despite being corrected.
- **Rule**: ALL planning, tracking, and updates go in `tasks/todo.md`. Never use plan mode. Never edit the plan file. If the system forces plan mode, immediately exit and update tasks/todo.md instead.
- **Why**: Plan mode is read-only and blocks execution. tasks/todo.md is the single source of truth per CLAUDE.md task management rules. Using both creates confusion and duplicate state.

## 2. Don't assume framework support -- verify first
- **Pattern**: Initially assumed TorchForge supported K8s. It doesn't (only MAST/Slurm).
- **Rule**: Always fetch and verify docs before committing to a framework choice.

## 3. Don't duplicate code across implementations -- create shared modules
- **Pattern**: Planned to copy reward.py and grpo.py into src/monarch/ instead of sharing.
- **Rule**: Framework-agnostic code (reward, advantage computation, dataset loading) goes in `src/shared/`. Each implementation imports from there. Original files become thin re-exports to preserve existing imports.
- **Why**: Three copies = three places to fix bugs. One shared module = one source of truth.

## 4. Use virtual environments, not system Python
- **Pattern**: Installed pytest with `--break-system-packages` directly into system Python.
- **Rule**: Always create a venv first. Never modify system Python.
- **Why**: System Python is shared by OS tools. Breaking it can cause system instability.

## 5. Don't conflate config flags with hardcoded behavior
- **Pattern**: Implied veRL had a placement config flag. It's actually hardcoded in `main_ppo.py`.
- **Rule**: When explaining framework behavior, distinguish between configurable and hardcoded.

## 6. proc_mesh.activate() ≠ NCCL process groups
- **Pattern**: Assumed `proc_mesh.activate()` was the Monarch way to set up FSDP. It's not — it's for Monarch's distributed tensor engine.
- **Rule**: FSDP needs `dist.init_process_group("nccl")`. Use `setup_torch_elastic_env(proc_mesh)` to set RANK/WORLD_SIZE/MASTER_ADDR env vars, then call `dist.init_process_group` in a post-spawn endpoint.
- **Why**: Wasted an entire run attempt before discovering these are different systems.

## 7. HuggingFace gradient_checkpointing_enable() doesn't work with FSDP2
- **Pattern**: Used `model.gradient_checkpointing_enable()` assuming it would reduce activation memory with composable FSDP.
- **Rule**: Always use PyTorch's `apply_activation_checkpointing` with FSDP2. HF's version has zero effect.
- **Why**: This was invisible — no error, no warning, just silently did nothing. Only discovered by testing memory usage.

## 8. Don't patch pods manually — rebuild the image
- **Pattern**: Repeatedly copied files to pods and uninstalled packages manually. Every pod restart lost the patches.
- **Rule**: After validating a fix works, bake it into the Docker image immediately. Manual patching is for one-off debugging, not iteration.
- **Why**: Wasted significant time re-patching after pod restarts. The flash-attn uninstall alone had to be done 6+ times.

## 9. Verify framework APIs against docs, don't assume
- **Pattern**: Flagged `context().message_rank` as wrong, assumed `proc_mesh.activate()` was for FSDP, assumed HostMesh supported Python slicing.
- **Rule**: When unsure about a framework API, check the docs or source. Say "worth verifying" instead of "this is wrong."
- **Why**: Multiple wrong assumptions cost debugging time. The Monarch docs at meta-pytorch.org/monarch/ are the source of truth.
