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
