# CLAUDE.md

## 1. Code Collaboration Style
- **For new code**: Pair-program style -- describe what you're about to write before writing it. Let the user adjust the approach, then write the code.
- **For bug fixes**: Same pair-program style -- explain the root cause and proposed fix before changing code. Don't silently fix and retry.
- **For already written code**: Create a code walkthrough doc explaining the implementation file by file with the reasoning behind each design choice.

## 2. Plan Mode Default

- Enter plan mode for any non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, stop and re-plan immediately — do not keep pushing.
- Use plan mode for verification steps, not just building.
- Write detailed specs up front to reduce ambiguity.
- Always check latest documentation for any product you are working on, for example verl, dont assume

## 2. Subagent Strategy

- Use subagents liberally to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems, throw more compute at it via subagents.
- Keep one task per subagent for focused execution.

## 3. Self-Improvement Loop

- After any correction from the user, update `tasks/lessons.md` with the pattern.
- Write rules for yourself that prevent the same mistake.
- Ruthlessly iterate on these lessons until the mistake rate drops.
- Review lessons at session start for the relevant project.

## 4. Verification Before Done

- Always write unit tests for new implementations.
- Never mark a task complete without proving it works.
- Diff behavior between main and your changes when relevant.
- Ask yourself: “Would a staff engineer approve this?”
- Run tests, check logs, and demonstrate correctness.

## 5. Demand Elegance (Balanced)

- For non-trivial changes, pause and ask “Is there a more elegant way?”
- If a fix feels hacky, “Knowing everything I know now, implement the elegant solution.”
- Skip this for simple, obvious fixes — do not over-engineer.
- Challenge your own work before presenting it.

## 6. Autonomous Bug Fixing

- When given a bug report, just fix it. Do not ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them.
- Aim for zero context switching required from the user.
- Fix failing CI tests without being told how.
- **When there is an error, investigate it thoroughly. Don't give up and switch to the easiest crappy option.** Find the root cause, understand why it fails, and fix the actual problem. Reverting to a worse approach is not a fix.

## Task Management

1. **Plan First**: Write a plan to `tasks/todo.md` with checkable items.
2. **Verify Plan**: Check in before starting implementation.
3. **Track Progress**: Mark items complete as you go.
4. **Explain Changes**: Provide a high-level summary at each step.
5. **Document Results**: Add a review section to `tasks/todo.md`.
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections.

## Core Principles

- **Use virtual environments**: Always use `venv` for installing Python packages. Never install with `--break-system-packages` or modify the system Python.
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes, avoid temporary fixes, and aim for senior developer standards.

## Security
- Never hardcode AWS account IDs, VPC IDs, subnet IDs, security group IDs, or capacity reservation IDs in code or manifests. Use placeholders (e.g., ACCOUNT_ID, vpc-REDACTED) or environment variables.
- Never commit secrets, API keys, or credentials. Use IRSA, environment variables, or K8s Secrets.
- Always use `*.tfvars` in .gitignore. Provide a `.tfvars.example` with dummy values instead.

## Commit 
- dont add Co-Authored-By: 
