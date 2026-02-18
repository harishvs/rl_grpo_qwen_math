# Crash Log

## Trainer Pod

| Time (UTC) | Pod | Cause | Notes |
|---|---|---|---|
| 2026-02-17 ~21:07 | grpo-trainer-wl5dn | Checkpoint load failure | FSDP `_flat_param` keys incompatible with `from_pretrained()`. Model loaded with random weights for `lm_head.weight`, `model.embed_tokens.weight`, `model.norm.weight`. Pod restarted 4+ times before hitting backoff limit. |

## Environment Pod

| Time (UTC) | Pod | Cause | Notes |
|---|---|---|---|
| — | — | No crashes recorded | — |
