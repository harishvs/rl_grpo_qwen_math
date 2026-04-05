"""LearnerActor -- FSDP training on dedicated GPUs.

Wraps Qwen2.5-1.5B in FSDP, computes PPO-clipped loss with KL penalty,
runs optimizer step. Each actor instance is one FSDP rank.
Weight serialization for pushing to the Generator.
"""
import io
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from monarch.actor import Actor, endpoint, current_rank, current_size


class LearnerActor(Actor):
    """FSDP-wrapped training actor for GRPO policy updates.

    Each instance is one FSDP rank. Monarch's call() broadcasts the
    train_step to all ranks so FSDP collectives work correctly.
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-6,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "bf16",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        self.model_name = model_name
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.policy_version = 0

        point = current_rank()
        rank = point.rank  # flat rank index across the mesh
        world_size = current_size()
        device = torch.device(f"cuda:{rank}")

        # Initialize process group for FSDP
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Auto-wrap policy for Qwen decoder layers
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            wrap_cls = Qwen2DecoderLayer
        except ImportError:
            wrap_cls = nn.Module  # fallback

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_cls},
        )

        # Mixed precision
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap in FSDP
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_indices: List[int],
    ) -> List[torch.Tensor]:
        """Compute per-token log probs for the response portion only.

        Args:
            input_ids: Full sequence (prompt + response) [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start_indices: Where the response starts in each sequence

        Returns:
            List of per-token log prob tensors, one per sequence
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # [batch, seq_len, vocab]
        log_probs_all = torch.log_softmax(logits, dim=-1)

        per_sequence_logps = []
        for i in range(input_ids.shape[0]):
            start = response_start_indices[i]
            # Shift: predict token t+1 from position t
            token_ids = input_ids[i, start:]
            logps = log_probs_all[i, start - 1:-1]  # shifted
            gathered = logps.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            per_sequence_logps.append(gathered)

        return per_sequence_logps

    @endpoint
    async def train_step(self, batch: dict) -> dict:
        """Perform one GRPO training step.

        Args:
            batch: Dict with 'prompts', 'completions', 'old_log_probs', 'advantages'

        Returns:
            Dict with training metrics
        """
        device = next(self.model.parameters()).device
        prompts = batch["prompts"]
        completions = batch["completions"]
        old_log_probs_list = batch.get("old_log_probs", batch.get("log_probs", []))
        advantages_list = batch["advantages"]

        # Tokenize prompt + completion pairs
        full_texts = [p + c for p, c in zip(prompts, completions)]
        prompt_encodings = self.tokenizer(prompts, add_special_tokens=False)
        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        response_start_indices = [len(ids) for ids in prompt_encodings["input_ids"]]

        # Forward pass: get new log probs
        new_log_probs_list = self._compute_log_probs(
            full_encodings["input_ids"],
            full_encodings["attention_mask"],
            response_start_indices,
        )

        # Compute PPO-clipped loss with KL penalty
        total_loss = torch.tensor(0.0, device=device)
        total_clip_fraction = 0.0
        total_kl = 0.0
        n_tokens = 0

        for i in range(len(prompts)):
            new_logps = new_log_probs_list[i]
            old_logps = torch.tensor(old_log_probs_list[i], device=device)
            advantage = advantages_list[i]

            # Truncate to same length
            min_len = min(len(new_logps), len(old_logps))
            if min_len == 0:
                continue
            new_logps = new_logps[:min_len]
            old_logps = old_logps[:min_len]

            # Per-token ratio and clipping
            ratio = (new_logps - old_logps).exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            # PPO objective (negative because we minimize)
            surr1 = ratio * advantage
            surr2 = clipped_ratio * advantage
            ppo_loss = -torch.min(surr1, surr2).mean()

            # KL penalty (approximate)
            kl = (ratio - 1) - (new_logps - old_logps)
            kl_loss = kl.mean()

            total_loss += ppo_loss + self.kl_coef * kl_loss
            total_clip_fraction += (
                (ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)
            ).float().mean().item()
            total_kl += kl.mean().item()
            n_tokens += min_len

        # Average over sequences
        n_sequences = max(len(prompts), 1)
        loss = total_loss / n_sequences

        # Backward + optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.policy_version += 1

        return {
            "loss": loss.item(),
            "kl_divergence": total_kl / n_sequences,
            "clip_fraction": total_clip_fraction / n_sequences,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "policy_version": self.policy_version,
            "n_tokens": n_tokens,
        }

    @endpoint
    async def get_weights(self) -> bytes:
        """Gather full state dict and serialize for weight sync.

        Uses FSDP.summon_full_params to gather sharded weights on rank 0.

        Returns:
            Serialized state dict bytes (only meaningful on rank 0)
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(self.model, writeback=False):
            state_dict = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    @endpoint
    async def save_checkpoint(self, path: str) -> str:
        """Save model checkpoint in HuggingFace format.

        Args:
            path: Directory to save checkpoint to

        Returns:
            Path where checkpoint was saved
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        point = current_rank()
        rank = point.rank

        with FSDP.summon_full_params(self.model, writeback=False):
            if rank == 0:
                # Save in HF format for easy loading
                unwrapped = self.model.module
                unwrapped.save_pretrained(path)
                self.tokenizer.save_pretrained(path)

        return path

    @endpoint
    async def get_stats(self) -> dict:
        """Return learner statistics."""
        return {
            "model_name": self.model_name,
            "policy_version": self.policy_version,
        }
