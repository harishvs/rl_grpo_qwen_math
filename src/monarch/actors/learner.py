"""LearnerActor -- FSDP training on dedicated GPUs via Monarch actors.

Uses setup_torch_elastic_env() to configure NCCL process groups (RANK,
WORLD_SIZE, MASTER_ADDR, etc.) on the proc mesh before spawning.
Then dist.init_process_group("nccl") + composable fully_shard() in the
initialize endpoint -- after the actor is fully constructed.
"""
import io
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from monarch.actor import Actor, endpoint, current_rank, current_size


class LearnerActor(Actor):
    """FSDP-wrapped training actor for GRPO policy updates.

    Env vars (RANK, WORLD_SIZE, etc.) are set by setup_torch_elastic_env()
    on the proc mesh before this actor spawns. The initialize() endpoint
    then calls dist.init_process_group("nccl") and sets up FSDP.
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-6,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.policy_version = 0
        self.model = None
        self.optimizer = None
        self.tokenizer = None

    @endpoint
    async def initialize(self) -> str:
        """Set up NCCL process group, load actor + reference models with FSDP.

        Must be called after spawn. Env vars are already set by
        setup_torch_elastic_env() on the proc mesh.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if not dist.is_initialized():
            dist.init_process_group("nccl")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # --- Actor model (trainable, FSDP on GPU) ---
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        for layer in model.model.layers:
            fully_shard(layer, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=False)

        if self.gradient_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, Qwen2DecoderLayer)
            )

        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # --- Reference model (frozen, FSDP with CPU param offload) ---
        # Params live on CPU, moved to GPU only during forward pass.
        # Same approach as veRL's ref.fsdp_param_offload: true
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        offload = CPUOffloadPolicy()
        for layer in ref_model.model.layers:
            fully_shard(layer, reshard_after_forward=True, offload_policy=offload)
        fully_shard(ref_model, reshard_after_forward=True, offload_policy=offload)

        self.ref_model = ref_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return f"rank {rank}/{world_size} initialized on cuda:{local_rank}"

    def _forward_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_indices: List[int],
    ) -> List[torch.Tensor]:
        """Compute per-token log probs for the response portion using given model."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        logits = outputs.logits
        log_probs_all = torch.log_softmax(logits, dim=-1)

        per_sequence_logps = []
        for i in range(input_ids.shape[0]):
            start = response_start_indices[i]
            token_ids = input_ids[i, start:]
            logps = log_probs_all[i, start - 1:-1]
            gathered = logps.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            per_sequence_logps.append(gathered)

        return per_sequence_logps

    def _compute_log_probs(self, input_ids, attention_mask, response_start_indices):
        """Compute per-token log probs using the actor model."""
        return self._forward_log_probs(self.model, input_ids, attention_mask, response_start_indices)

    def _compute_ref_log_probs(self, input_ids, attention_mask, response_start_indices):
        """Compute per-token log probs using the frozen reference model (CPU offloaded)."""
        with torch.no_grad():
            return self._forward_log_probs(self.ref_model, input_ids, attention_mask, response_start_indices)

    @endpoint
    async def train_step(self, batch: dict) -> dict:
        """Perform one GRPO training step. All FSDP ranks must call this together.

        Each rank processes its shard of the batch (data parallelism).
        FSDP handles weight sharding and gradient all-reduce.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Shard data across ranks
        all_prompts = batch["prompts"]
        all_completions = batch["completions"]
        all_old_log_probs = batch.get("old_log_probs", batch.get("log_probs", []))
        all_advantages = batch["advantages"]

        n = len(all_prompts)
        per_rank = max(n // world_size, 1)
        start = rank * per_rank
        end = start + per_rank if rank < world_size - 1 else n

        prompts = all_prompts[start:end]
        completions = all_completions[start:end]
        old_log_probs_list = all_old_log_probs[start:end]
        advantages_list = all_advantages[start:end]

        device = next(self.model.parameters()).device
        micro_bs = 4  # Limit activation memory per forward pass

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0
        n_tokens = 0
        n_sequences = max(len(prompts), 1)

        # Micro-batch loop: forward + backward per chunk, gradients accumulate.
        # Disable FSDP gradient sync for all but the last micro-batch —
        # otherwise FSDP all-reduces at each .backward(), averaging prematurely.
        n_micro_batches = (len(prompts) + micro_bs - 1) // micro_bs

        for mb_idx, mb_start in enumerate(range(0, len(prompts), micro_bs)):
            is_last_mb = (mb_idx == n_micro_batches - 1)
            self.model.set_requires_gradient_sync(is_last_mb)

            mb_end = min(mb_start + micro_bs, len(prompts))
            mb_prompts = prompts[mb_start:mb_end]
            mb_completions = completions[mb_start:mb_end]
            mb_old_logps = old_log_probs_list[mb_start:mb_end]
            mb_advantages = advantages_list[mb_start:mb_end]

            full_texts = [p + c for p, c in zip(mb_prompts, mb_completions)]
            prompt_encodings = self.tokenizer(mb_prompts, add_special_tokens=False)
            full_encodings = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            response_start_indices = [len(ids) for ids in prompt_encodings["input_ids"]]

            new_log_probs_list = self._compute_log_probs(
                full_encodings["input_ids"],
                full_encodings["attention_mask"],
                response_start_indices,
            )

            # Reference model log probs for KL (CPU offloaded, no grad)
            ref_log_probs_list = self._compute_ref_log_probs(
                full_encodings["input_ids"],
                full_encodings["attention_mask"],
                response_start_indices,
            )

            mb_loss = torch.tensor(0.0, device=device)
            for i in range(len(mb_prompts)):
                new_logps = new_log_probs_list[i]
                old_logps = torch.tensor(mb_old_logps[i], device=device)
                ref_logps = ref_log_probs_list[i].detach()
                advantage = mb_advantages[i]

                min_len = min(len(new_logps), len(old_logps), len(ref_logps))
                if min_len == 0:
                    continue
                new_logps = new_logps[:min_len]
                old_logps = old_logps[:min_len]
                ref_logps = ref_logps[:min_len]

                # PPO ratio and clipping (against generation-time log probs)
                ratio = (new_logps - old_logps).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage
                ppo_loss = -torch.min(surr1, surr2).mean()

                # KL against frozen reference (detached — monitoring only)
                ref_ratio = (new_logps.detach() - ref_logps).exp()
                kl = (ref_ratio - 1) - torch.log(ref_ratio)
                kl_loss = kl.mean()

                # Loss = PPO clipped objective only. KL is for monitoring.
                mb_loss += ppo_loss
                total_clip_fraction += (
                    (ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)
                ).float().mean().item()
                total_kl += kl_loss.item()
                n_tokens += min_len

            # Scale by fraction of total batch and backward (accumulates grads)
            scaled_loss = mb_loss / n_sequences
            scaled_loss.backward()
            total_loss_val += scaled_loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.policy_version += 1

        return {
            "loss": total_loss_val,
            "kl_divergence": total_kl / n_sequences,
            "clip_fraction": total_clip_fraction / n_sequences,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "policy_version": self.policy_version,
            "n_tokens": n_tokens,
        }

    @endpoint
    async def expose_weights(self) -> dict:
        """Gather FSDP shards and expose as RDMA buffers for zero-copy weight sync.

        Returns dict of {param_name: (tensor, RDMABuffer)} that the generator
        can read_into directly. Call once after init — buffers stay valid as
        the optimizer updates tensors in-place.
        """
        from monarch.rdma import RDMABuffer

        self._weight_buffers = {}
        for k, v in self.model.state_dict().items():
            if hasattr(v, 'full_tensor'):
                t = v.full_tensor().contiguous()
            else:
                t = v.detach().contiguous()
            buf = RDMABuffer(t.view(torch.uint8).flatten())
            self._weight_buffers[k] = (t, buf)

        return self._weight_buffers

    @endpoint
    async def get_weights(self) -> bytes:
        """Gather full state dict and serialize (fallback for non-RDMA sync)."""
        state_dict = {}
        for k, v in self.model.state_dict().items():
            if hasattr(v, 'full_tensor'):
                state_dict[k] = v.full_tensor().cpu().clone()
            else:
                state_dict[k] = v.detach().cpu().clone()
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    @endpoint
    async def save_checkpoint(self, path: str) -> str:
        """Save model checkpoint in HuggingFace format."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.model.config.save_pretrained(path)
            torch.save(state_dict, f"{path}/pytorch_model.bin")
            self.tokenizer.save_pretrained(path)
        return path

    @endpoint
    async def get_stats(self) -> dict:
        return {
            "model_name": self.model_name,
            "policy_version": self.policy_version,
        }
