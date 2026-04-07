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

    def _ppo_update(self, prompts, completions, old_log_probs_list, advantages_list, device):
        """One PPO optimizer step over a mini-batch with micro-batching."""
        micro_bs = 4
        n_sequences = max(len(prompts), 1)

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0
        n_tokens = 0

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

                ratio = (new_logps - old_logps).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage
                ppo_loss = -torch.min(surr1, surr2).mean()

                ref_ratio = (new_logps - ref_logps).exp()
                kl = (ref_ratio - 1) - torch.log(ref_ratio)
                kl_loss = kl.mean()

                mb_loss += ppo_loss + self.kl_coef * kl_loss
                total_clip_fraction += (
                    (ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)
                ).float().mean().item()
                total_kl += kl_loss.item()
                n_tokens += min_len

            scaled_loss = mb_loss / n_sequences
            scaled_loss.backward()
            total_loss_val += scaled_loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return total_loss_val, total_kl / n_sequences, total_clip_fraction / n_sequences, \
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, n_tokens

    @endpoint
    async def train_step(self, batch: dict) -> dict:
        """Perform GRPO training with mini-batch splitting (like veRL).

        Splits the rollout batch into mini-batches, does a separate PPO
        optimizer step for each. 2x more weight updates from the same data.
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

        # Split into mini-batches — each gets its own optimizer step
        # 128 global mini-batch / 8 ranks = 16 per rank
        mini_batch_size = max(len(prompts) // 2, 1)  # 2 mini-batches

        total_loss = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_grad_norm = 0.0
        total_tokens = 0
        n_mini_batches = 0

        for mini_start in range(0, len(prompts), mini_batch_size):
            mini_end = min(mini_start + mini_batch_size, len(prompts))

            loss, kl, clip, grad_norm, tokens = self._ppo_update(
                prompts[mini_start:mini_end],
                completions[mini_start:mini_end],
                old_log_probs_list[mini_start:mini_end],
                advantages_list[mini_start:mini_end],
                device,
            )
            total_loss += loss
            total_kl += kl
            total_clip += clip
            total_grad_norm += grad_norm
            total_tokens += tokens
            n_mini_batches += 1

        self.policy_version += 1

        return {
            "loss": total_loss / n_mini_batches,
            "kl_divergence": total_kl / n_mini_batches,
            "clip_fraction": total_clip / n_mini_batches,
            "grad_norm": total_grad_norm / n_mini_batches,
            "policy_version": self.policy_version,
            "n_tokens": n_tokens,
        }

    @endpoint
    async def set_param_server(self, param_server) -> None:
        """Store reference to the parameter server for weight pushing."""
        self._param_server = param_server

    @endpoint
    async def push_to_param_server(self) -> None:
        """Push this rank's local FSDP shards to the parameter server.

        Each rank copies its GPU shard → CPU → sends bytes to param server.
        No FSDP collectives — each rank only touches its own data.
        """
        params = sorted(self.model.named_parameters(), key=lambda x: x[0])

        for name, param in params:
            lt = param.data._local_tensor if hasattr(param.data, '_local_tensor') else param.data
            cpu_bytes = lt.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            offset = self._shard_layout[name]
            await self._param_server.push_shard.call_one(offset, cpu_bytes)

    @endpoint
    async def init_param_server_layout(self) -> dict:
        """Compute layout for the parameter server flat buffer.

        Returns layout dict and total buffer size. All ranks return
        the same layout since FSDP shards have identical sizes per rank.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        params = sorted(self.model.named_parameters(), key=lambda x: x[0])

        layout = {}
        self._shard_layout = {}  # name -> offset for this rank's push
        offset = 0

        for name, param in params:
            lt = param.data._local_tensor if hasattr(param.data, '_local_tensor') else param.data
            shard_bytes = lt.numel() * lt.element_size()

            # Each rank's shard goes at: rank * shard_bytes + param_base_offset
            param_base = offset
            for r in range(world_size):
                if r == rank:
                    self._shard_layout[name] = param_base + r * shard_bytes
                layout_key = f"{name}__rank{r}"
                layout[layout_key] = (param_base + r * shard_bytes, shard_bytes, str(lt.dtype), list(lt.shape))

            # Full param metadata (for generator reconstruction)
            full_shape = list(param.data.shape)
            layout[name] = (param_base, shard_bytes * world_size, str(lt.dtype), full_shape)
            offset += shard_bytes * world_size

        return {"layout": layout, "total_bytes": offset, "world_size": world_size}

    @endpoint
    async def expose_shards(self) -> dict:
        """Expose this rank's local FSDP shard as a single RDMA buffer.

        No collectives — each rank copies its own GPU shards to a CPU buffer
        and registers one RDMA handle. Generator reads all 8 rank buffers.
        """
        from monarch.rdma import RDMABuffer

        rank = dist.get_rank() if dist.is_initialized() else 0
        params = sorted(self.model.named_parameters(), key=lambda x: x[0])

        # Copy local shards from GPU to one flat CPU buffer
        local_parts = []
        meta = []
        for name, param in params:
            lt = param.data._local_tensor if hasattr(param.data, '_local_tensor') else param.data
            local_parts.append(lt.detach().cpu().contiguous().view(-1).to(torch.bfloat16))
            meta.append((name, list(param.data.shape), list(lt.shape)))

        flat = torch.cat(local_parts)
        self._shard_flat = flat
        self._shard_buf = RDMABuffer(flat.view(torch.uint8).flatten())
        self._shard_params = params  # keep ordered param list for refresh

        return {
            "rank": rank,
            "buffer": (self._shard_flat, self._shard_buf),
            "meta": meta,
        }

    @endpoint
    async def refresh_shards(self) -> None:
        """Copy latest GPU weights into the RDMA CPU buffer in-place.

        No collectives — each rank copies its own ~444 MB shard. Fast (~100ms).
        RDMA buffer handle stays valid since the CPU memory address doesn't change.
        """
        offset = 0
        for name, param in self._shard_params:
            lt = param.data._local_tensor if hasattr(param.data, '_local_tensor') else param.data
            flat_shard = lt.detach().cpu().contiguous().view(-1).to(torch.bfloat16)
            self._shard_flat[offset:offset + flat_shard.numel()].copy_(flat_shard)
            offset += flat_shard.numel()

    @endpoint
    async def init_weight_sync_pg(self, store_host: str, store_port: int) -> str:
        """Join a gloo process group for direct tensor send to generator.

        Only rank 0 participates. Other ranks return immediately.
        """
        rank = dist.get_rank()
        if rank != 0:
            return "skipped"

        from torch.distributed import TCPStore
        # Create a separate PG for weight sync between learner rank 0 and generator.
        # Use gloo (CPU tensors) since ProcessGroupNCCL needs registration with global map.
        store = TCPStore(store_host, store_port, 2, True, timeout=dist.default_pg_timeout)
        from torch.distributed import ProcessGroupGloo
        self._sync_pg = ProcessGroupGloo(store, 0, 2)
        return f"rank 0 joined weight_sync gloo pg on {store_host}:{store_port}"

    @endpoint
    async def send_weights_direct(self) -> float:
        """Gather FSDP shards via NCCL, then send full tensor to generator via gloo.

        Uses state_dict() keys (HF namespace) so vLLM can load them.
        """
        import time
        t0 = time.time()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Use state_dict() — returns HF-namespace keys with DTensor values
        sd = self.model.state_dict()
        sd_keys = sorted(sd.keys())

        # Each rank: flatten local shards from state_dict DTensors
        local_parts = []
        for k in sd_keys:
            v = sd[k]
            lt = v._local_tensor if hasattr(v, '_local_tensor') else v.detach()
            local_parts.append(lt.contiguous().view(-1))
        local_flat = torch.cat(local_parts)

        # NCCL all_gather
        if world_size > 1:
            gathered = torch.empty(local_flat.numel() * world_size, dtype=local_flat.dtype, device=local_flat.device)
            dist.all_gather_into_tensor(gathered, local_flat)
        else:
            gathered = local_flat

        # Rank 0: send to generator via gloo
        if rank == 0 and hasattr(self, '_sync_pg'):
            cpu_tensor = gathered.cpu().contiguous()
            self._sync_pg.send([cpu_tensor], 1, 0).wait()

        return time.time() - t0

    @endpoint
    async def get_weight_meta(self) -> list:
        """Return param metadata using state_dict() keys (HF namespace).

        vLLM expects HF keys, not FSDP named_parameters() keys.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        sd = self.model.state_dict()
        sd_keys = sorted(sd.keys())

        meta = []
        shard_offset = 0
        for k in sd_keys:
            v = sd[k]
            lt = v._local_tensor if hasattr(v, '_local_tensor') else v.detach()
            meta.append((k, list(v.shape), lt.numel(), shard_offset))
            shard_offset += lt.numel()

        return {"meta": meta, "shard_size": shard_offset, "world_size": world_size,
                "dtype": str(next(iter(sd.values())).dtype)}

    @endpoint
    async def gather_weights_nccl(self) -> bytes:
        """Gather FSDP shards via one NCCL all_gather, then serialize.

        Instead of 339 per-param full_tensor() calls, each rank flattens
        its local shards into one tensor, then one all_gather_into_tensor
        collects everything on rank 0. One collective instead of 339.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        params = sorted(self.model.named_parameters(), key=lambda x: x[0])

        # Each rank: flatten all local shards into one contiguous GPU tensor
        local_parts = []
        meta = []  # (name, full_shape, shard_numel) for reconstruction
        for name, param in params:
            lt = param.data._local_tensor if hasattr(param.data, '_local_tensor') else param.data
            local_parts.append(lt.detach().contiguous().view(-1))
            meta.append((name, list(param.data.shape), lt.numel()))

        local_flat = torch.cat(local_parts)  # ~444 MB on GPU

        # One NCCL all_gather: all ranks contribute, rank 0 gets full buffer
        if world_size > 1:
            gathered = torch.empty(local_flat.numel() * world_size, dtype=local_flat.dtype, device=local_flat.device)
            dist.all_gather_into_tensor(gathered, local_flat)
        else:
            gathered = local_flat

        # Rank 0: reconstruct state_dict from gathered shards and serialize
        # gathered layout: [rank0_all_params | rank1_all_params | ... | rank7_all_params]
        if rank == 0:
            state_dict = {}
            shard_size = local_flat.numel()  # total elements per rank

            # Precompute param offsets within each rank's shard
            param_offsets = []
            offset = 0
            for _, _, shard_numel in meta:
                param_offsets.append(offset)
                offset += shard_numel

            for i, (name, full_shape, shard_numel) in enumerate(meta):
                param_shards = []
                for r in range(world_size):
                    start = r * shard_size + param_offsets[i]
                    param_shards.append(gathered[start:start + shard_numel])
                full_param = torch.cat(param_shards)
                full_numel = 1
                for s in full_shape:
                    full_numel *= s
                state_dict[name] = full_param[:full_numel].reshape(full_shape).cpu()

            buf = io.BytesIO()
            torch.save(state_dict, buf)
            return buf.getvalue()
        return b""

    @endpoint
    async def get_weights(self) -> bytes:
        """Gather full state dict and serialize (fallback, 339 collectives)."""
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
