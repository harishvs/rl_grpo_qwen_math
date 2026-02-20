"""GRPO Trainer using veRL framework patterns."""

import asyncio
import logging
import os
from contextlib import nullcontext
from typing import List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.trainer.config import TrainingConfig, FSDPConfig
from src.trainer.models import (
    Rollout,
    Trajectory,
    TrainStepMetrics,
    PolicyUpdateMetrics,
)
from src.trainer.grpo import compute_grpo_advantages
from src.trainer.environment_client import EnvironmentClient, EnvironmentClientConfig
from src.trainer.dataset import MathProblem


logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Orchestrates the GRPO training loop using veRL patterns.
    
    Manages actor model, reference model, and communication with environment service.
    Supports distributed training with FSDP.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._actor = None
        self._reference = None
        self._rollout_engine = None
        self._optimizer = None
        self._env_client = None
        self._step_count = 0
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0
    
    def setup(self, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        """Initialize models and components with distributed support."""
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        
        logger.info(f"Setting up GRPO trainer with model: {self.config.model_name}")
        logger.info(f"Distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize actor model with FSDP
        self._actor = ActorModel(
            model_path=self.config.model_name,
            fsdp_config=self.config.fsdp,
        )
        self._actor.setup(local_rank=local_rank, world_size=world_size)
        
        # Skip reference model to save GPU memory - use old_log_probs for KL instead
        self._reference = None
        
        # Initialize rollout engine with separate generation model (only on rank 0)
        self._rollout_engine = RolloutEngine(
            model_path=self.config.model_name,
            max_tokens=self.config.max_new_tokens,
        )
        self._rollout_engine.setup(rank=rank, world_size=world_size)
        
        # Initialize optimizer
        self._optimizer = torch.optim.AdamW(
            self._actor.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Initialize environment client
        env_config = EnvironmentClientConfig(
            base_url=self.config.environment_service_url
        )
        self._env_client = EnvironmentClient(env_config)
        
        # Initialize NCCL weight sync group to vLLM server (rank 0 only)
        self._init_weight_sync_group()
        
        logger.info("GRPO trainer setup complete")
    
    async def train_step(
        self,
        problems: List[MathProblem],
    ) -> TrainStepMetrics:
        """Execute one GRPO training step.
        
        1. Sync generation model weights (if distributed)
        2. Generate G rollouts per prompt
        3. Send to environment for reward computation
        4. Compute advantages using group normalization
        5. Update policy with clipped objective
        """
        self._step_count += 1
        
        # Sync generation model weights from FSDP actor (every 3 steps to reduce overhead)
        # This is a collective operation - all ranks must participate
        if self._step_count % 3 == 1 or self._step_count == 1:
            self._sync_generation_model()
        
        # Format prompts
        prompts = [p.to_prompt() for p in problems]
        ground_truths = [p.answer for p in problems]
        
        # Generate rollouts
        rollouts = self._rollout_engine.generate_rollouts(
            prompts=prompts,
            group_size=self.config.group_size,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        # Build trajectories for environment
        trajectories = []
        for i, rollout in enumerate(rollouts):
            prompt_idx = i // self.config.group_size
            trajectories.append(Trajectory(
                prompt=rollout.prompt,
                completion=rollout.completion,
                ground_truth=ground_truths[prompt_idx],
            ))
        
        # Get rewards from environment
        async with self._env_client as client:
            reward_response = await client.compute_rewards(trajectories)
        
        # Get device from actor model
        device = next(self._actor.parameters()).device
        
        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32, device=device)
        
        # Compute GRPO advantages
        advantages = compute_grpo_advantages(rewards, self.config.group_size)
        
        # Get old log probs from generation time (per-token, before model update)
        old_log_probs_per_token = [
            r.log_probs.to(device).detach() if r.log_probs.numel() > 0
            else torch.zeros(1, device=device)
            for r in rollouts
        ]
        
        # Ensure advantages are on the same device as actor
        advantages = advantages.to(device)
        
        # Policy update with per-token clipping and gradient accumulation
        update_metrics, kl_div = self._update_policy(
            rollouts=rollouts,
            old_log_probs_per_token=old_log_probs_per_token,
            advantages=advantages,
        )
        
        # Checkpoint if needed
        if self._step_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint(
                f"{self.config.checkpoint_dir}/step_{self._step_count}"
            )
        
        return TrainStepMetrics(
            policy_loss=update_metrics.loss,
            kl_divergence=kl_div,
            mean_reward=rewards.mean().item(),
            clip_fraction=update_metrics.clip_fraction,
            advantages_mean=advantages.mean().item(),
            advantages_std=advantages.std().item(),
        )
    
    def _compute_log_probs(self, rollouts: List[Rollout]) -> List[torch.Tensor]:
        """Compute per-token log probabilities for rollouts using actor model."""
        return self._actor.compute_log_probs(rollouts)
    
    def _update_policy(
        self,
        rollouts: List[Rollout],
        old_log_probs_per_token: List[torch.Tensor],
        advantages: torch.Tensor,
    ) -> tuple:
        """Apply GRPO policy update with per-token clipping and gradient accumulation.
        
        Processes rollouts in micro-batches to balance memory and speed.
        """
        self._optimizer.zero_grad()
        device = advantages.device
        n_rollouts = len(rollouts)
        micro_batch_size = 4  # Process 4 rollouts at a time (batched forward pass)
        
        total_loss_val = 0.0
        total_kl_val = 0.0
        all_ratios = []
        n_valid = 0
        
        for mb_start in range(0, n_rollouts, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, n_rollouts)
            mb_rollouts = rollouts[mb_start:mb_end]
            mb_old_lps = old_log_probs_per_token[mb_start:mb_end]
            mb_advs = advantages[mb_start:mb_end]
            
            # Forward pass for micro-batch
            mb_new_lps = self._actor.compute_log_probs(mb_rollouts)
            
            mb_losses = []
            mb_kl_sum = 0.0
            mb_valid = 0
            
            for j, (new_lp, old_lp) in enumerate(zip(mb_new_lps, mb_old_lps)):
                min_len = min(len(new_lp), len(old_lp))
                if min_len == 0:
                    continue
                
                token_ratio = torch.exp(new_lp[:min_len] - old_lp[:min_len])
                all_ratios.append(token_ratio.detach())
                
                kl_tokens = (token_ratio.detach() - 1) - torch.log(token_ratio.detach())
                mb_kl_sum += kl_tokens.mean().item()
                
                adv = mb_advs[j]
                clipped = torch.clamp(token_ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                token_loss = -torch.min(token_ratio * adv, clipped * adv)
                mb_losses.append(token_loss.mean() + self.config.kl_coef * kl_tokens.mean())
                mb_valid += 1
            
            if mb_losses:
                mb_loss = torch.stack(mb_losses).mean() / (n_rollouts / len(mb_losses))
                mb_loss.backward()
                total_loss_val += sum(l.item() for l in mb_losses)
                total_kl_val += mb_kl_sum
                n_valid += mb_valid
        
        torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(),
            self.config.max_grad_norm,
        )
        self._optimizer.step()
        
        avg_loss = total_loss_val / max(n_valid, 1)
        avg_kl = total_kl_val / max(n_valid, 1)
        
        if all_ratios:
            all_r = torch.cat(all_ratios)
            clip_fraction = (
                (all_r < 1.0 - self.config.clip_range) |
                (all_r > 1.0 + self.config.clip_range)
            ).float().mean().item()
        else:
            clip_fraction = 0.0
        
        return PolicyUpdateMetrics(
            loss=avg_loss,
            clip_fraction=clip_fraction,
            approx_kl=avg_kl,
        ), avg_kl
    
    def _init_weight_sync_group(self):
        """No-op — weight sync uses HTTP POST to vLLM server."""
        pass

    def _sync_generation_model(self):
        """Push weights from FSDP actor to vLLM server via HTTP.
        
        All ranks must call this (summon_full_params is collective).
        Only rank 0 sends weights to vLLM server.
        """
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        import io
        import requests

        is_fsdp = isinstance(self._actor._model, FSDP)
        is_distributed = self._world_size > 1 and dist.is_initialized()

        if not (is_fsdp and is_distributed):
            return

        with FSDP.summon_full_params(self._actor._model, writeback=False):
            if self._rank == 0:
                try:
                    import time as _t
                    t0 = _t.time()
                    state_dict = self._actor._model.state_dict()
                    t1 = _t.time()
                    buf = io.BytesIO()
                    torch.save(state_dict, buf)
                    data = buf.getvalue()
                    t2 = _t.time()
                    vllm_url = self._rollout_engine._vllm_url
                    resp = requests.post(
                        f"{vllm_url}/update_weights",
                        data=data,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    t3 = _t.time()
                    result = resp.json()
                    logger.info(f"Weight sync: state_dict={t1-t0:.1f}s serialize={t2-t1:.1f}s http={t3-t2:.1f}s server={result['duration_ms']:.0f}ms")
                except Exception as e:
                    logger.error(f"Weight sync failed: {e}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        self._actor.save(path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        self._actor.load(path)


class ActorModel:
    """Policy model with FSDP sharding for distributed training."""
    
    def __init__(self, model_path: str, fsdp_config: FSDPConfig):
        self.model_path = model_path
        self.fsdp_config = fsdp_config
        self._model = None
        self._tokenizer = None
        self._local_rank = 0
        self._world_size = 1
    
    def setup(self, local_rank: int = 0, world_size: int = 1):
        """Initialize the model with FSDP wrapping for distributed training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial
        import torch.distributed as dist
        
        self._local_rank = local_rank
        self._world_size = world_size
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        dtype = torch.bfloat16 if self.fsdp_config.mixed_precision == "bf16" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )
        
        # Enable gradient checkpointing to reduce memory usage
        self._model.gradient_checkpointing_enable()
        
        # Move to GPU
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            self._model = self._model.to(device)
        
        # Apply FSDP if distributed
        if world_size > 1 and dist.is_initialized():
            logger.info(f"Wrapping model with FSDP (world_size={world_size})")
            
            # Get the transformer layer class for auto wrapping
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen2DecoderLayer},
            )
            
            # Mixed precision config
            mp_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )
            
            # Sharding strategy
            sharding_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }
            sharding = sharding_map.get(
                self.fsdp_config.sharding_strategy,
                ShardingStrategy.FULL_SHARD
            )
            
            self._model = FSDP(
                self._model,
                sharding_strategy=sharding,
                mixed_precision=mp_policy,
                auto_wrap_policy=auto_wrap_policy,
                device_id=local_rank,
            )
            logger.info("FSDP wrapping complete")
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self._model.parameters() if self._model else []
    
    def compute_log_probs(self, rollouts: List[Rollout]) -> List[torch.Tensor]:
        """Compute per-token log probabilities for given rollouts (batched).
        
        Returns a list of 1D tensors (one per rollout), each containing
        per-token log probs with gradient flow for policy updates.
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        device = next(self._model.parameters()).device
        
        # Tokenize all sequences and compute prompt lengths
        full_texts = [r.prompt + r.completion for r in rollouts]
        prompt_lens = []
        for r in rollouts:
            p = self._tokenizer(r.prompt, truncation=True, max_length=2048)
            prompt_lens.append(len(p.input_ids))
        
        # Batch tokenize with padding
        batch = self._tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(device)
        
        # Single batched forward pass
        outputs = self._model(**batch)
        logits = outputs.logits  # (B, seq_len, vocab)
        
        # Extract per-rollout completion log probs
        log_probs_list = []
        input_ids = batch.input_ids  # (B, seq_len)
        attention_mask = batch.attention_mask  # (B, seq_len)
        
        for i in range(len(rollouts)):
            plen = prompt_lens[i]
            # Find actual (non-pad) length for this sequence
            seq_len = attention_mask[i].sum().item()
            
            if seq_len <= plen:
                log_probs_list.append(torch.zeros(1, device=device, requires_grad=True))
                continue
            
            shift_logits = logits[i, plen-1:seq_len-1, :]  # (comp_len, vocab)
            shift_labels = input_ids[i, plen:seq_len]       # (comp_len,)
            
            log_softmax = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_softmax.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)  # (comp_len,)
            
            log_probs_list.append(token_log_probs)
        
        return log_probs_list
    
    def save(self, path: str):
        """Save unflattened FSDP checkpoint in standard HF format."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
        import torch.distributed as dist

        if not self._model:
            return

        if isinstance(self._model, FSDP) and self._world_size > 1 and dist.is_initialized():
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self._model.state_dict()
                if self._rank == 0:
                    from transformers import AutoModelForCausalLM
                    dtype = torch.bfloat16 if self.fsdp_config.mixed_precision == "bf16" else torch.float32
                    ref = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype)
                    ref.load_state_dict(state_dict)
                    ref.save_pretrained(path)
                    self._tokenizer.save_pretrained(path)
                    del ref
            dist.barrier()
        else:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load standard HF checkpoint into FSDP-wrapped model."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if isinstance(self._model, FSDP) and self._world_size > 1 and dist.is_initialized():
            dtype = torch.bfloat16 if self.fsdp_config.mixed_precision == "bf16" else torch.float32
            full_state = None
            if self._local_rank == 0:
                tmp = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
                full_state = tmp.state_dict()
                del tmp

            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT, load_policy):
                self._model.load_state_dict(full_state or {})
            dist.barrier()
        else:
            self._model = AutoModelForCausalLM.from_pretrained(path)


class ReferenceModel:
    """Frozen reference policy for KL divergence computation."""
    
    def __init__(self, model_path: str, fsdp_config: FSDPConfig):
        self.model_path = model_path
        self.fsdp_config = fsdp_config
        self._model = None
        self._tokenizer = None
    
    def setup(self):
        """Initialize the frozen reference model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.fsdp_config.mixed_precision == "bf16" else torch.float32,
        )
        
        # Freeze parameters
        for param in self._model.parameters():
            param.requires_grad = False
        
        # CPU offload if configured
        if not self.fsdp_config.cpu_offload and torch.cuda.is_available():
            self._model = self._model.cuda()
    
    def compute_log_probs(self, rollouts: List[Rollout]) -> List[torch.Tensor]:
        """Compute per-token reference log probabilities (no gradients)."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Reference model not initialized. Call setup() first.")
        
        log_probs_list = []
        device = next(self._model.parameters()).device
        
        with torch.no_grad():
            for rollout in rollouts:
                full_text = rollout.prompt + rollout.completion
                inputs = self._tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
                
                prompt_inputs = self._tokenizer(
                    rollout.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                prompt_len = prompt_inputs.input_ids.shape[1]
                
                outputs = self._model(**inputs)
                logits = outputs.logits
                
                shift_logits = logits[:, prompt_len-1:-1, :]
                shift_labels = inputs.input_ids[:, prompt_len:]
                
                log_softmax = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_softmax.gather(
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1).squeeze(0)
                
                if token_log_probs.numel() == 0:
                    token_log_probs = torch.zeros(1, device=device)
                
                log_probs_list.append(token_log_probs)
        
        return log_probs_list


class RolloutEngine:
    """Rollout generation via remote vLLM server.
    
    Sends HTTP requests to a separate vLLM pod for generation.
    Only rank 0 calls the server, then broadcasts results to all ranks.
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._tokenizer = None
        self._rank = 0
        self._world_size = 1
        self._vllm_url = os.environ.get("VLLM_SERVER_URL", "http://vllm-server:8000")
    
    def setup(self, rank: int = 0, world_size: int = 1):
        """Initialize tokenizer (all ranks) — no local generation model needed."""
        from transformers import AutoTokenizer
        
        self._rank = rank
        self._world_size = world_size
        
        logger.info(f"Setting up RolloutEngine (HTTP client to {self._vllm_url}, rank={rank})")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        logger.info("RolloutEngine setup complete")
    
    def sync_weights_from_actor(self, actor_model):
        """No-op — weight sync handled by GRPOTrainer via NCCL."""
        pass
    
    def generate_rollouts(
        self,
        prompts: List[str],
        group_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Rollout]:
        """Generate rollouts via HTTP call to vLLM server.
        
        Only rank 0 calls the server, then broadcasts results to all ranks.
        """
        import torch.distributed as dist
        import requests
        
        if self._tokenizer is None:
            raise RuntimeError("RolloutEngine not initialized. Call setup() first.")
        
        is_distributed = self._world_size > 1 and dist.is_initialized()
        
        rollouts = []
        
        if self._rank == 0:
            import time as _time
            for _attempt in range(30):
                try:
                    resp = requests.post(
                        f"{self._vllm_url}/generate",
                        json={
                            "prompts": prompts,
                            "group_size": group_size,
                            "max_tokens": self.max_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                        },
                        timeout=300,
                    )
                    resp.raise_for_status()
                    break
                except (requests.ConnectionError, requests.Timeout) as e:
                    logger.warning(f"vLLM server unavailable (attempt {_attempt+1}/30): {e}")
                    _time.sleep(10)
            else:
                raise RuntimeError("vLLM server unreachable after 30 attempts")
            data = resp.json()
            
            for c in data["completions"]:
                log_probs = torch.tensor(c["log_probs"], dtype=torch.float32) if c["log_probs"] else torch.zeros(1)
                rollouts.append(Rollout(
                    prompt=c["prompt"],
                    completion=c["completion"],
                    log_probs=log_probs,
                    tokens=c["token_ids"],
                ))
        
        if is_distributed:
            rollouts = self._broadcast_rollouts(rollouts, prompts, group_size)
        
        return rollouts
    
    def _broadcast_rollouts(
        self,
        rollouts: List[Rollout],
        prompts: List[str],
        group_size: int,
    ) -> List[Rollout]:
        """Broadcast rollouts from rank 0 to all ranks."""
        import torch.distributed as dist
        import pickle
        
        if self._rank == 0:
            rollout_data = []
            for r in rollouts:
                rollout_data.append({
                    'prompt': r.prompt,
                    'completion': r.completion,
                    'log_probs': r.log_probs.tolist(),
                    'tokens': r.tokens,
                })
            data_bytes = pickle.dumps(rollout_data)
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device='cuda')
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device='cuda')
        
        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()
        
        if self._rank == 0:
            data_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8, device='cuda')
        else:
            data_tensor = torch.empty(size, dtype=torch.uint8, device='cuda')
        
        dist.broadcast(data_tensor, src=0)
        
        if self._rank != 0:
            data_bytes = bytes(data_tensor.cpu().tolist())
            rollout_data = pickle.loads(data_bytes)
            rollouts = []
            for rd in rollout_data:
                rollouts.append(Rollout(
                    prompt=rd['prompt'],
                    completion=rd['completion'],
                    log_probs=torch.tensor(rd['log_probs']),
                    tokens=rd['tokens'],
                ))
        
        return rollouts
