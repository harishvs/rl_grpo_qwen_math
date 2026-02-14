"""GRPO Trainer using veRL framework patterns."""

import asyncio
import logging
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
        
        # Sync generation model weights from FSDP actor (every step for on-policy)
        # This is a collective operation - all ranks must participate
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
        
        # Get old log probs from generation time (before model update)
        # These were computed during rollout generation
        old_log_probs = torch.stack([
            r.log_probs.mean() if r.log_probs.numel() > 0 else torch.tensor(0.0)
            for r in rollouts
        ]).to(device).detach()
        
        # Compute NEW actor log probs through the model (with gradients)
        # This is different from old_log_probs because the model may have been updated
        actor_log_probs = self._compute_log_probs(rollouts)
        
        # KL divergence between current policy and policy at generation time
        kl_div = (actor_log_probs - old_log_probs).mean()
        
        # Ensure advantages are on the same device as actor
        advantages = advantages.to(device)
        
        # Policy update with clipping
        update_metrics = self._update_policy(
            actor_log_probs=actor_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            kl_div=kl_div,
        )
        
        # Checkpoint if needed
        if self._step_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint(
                f"{self.config.checkpoint_dir}/step_{self._step_count}"
            )
        
        return TrainStepMetrics(
            policy_loss=update_metrics.loss,
            kl_divergence=kl_div.item(),
            mean_reward=rewards.mean().item(),
            clip_fraction=update_metrics.clip_fraction,
            advantages_mean=advantages.mean().item(),
            advantages_std=advantages.std().item(),
        )
    
    def _compute_log_probs(self, rollouts: List[Rollout]) -> torch.Tensor:
        """Compute log probabilities for rollouts using actor model."""
        return self._actor.compute_log_probs(rollouts)
    
    def _update_policy(
        self,
        actor_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        kl_div: torch.Tensor,
    ) -> PolicyUpdateMetrics:
        """Apply GRPO policy update with clipping."""
        self._optimizer.zero_grad()
        
        # Compute ratio
        ratio = torch.exp(actor_log_probs - old_log_probs)
        
        # Clipped objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        )
        
        # Policy loss (negative because we maximize)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        
        # Add KL penalty
        total_loss = policy_loss + self.config.kl_coef * kl_div
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(),
            self.config.max_grad_norm,
        )
        
        # Update
        self._optimizer.step()
        
        # Compute metrics
        clip_fraction = (
            (ratio < 1.0 - self.config.clip_range) |
            (ratio > 1.0 + self.config.clip_range)
        ).float().mean().item()
        
        approx_kl = (old_log_probs - actor_log_probs).mean().item()
        
        return PolicyUpdateMetrics(
            loss=total_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )
    
    def _sync_generation_model(self):
        """Sync weights from FSDP actor to generation model.
        
        This is a collective operation - all ranks must call this.
        Only rank 0 actually copies the weights.
        """
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        is_fsdp = isinstance(self._actor._model, FSDP)
        is_distributed = self._world_size > 1 and dist.is_initialized()
        
        if is_fsdp and is_distributed:
            # All ranks must participate in summon_full_params
            with FSDP.summon_full_params(self._actor._model, writeback=False):
                if self._rank == 0:
                    # Copy weights to generation model
                    actor_state = self._actor._model.state_dict()
                    self._rollout_engine._model.load_state_dict(actor_state)
        elif self._rank == 0 and self._rollout_engine._model is not None:
            # Non-distributed case
            self._rollout_engine._model.load_state_dict(
                self._actor._model.state_dict()
            )
    
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
    
    def compute_log_probs(self, rollouts: List[Rollout]) -> torch.Tensor:
        """Compute log probabilities for given rollouts through the model.
        
        This recomputes log probs through the model to maintain gradient flow
        for policy updates.
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        log_probs_list = []
        device = next(self._model.parameters()).device
        
        for rollout in rollouts:
            # Tokenize the full sequence (prompt + completion)
            full_text = rollout.prompt + rollout.completion
            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)
            
            # Get prompt length for masking
            prompt_inputs = self._tokenizer(
                rollout.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            # Forward pass through model
            outputs = self._model(**inputs)
            logits = outputs.logits
            
            # Compute log probs for completion tokens only
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, prompt_len-1:-1, :]
            shift_labels = inputs.input_ids[:, prompt_len:]
            
            # Compute log softmax
            log_softmax = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            token_log_probs = log_softmax.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mean log prob for this rollout
            if token_log_probs.numel() > 0:
                mean_log_prob = token_log_probs.mean()
            else:
                # Empty completion - use a small tensor with grad
                mean_log_prob = torch.tensor(0.0, device=device, requires_grad=True)
            
            log_probs_list.append(mean_log_prob)
        
        return torch.stack(log_probs_list)
    
    def save(self, path: str):
        """Save model to path."""
        if self._model:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model from path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(path)
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
    
    def compute_log_probs(self, rollouts: List[Rollout]) -> torch.Tensor:
        """Compute reference log probabilities (no gradients).
        
        Computes log probs through the frozen reference model for KL divergence.
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Reference model not initialized. Call setup() first.")
        
        log_probs_list = []
        device = next(self._model.parameters()).device
        
        with torch.no_grad():
            for rollout in rollouts:
                # Tokenize the full sequence (prompt + completion)
                full_text = rollout.prompt + rollout.completion
                inputs = self._tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
                
                # Get prompt length for masking
                prompt_inputs = self._tokenizer(
                    rollout.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                prompt_len = prompt_inputs.input_ids.shape[1]
                
                # Forward pass through model
                outputs = self._model(**inputs)
                logits = outputs.logits
                
                # Compute log probs for completion tokens only
                shift_logits = logits[:, prompt_len-1:-1, :]
                shift_labels = inputs.input_ids[:, prompt_len:]
                
                # Compute log softmax
                log_softmax = F.log_softmax(shift_logits, dim=-1)
                
                # Gather log probs for actual tokens
                token_log_probs = log_softmax.gather(
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Mean log prob for this rollout
                if token_log_probs.numel() > 0:
                    mean_log_prob = token_log_probs.mean()
                else:
                    mean_log_prob = torch.tensor(0.0, device=device)
                
                log_probs_list.append(mean_log_prob)
        
        return torch.stack(log_probs_list)


class RolloutEngine:
    """Rollout generation using HuggingFace transformers.
    
    Uses a separate non-sharded model for generation to avoid FSDP complexity.
    Only rank 0 loads the generation model and generates rollouts.
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._rank = 0
        self._world_size = 1
    
    def setup(self, rank: int = 0, world_size: int = 1):
        """Initialize tokenizer and generation model (only on rank 0)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self._rank = rank
        self._world_size = world_size
        
        logger.info(f"Setting up RolloutEngine with model: {self.model_path} (rank={rank})")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Only rank 0 loads the generation model
        if rank == 0:
            logger.info("Loading separate generation model on rank 0")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
        
        logger.info("RolloutEngine setup complete")
    
    def sync_weights_from_actor(self, actor_model):
        """Sync weights from the FSDP actor model to the generation model.
        
        This should be called periodically to keep generation model updated.
        Only rank 0 needs to do this.
        """
        if self._rank != 0 or self._model is None:
            return
        
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        import torch.distributed as dist
        
        is_fsdp = isinstance(actor_model, FSDP)
        is_distributed = self._world_size > 1 and dist.is_initialized()
        
        if is_fsdp and is_distributed:
            # Use summon_full_params to get full state dict
            # All ranks must participate in this collective operation
            with FSDP.summon_full_params(actor_model, writeback=False):
                if self._rank == 0:
                    # Copy weights to generation model
                    actor_state = actor_model.state_dict()
                    self._model.load_state_dict(actor_state)
                    logger.info("Synced weights from actor to generation model")
        elif not is_fsdp:
            # Non-FSDP case - direct copy
            self._model.load_state_dict(actor_model.state_dict())
    
    def generate_rollouts(
        self,
        prompts: List[str],
        group_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Rollout]:
        """Generate group_size rollouts per prompt.
        
        Only rank 0 generates, then broadcasts results to all ranks.
        """
        import torch.distributed as dist
        
        if self._tokenizer is None:
            raise RuntimeError("RolloutEngine not initialized. Call setup() first.")
        
        is_distributed = self._world_size > 1 and dist.is_initialized()
        
        rollouts = []
        
        # Only rank 0 does actual generation
        if self._rank == 0:
            if self._model is None:
                raise RuntimeError("Generation model not loaded on rank 0")
            
            device = next(self._model.parameters()).device
            
            for prompt in prompts:
                # Tokenize prompt
                prompt_inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536,
                ).to(device)
                prompt_len = prompt_inputs.input_ids.shape[1]
                
                # Generate group_size completions for this prompt
                for _ in range(group_size):
                    with torch.no_grad():
                        outputs = self._model.generate(
                            **prompt_inputs,
                            max_new_tokens=self.max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=self._tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        
                        # Extract generated tokens
                        generated_ids = outputs.sequences[0, prompt_len:]
                        completion = self._tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                        )
                        
                        # Compute log probs from scores
                        log_probs_list = []
                        if outputs.scores and len(generated_ids) > 0:
                            for i, score in enumerate(outputs.scores):
                                if i < len(generated_ids):
                                    log_softmax = F.log_softmax(score[0], dim=-1)
                                    token_log_prob = log_softmax[generated_ids[i]]
                                    log_probs_list.append(token_log_prob.cpu())
                        
                        if log_probs_list:
                            log_probs = torch.stack(log_probs_list)
                        else:
                            log_probs = torch.zeros(1)
                    
                    rollouts.append(Rollout(
                        prompt=prompt,
                        completion=completion,
                        log_probs=log_probs,
                        tokens=generated_ids.tolist() if len(generated_ids) > 0 else [],
                    ))
        
        # Broadcast rollouts from rank 0 to all other ranks
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
        
        # Serialize rollouts on rank 0
        if self._rank == 0:
            # Convert to serializable format
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
        
        # Broadcast size
        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()
        
        # Broadcast data
        if self._rank == 0:
            data_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8, device='cuda')
        else:
            data_tensor = torch.empty(size, dtype=torch.uint8, device='cuda')
        
        dist.broadcast(data_tensor, src=0)
        
        # Deserialize on non-rank-0
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
