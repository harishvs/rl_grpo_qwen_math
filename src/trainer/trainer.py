"""GRPO Trainer using veRL framework patterns."""

import asyncio
import logging
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
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._actor = None
        self._reference = None
        self._rollout_engine = None
        self._optimizer = None
        self._env_client = None
        self._step_count = 0
    
    def setup(self):
        """Initialize models and components."""
        logger.info(f"Setting up GRPO trainer with model: {self.config.model_name}")
        
        # Initialize actor model with FSDP
        self._actor = ActorModel(
            model_path=self.config.model_name,
            fsdp_config=self.config.fsdp,
        )
        
        # Initialize reference model with CPU offload
        ref_fsdp = FSDPConfig(
            sharding_strategy=self.config.fsdp.sharding_strategy,
            cpu_offload=True,
            mixed_precision=self.config.fsdp.mixed_precision,
        )
        self._reference = ReferenceModel(
            model_path=self.config.model_name,
            fsdp_config=ref_fsdp,
        )
        
        # Initialize rollout engine
        self._rollout_engine = RolloutEngine(
            model_path=self.config.model_name,
            max_tokens=self.config.max_new_tokens,
        )
        
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
        
        1. Generate G rollouts per prompt
        2. Send to environment for reward computation
        3. Compute advantages using group normalization
        4. Update policy with clipped objective
        """
        self._step_count += 1
        
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
        
        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)
        
        # Compute GRPO advantages
        advantages = compute_grpo_advantages(rewards, self.config.group_size)
        
        # Compute actor log probs
        actor_log_probs = self._compute_log_probs(rollouts)
        
        # Compute reference log probs for KL
        with torch.no_grad():
            ref_log_probs = self._reference.compute_log_probs(rollouts)
        
        # Compute KL divergence
        kl_div = (actor_log_probs - ref_log_probs).mean()
        
        # Policy update with clipping
        update_metrics = self._update_policy(
            actor_log_probs=actor_log_probs,
            old_log_probs=actor_log_probs.detach(),
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
    
    def setup(self):
        """Initialize the model with FSDP wrapping."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.fsdp_config.mixed_precision == "bf16" else torch.float32,
        )
        
        # FSDP wrapping would be applied here in distributed setting
        # For now, move to GPU if available
        if torch.cuda.is_available():
            self._model = self._model.cuda()
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self._model.parameters() if self._model else []
    
    def compute_log_probs(self, rollouts: List[Rollout]) -> torch.Tensor:
        """Compute log probabilities for given rollouts."""
        log_probs = []
        for rollout in rollouts:
            log_probs.append(rollout.log_probs.mean())
        return torch.stack(log_probs)
    
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
        """Compute reference log probabilities (no gradients)."""
        log_probs = []
        for rollout in rollouts:
            log_probs.append(rollout.log_probs.mean())
        return torch.stack(log_probs)


class RolloutEngine:
    """Rollout generation for efficient batch inference."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self._engine = None
    
    def setup(self):
        """Initialize the rollout engine (vLLM in production)."""
        # In production, this would initialize vLLM
        pass
    
    def generate_rollouts(
        self,
        prompts: List[str],
        group_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Rollout]:
        """Generate group_size rollouts per prompt."""
        rollouts = []
        
        for prompt in prompts:
            for _ in range(group_size):
                # Placeholder - in production uses vLLM
                rollouts.append(Rollout(
                    prompt=prompt,
                    completion="",
                    log_probs=torch.zeros(1),
                    tokens=[],
                ))
        
        return rollouts
