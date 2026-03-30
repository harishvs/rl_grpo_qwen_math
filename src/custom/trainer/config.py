"""Configuration dataclasses for GRPO training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSDPConfig:
    """FSDP configuration for distributed training."""
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    mixed_precision: str = "bf16"
    reshard_after_forward: bool = True


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    
    # GRPO hyperparameters
    group_size: int = 8
    learning_rate: float = 1e-6
    kl_coef: float = 0.1 #kl_coef controls how much the model is penalized for drifting away from its original behavior.
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    
    # Training loop
    batch_size: int = 32
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Infrastructure
    environment_service_url: str = "http://environment-service:8080"
    checkpoint_dir: str = "/checkpoints"
    checkpoint_interval: int = 100
    
    # Dataset
    max_samples: Optional[int] = None
    
    # FSDP config
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 5000
    backoff_multiplier: float = 2.0
    timeout_ms: int = 30000
    skip_on_failure: bool = True
