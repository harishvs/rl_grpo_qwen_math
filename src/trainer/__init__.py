"""GRPO trainer module for RL-based LLM training."""

from src.trainer.config import TrainingConfig, FSDPConfig, RetryConfig
from src.trainer.models import (
    Rollout,
    Trajectory,
    RewardRequest,
    RewardResponse,
    TrainStepMetrics,
    PolicyUpdateMetrics,
)
from src.trainer.grpo import compute_grpo_advantages, compute_grpo_advantages_from_list
from src.trainer.dataset import MathProblem, MathProblemDataset
from src.trainer.environment_client import EnvironmentClient, EnvironmentClientConfig
from src.trainer.trainer import GRPOTrainer

__all__ = [
    "TrainingConfig",
    "FSDPConfig",
    "RetryConfig",
    "Rollout",
    "Trajectory",
    "RewardRequest",
    "RewardResponse",
    "TrainStepMetrics",
    "PolicyUpdateMetrics",
    "compute_grpo_advantages",
    "compute_grpo_advantages_from_list",
    "MathProblem",
    "MathProblemDataset",
    "EnvironmentClient",
    "EnvironmentClientConfig",
    "GRPOTrainer",
]
