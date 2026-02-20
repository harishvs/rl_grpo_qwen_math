"""Data models for the GRPO trainer."""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class Rollout:
    """A single generated trajectory from the model."""
    prompt: str
    completion: str
    log_probs: torch.Tensor
    tokens: List[int]


@dataclass
class Trajectory:
    """A trajectory to be evaluated by the environment."""
    prompt: str
    completion: str
    ground_truth: str


@dataclass
class RewardRequest:
    """Request to compute rewards for trajectories."""
    trajectories: List[Trajectory]


@dataclass
class RewardDetails:
    """Breakdown of reward computation."""
    correctness_score: float
    format_score: float
    extracted_answer: Optional[str]
    is_correct: bool


@dataclass
class RewardResponse:
    """Response containing computed rewards."""
    rewards: List[float]
    details: List[RewardDetails]


@dataclass
class TrainStepMetrics:
    """Metrics from a single training step."""
    policy_loss: float
    kl_divergence: float
    mean_reward: float
    clip_fraction: float
    advantages_mean: float
    advantages_std: float


@dataclass
class PolicyUpdateMetrics:
    """Metrics from policy update."""
    loss: float
    clip_fraction: float
    approx_kl: float
