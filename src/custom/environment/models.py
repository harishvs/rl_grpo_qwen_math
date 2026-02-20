"""Data models for the environment service."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Trajectory(BaseModel):
    """A trajectory to be evaluated."""
    prompt: str = Field(..., description="The math problem prompt")
    completion: str = Field(..., description="Model-generated completion")
    ground_truth: str = Field(..., description="Expected numeric answer")


class RewardRequest(BaseModel):
    """Request to compute rewards for trajectories."""
    trajectories: List[Trajectory] = Field(
        ..., 
        description="List of trajectories to evaluate"
    )


class RewardDetails(BaseModel):
    """Breakdown of reward computation."""
    correctness_score: float = Field(..., description="1.0 if correct, 0.0 otherwise")
    format_score: float = Field(..., description="Format compliance score (0.0-0.2)")
    extracted_answer: Optional[str] = Field(None, description="Extracted answer from completion")
    is_correct: bool = Field(..., description="Whether the answer is correct")


class RewardResponse(BaseModel):
    """Response containing computed rewards."""
    rewards: List[float] = Field(..., description="Scalar rewards for each trajectory")
    details: List[RewardDetails] = Field(..., description="Detailed breakdown per trajectory")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
