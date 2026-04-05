"""FastAPI reward service for GSM8K math problems.

Shared across implementations that need an HTTP-based reward service.
Uses compute_score from src/shared/reward.py for the actual scoring.

Usage:
    uvicorn src.shared.reward_service:app --host 0.0.0.0 --port 8080
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.shared.reward import compute_score


SERVICE_VERSION = "2.0.0"
REQUEST_TIMEOUT_SECONDS = 30.0


# --- Pydantic models ---

class Trajectory(BaseModel):
    prompt: str = Field(..., description="The math problem prompt")
    completion: str = Field(..., description="Model-generated completion")
    ground_truth: str = Field(..., description="Expected numeric answer")


class RewardRequest(BaseModel):
    trajectories: List[Trajectory] = Field(..., description="Trajectories to evaluate")


class ScoreBatchRequest(BaseModel):
    """Simpler interface: just completions and ground truths."""
    completions: List[str]
    ground_truths: List[str]
    data_source: str = "openai/gsm8k"


class RewardDetails(BaseModel):
    correctness_score: float
    extracted_answer: Optional[str] = None
    is_correct: bool


class RewardResponse(BaseModel):
    rewards: List[float]
    details: List[RewardDetails]


class HealthResponse(BaseModel):
    status: str
    version: str


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="Reward Service",
    description="Shared reward computation for GRPO training",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", version=SERVICE_VERSION)


@app.post("/compute_rewards", response_model=RewardResponse)
async def compute_rewards(request: RewardRequest) -> RewardResponse:
    """Score trajectories (custom trainer compatible interface)."""
    if not request.trajectories:
        return RewardResponse(rewards=[], details=[])

    try:
        result = await asyncio.wait_for(
            _score_trajectories(request.trajectories),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Timed out after {REQUEST_TIMEOUT_SECONDS}s",
        )


@app.post("/score")
async def score_batch(request: ScoreBatchRequest) -> List[float]:
    """Score completions against ground truths (simpler interface for Monarch)."""
    if len(request.completions) != len(request.ground_truths):
        raise HTTPException(400, "completions and ground_truths must have same length")

    return [
        compute_score(request.data_source, c, gt)
        for c, gt in zip(request.completions, request.ground_truths)
    ]


async def _score_trajectories(trajectories: List[Trajectory]) -> RewardResponse:
    loop = asyncio.get_event_loop()
    rewards = []
    details = []

    for t in trajectories:
        reward = await loop.run_in_executor(
            None,
            compute_score,
            "openai/gsm8k",
            t.completion,
            t.ground_truth,
        )
        rewards.append(reward)
        details.append(RewardDetails(
            correctness_score=reward,
            is_correct=reward == 1.0,
        ))

    return RewardResponse(rewards=rewards, details=details)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
