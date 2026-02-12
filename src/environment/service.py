"""
FastAPI environment service for reward computation.

This service runs on CPU nodes and provides reward computation
for the GRPO training loop running on GPU nodes.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.environment.models import (
    HealthResponse,
    RewardDetails,
    RewardRequest,
    RewardResponse,
)
from src.environment.reward import RewardWorker


# Service configuration
SERVICE_VERSION = "1.0.0"
REQUEST_TIMEOUT_SECONDS = 30.0


# Global reward worker instance
reward_worker: RewardWorker = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and cleanup resources."""
    global reward_worker
    reward_worker = RewardWorker()
    yield
    # Cleanup if needed


app = FastAPI(
    title="Environment Service",
    description="Reward computation service for RL training",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint for Kubernetes probes.
    
    Returns service status and version.
    """
    return HealthResponse(status="healthy", version=SERVICE_VERSION)


@app.post("/compute_rewards", response_model=RewardResponse)
async def compute_rewards(request: RewardRequest) -> RewardResponse:
    """
    Compute rewards for a batch of trajectories.
    
    Args:
        request: Contains list of (prompt, completion, ground_truth) tuples
        
    Returns:
        RewardResponse with scalar rewards and details for each trajectory
        
    Raises:
        HTTPException: On timeout or processing errors
    """
    if not request.trajectories:
        return RewardResponse(rewards=[], details=[])
    
    try:
        # Run reward computation with timeout
        result = await asyncio.wait_for(
            _compute_rewards_async(request),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing rewards: {str(e)}",
        )


async def _compute_rewards_async(request: RewardRequest) -> RewardResponse:
    """
    Async wrapper for reward computation.
    
    Runs the synchronous reward computation in a thread pool
    to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    
    # Compute rewards for all trajectories
    rewards = []
    details = []
    
    for trajectory in request.trajectories:
        # Run synchronous computation in thread pool
        reward = await loop.run_in_executor(
            None,
            reward_worker.compute_reward,
            trajectory.prompt,
            trajectory.completion,
            trajectory.ground_truth,
        )
        
        # Get detailed breakdown
        extracted = reward_worker.extract_answer(trajectory.completion)
        format_score = reward_worker.check_format(trajectory.completion)
        correctness_score = reward - format_score
        
        is_correct = correctness_score == 1.0
        
        rewards.append(reward)
        details.append(RewardDetails(
            correctness_score=correctness_score,
            format_score=format_score,
            extracted_answer=extracted,
            is_correct=is_correct,
        ))
    
    return RewardResponse(rewards=rewards, details=details)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
