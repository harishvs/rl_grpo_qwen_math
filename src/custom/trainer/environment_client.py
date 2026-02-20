"""Async HTTP client for the environment service."""

import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass

import aiohttp

from src.custom.trainer.config import RetryConfig
from src.custom.trainer.models import Trajectory, RewardDetails, RewardResponse


logger = logging.getLogger(__name__)


@dataclass
class EnvironmentClientConfig:
    """Configuration for the environment client."""
    base_url: str = "http://environment-service:8080"
    retry: RetryConfig = None
    
    def __post_init__(self):
        if self.retry is None:
            self.retry = RetryConfig()


class EnvironmentClient:
    """Async HTTP client for communicating with the environment service.
    
    Implements retry logic with exponential backoff and timeout handling.
    """
    
    def __init__(self, config: Optional[EnvironmentClientConfig] = None):
        self.config = config or EnvironmentClientConfig()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self) -> "EnvironmentClient":
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.retry.timeout_ms / 1000
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def compute_rewards(
        self,
        trajectories: List[Trajectory],
    ) -> RewardResponse:
        """Send trajectories to environment service and get rewards.
        
        Args:
            trajectories: List of trajectories to evaluate
            
        Returns:
            RewardResponse with rewards and details
            
        Raises:
            EnvironmentServiceError: If all retries fail
        """
        await self._ensure_session()
        
        payload = {
            "trajectories": [
                {
                    "prompt": t.prompt,
                    "completion": t.completion,
                    "ground_truth": t.ground_truth,
                }
                for t in trajectories
            ]
        }
        
        url = f"{self.config.base_url}/compute_rewards"
        
        last_error = None
        backoff_ms = self.config.retry.initial_backoff_ms
        
        for attempt in range(self.config.retry.max_retries + 1):
            try:
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_response(data)
                    else:
                        error_text = await response.text()
                        last_error = EnvironmentServiceError(
                            f"HTTP {response.status}: {error_text}"
                        )
            except asyncio.TimeoutError:
                last_error = EnvironmentServiceError("Request timed out")
            except aiohttp.ClientError as e:
                last_error = EnvironmentServiceError(f"Client error: {e}")
            
            if attempt < self.config.retry.max_retries:
                logger.warning(
                    f"Environment service request failed (attempt {attempt + 1}), "
                    f"retrying in {backoff_ms}ms: {last_error}"
                )
                await asyncio.sleep(backoff_ms / 1000)
                backoff_ms = min(
                    int(backoff_ms * self.config.retry.backoff_multiplier),
                    self.config.retry.max_backoff_ms,
                )
        
        if self.config.retry.skip_on_failure:
            logger.error(f"All retries failed, returning zero rewards: {last_error}")
            return RewardResponse(
                rewards=[0.0] * len(trajectories),
                details=[
                    RewardDetails(
                        correctness_score=0.0,
                        format_score=0.0,
                        extracted_answer=None,
                        is_correct=False,
                    )
                    for _ in trajectories
                ],
            )
        
        raise last_error
    
    def _parse_response(self, data: dict) -> RewardResponse:
        """Parse JSON response into RewardResponse."""
        details = [
            RewardDetails(
                correctness_score=d["correctness_score"],
                format_score=d["format_score"],
                extracted_answer=d.get("extracted_answer"),
                is_correct=d["is_correct"],
            )
            for d in data["details"]
        ]
        return RewardResponse(rewards=data["rewards"], details=details)
    
    async def health_check(self) -> bool:
        """Check if the environment service is healthy."""
        await self._ensure_session()
        
        try:
            url = f"{self.config.base_url}/health"
            async with self._session.get(url) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


class EnvironmentServiceError(Exception):
    """Error communicating with the environment service."""
    pass
