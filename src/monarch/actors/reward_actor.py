"""RewardActor -- scores completions on a CPU pod.

Wraps the shared binary reward function as a Monarch actor.
Runs on a separate CPU node (c7i.large) to keep GPU memory free.
"""
from typing import List

from monarch.actor import Actor, endpoint

from src.shared.reward import compute_score


class RewardActor(Actor):
    """Scores model completions against ground truth answers.

    Uses binary reward: 1.0 if the extracted answer after '####' matches
    the ground truth, 0.0 otherwise. No GPU needed.
    """

    def __init__(self):
        self.total_scored = 0

    @endpoint
    async def score(
        self,
        completions: List[str],
        ground_truths: List[str],
        data_source: str = "openai/gsm8k",
    ) -> List[float]:
        """Score a batch of completions against ground truths.

        Args:
            completions: Model-generated text outputs
            ground_truths: Expected answers
            data_source: Dataset identifier

        Returns:
            List of binary rewards (0.0 or 1.0)
        """
        rewards = [
            compute_score(data_source, completion, gt)
            for completion, gt in zip(completions, ground_truths)
        ]
        self.total_scored += len(rewards)
        return rewards

    @endpoint
    async def get_stats(self) -> dict:
        """Return scoring statistics."""
        return {"total_scored": self.total_scored}
