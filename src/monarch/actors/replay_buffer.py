"""ReplayBufferActor -- stores scored episodes between generation and training.

Decouples generation from training so they can run at different speeds.
Tracks policy versions to control staleness (off-policy degree).
"""
import random
from typing import List, Optional

from monarch.actor import Actor, endpoint


class ReplayBufferActor(Actor):
    """Stores scored episodes with policy version tracking.

    In synchronous mode (max_policy_age=0), only returns episodes from the
    current policy version. Increasing max_policy_age enables async training
    where the learner trains on slightly stale data while the generator
    produces new episodes.
    """

    def __init__(self, max_size: int = 10000, max_policy_age: int = 0):
        self.storage: List[dict] = []  # (episode_dict, policy_version)
        self.max_size = max_size
        self.max_policy_age = max_policy_age
        self.total_added = 0
        self.total_sampled = 0

    @endpoint
    async def add(self, episodes: dict, policy_version: int) -> None:
        """Add scored episodes to the buffer.

        Args:
            episodes: Dict with 'prompts', 'completions', 'log_probs',
                     'rewards', 'advantages'
            policy_version: The training step that produced these episodes
        """
        self.storage.append({"data": episodes, "version": policy_version})
        self.total_added += 1

        # Evict oldest if over capacity
        if len(self.storage) > self.max_size:
            self.storage = self.storage[-self.max_size:]

    @endpoint
    async def sample(
        self,
        batch_size: int,
        current_version: int,
    ) -> Optional[dict]:
        """Sample a batch of episodes respecting policy age.

        Args:
            batch_size: Number of episode groups to sample
            current_version: Current policy version (for staleness filtering)

        Returns:
            Sampled episodes dict, or None if buffer is empty/no valid episodes
        """
        # Filter by policy age
        min_version = current_version - self.max_policy_age
        valid = [e for e in self.storage if e["version"] >= min_version]

        if not valid:
            return None

        # Sample with replacement if needed
        chosen = random.choices(valid, k=min(batch_size, len(valid)))
        self.total_sampled += len(chosen)

        # Merge sampled episodes into a single batch
        merged = {
            "prompts": [],
            "completions": [],
            "log_probs": [],
            "rewards": [],
            "advantages": [],
        }
        for item in chosen:
            data = item["data"]
            for key in merged:
                if key in data:
                    merged[key].extend(data[key])

        return merged

    @endpoint
    async def buffer_size(self) -> int:
        """Return current buffer size."""
        return len(self.storage)

    @endpoint
    async def get_stats(self) -> dict:
        """Return buffer statistics."""
        return {
            "current_size": len(self.storage),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
        }
