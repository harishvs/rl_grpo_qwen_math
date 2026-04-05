"""DatasetActor -- serves GSM8K batches from a CPU process.

Tracks position in the dataset, handles epoch cycling with reshuffling.
"""
import random
from typing import Optional

from monarch.actor import Actor, endpoint

from src.shared.dataset import MathProblemDataset


class DatasetActor(Actor):
    """Serves batches of GSM8K math problems.

    Loads the dataset once at init, then serves batches sequentially.
    Reshuffles at the start of each new epoch.
    """

    def __init__(
        self,
        dataset_name: str = "openai/gsm8k",
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = MathProblemDataset.from_huggingface(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
        )
        self.indices = list(range(len(self.dataset)))
        self.offset = 0
        self.epoch = 0
        self.seed = seed
        random.seed(seed)
        random.shuffle(self.indices)

    @endpoint
    async def next_batch(self, batch_size: int) -> dict:
        """Get the next batch of problems.

        Args:
            batch_size: Number of problems to return

        Returns:
            Dict with 'prompts' (chat format), 'ground_truths', 'epoch', 'offset'
        """
        if self.offset >= len(self.indices):
            self.epoch += 1
            self.offset = 0
            random.shuffle(self.indices)

        end = min(self.offset + batch_size, len(self.indices))
        batch_indices = self.indices[self.offset:end]
        self.offset = end

        problems = [self.dataset[i] for i in batch_indices]
        return {
            "prompts": [p.to_chat_prompt() for p in problems],
            "ground_truths": [p.answer for p in problems],
            "epoch": self.epoch,
            "offset": self.offset,
        }

    @endpoint
    async def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            "total_problems": len(self.dataset),
            "current_epoch": self.epoch,
            "current_offset": self.offset,
        }
