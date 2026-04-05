"""Dataset loader for GSM8K-style math problems.

Shared across all three GRPO implementations (custom, veRL, Monarch).
Supports HuggingFace and local JSON loading with multiple prompt formats.
"""
from dataclasses import dataclass
from typing import List, Optional, Iterator
import json
from pathlib import Path


@dataclass
class MathProblem:
    """A math word problem from GSM8K-style dataset."""
    question: str
    answer: str
    solution: Optional[str] = None

    def to_prompt(self) -> str:
        """Format as basic prompt for the model."""
        return f"Solve the following math problem step by step:\n\n{self.question}\n\nSolution:"

    def to_chat_prompt(self) -> list:
        """Format as chat message (veRL/Monarch style, proven to reach 77% accuracy)."""
        return [{"role": "user", "content": self.question + ' Let\'s think step by step and output the final answer after "####".'}]


class MathProblemDataset:
    """Dataset for GSM8K-style math problems.

    Supports loading from HuggingFace datasets or local JSON files.
    """

    def __init__(self, problems: List[MathProblem]):
        self._problems = problems

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, idx: int) -> MathProblem:
        return self._problems[idx]

    def __iter__(self) -> Iterator[MathProblem]:
        return iter(self._problems)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "openai/gsm8k",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> "MathProblemDataset":
        """Load dataset from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            max_samples: Maximum number of samples to load
        """
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, "main", split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        problems = []
        for item in dataset:
            answer = cls._extract_gsm8k_answer(item["answer"])
            problems.append(MathProblem(
                question=item["question"],
                answer=answer,
                solution=item["answer"],
            ))

        return cls(problems)

    @classmethod
    def from_json(cls, path: str) -> "MathProblemDataset":
        """Load dataset from local JSON file.

        Expected format: List of objects with 'question', 'answer', and optional 'solution' fields.
        """
        file_path = Path(path)
        with open(file_path, "r") as f:
            data = json.load(f)

        problems = [
            MathProblem(
                question=item["question"],
                answer=str(item["answer"]),
                solution=item.get("solution"),
            )
            for item in data
        ]

        return cls(problems)

    @staticmethod
    def _extract_gsm8k_answer(solution: str) -> str:
        """Extract the final numeric answer from GSM8K solution format.

        GSM8K answers end with '#### <number>'
        """
        if "####" in solution:
            return solution.split("####")[-1].strip()
        return solution.strip()

    def get_batch(self, batch_size: int, offset: int = 0) -> List[MathProblem]:
        """Get a batch of problems starting from offset."""
        end = min(offset + batch_size, len(self._problems))
        return self._problems[offset:end]

    def get_prompts(self) -> List[str]:
        """Get all problems formatted as prompts."""
        return [p.to_prompt() for p in self._problems]

    def get_ground_truths(self) -> List[str]:
        """Get all ground truth answers."""
        return [p.answer for p in self._problems]
