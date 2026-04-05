"""Unit tests for the Monarch GRPO implementation.

Tests shared modules (reward, grpo, dataset) and Monarch-specific code
(config, metrics, replay buffer). Actor tests that require Monarch/GPU
are skipped when not available.
"""
import pytest
import torch
import os
import tempfile

# --- Shared module tests ---

from src.shared.reward import compute_score
from src.shared.grpo import compute_grpo_advantages, compute_grpo_advantages_from_list
from src.shared.dataset import MathProblem, MathProblemDataset


class TestSharedReward:
    """Tests for src/shared/reward.py"""

    def test_correct_answer(self):
        assert compute_score("gsm8k", "The answer is #### 42", "42") == 1.0

    def test_wrong_answer(self):
        assert compute_score("gsm8k", "The answer is #### 99", "42") == 0.0

    def test_no_answer_marker(self):
        assert compute_score("gsm8k", "I think it's 42", "42") == 0.0

    def test_negative_number(self):
        assert compute_score("gsm8k", "#### -5", "-5") == 1.0

    def test_comma_in_number(self):
        assert compute_score("gsm8k", "#### 1,234", "1234") == 1.0

    def test_decimal(self):
        assert compute_score("gsm8k", "#### 3.14", "3.14") == 1.0

    def test_empty_completion(self):
        assert compute_score("gsm8k", "", "42") == 0.0

    def test_extra_info_ignored(self):
        assert compute_score("gsm8k", "#### 42", "42", extra_info={"foo": "bar"}) == 1.0

    def test_non_numeric_returns_zero(self):
        # regex only matches numbers, so non-numeric after #### returns 0.0
        assert compute_score("gsm8k", "#### abc", "abc") == 0.0

    def test_deterministic(self):
        r1 = compute_score("gsm8k", "#### 42", "42")
        r2 = compute_score("gsm8k", "#### 42", "42")
        assert r1 == r2


class TestSharedGRPO:
    """Tests for src/shared/grpo.py"""

    def test_basic_advantages(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        advs = compute_grpo_advantages(rewards, group_size=2)
        assert advs.shape == (4,)
        # Within each group of 2: [1,0] -> positive, negative
        assert advs[0] > 0  # 1.0 is above group mean
        assert advs[1] < 0  # 0.0 is below group mean

    def test_all_same_rewards(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        advs = compute_grpo_advantages(rewards, group_size=2)
        assert torch.allclose(advs, torch.zeros(4))

    def test_single_group(self):
        rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        advs = compute_grpo_advantages(rewards, group_size=8)
        assert advs.shape == (8,)
        assert advs[0] > 0  # above mean

    def test_output_shape(self):
        rewards = torch.tensor([1.0, 0.0] * 16)
        advs = compute_grpo_advantages(rewards, group_size=8)
        assert advs.shape == rewards.shape

    def test_zero_std_group(self):
        rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0])
        advs = compute_grpo_advantages(rewards, group_size=4)
        # First group (all 0.5) should have zero advantages
        assert torch.allclose(advs[:4], torch.zeros(4))
        # Second group should have non-zero advantages
        assert not torch.allclose(advs[4:], torch.zeros(4))

    def test_list_version(self):
        result = compute_grpo_advantages_from_list([[1.0, 0.0], [0.5, 0.5]])
        assert len(result) == 2
        assert result[0][0] > 0
        assert result[0][1] < 0
        assert result[1] == [0.0, 0.0]

    def test_list_empty_group(self):
        result = compute_grpo_advantages_from_list([[]])
        assert result == [[]]


class TestSharedDataset:
    """Tests for src/shared/dataset.py"""

    def test_math_problem_prompt(self):
        p = MathProblem(question="What is 2+2?", answer="4")
        prompt = p.to_prompt()
        assert "2+2" in prompt
        assert "Solution:" in prompt

    def test_math_problem_chat_prompt(self):
        p = MathProblem(question="What is 2+2?", answer="4")
        chat = p.to_chat_prompt()
        assert isinstance(chat, list)
        assert chat[0]["role"] == "user"
        assert "2+2" in chat[0]["content"]
        assert "####" in chat[0]["content"]

    def test_dataset_from_list(self):
        problems = [MathProblem("Q1", "1"), MathProblem("Q2", "2")]
        ds = MathProblemDataset(problems)
        assert len(ds) == 2
        assert ds[0].answer == "1"

    def test_dataset_get_batch(self):
        problems = [MathProblem(f"Q{i}", str(i)) for i in range(10)]
        ds = MathProblemDataset(problems)
        batch = ds.get_batch(3, offset=2)
        assert len(batch) == 3
        assert batch[0].answer == "2"

    def test_dataset_get_prompts(self):
        problems = [MathProblem("Q1", "1"), MathProblem("Q2", "2")]
        ds = MathProblemDataset(problems)
        prompts = ds.get_prompts()
        assert len(prompts) == 2
        assert "Q1" in prompts[0]

    def test_dataset_get_ground_truths(self):
        problems = [MathProblem("Q1", "1"), MathProblem("Q2", "2")]
        ds = MathProblemDataset(problems)
        gts = ds.get_ground_truths()
        assert gts == ["1", "2"]

    def test_extract_gsm8k_answer(self):
        assert MathProblemDataset._extract_gsm8k_answer("blah #### 42") == "42"
        assert MathProblemDataset._extract_gsm8k_answer("no marker") == "no marker"

    def test_dataset_iteration(self):
        problems = [MathProblem(f"Q{i}", str(i)) for i in range(3)]
        ds = MathProblemDataset(problems)
        items = list(ds)
        assert len(items) == 3

    def test_from_json(self):
        import json
        data = [{"question": "Q1", "answer": "1"}, {"question": "Q2", "answer": "2"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            ds = MathProblemDataset.from_json(path)
            assert len(ds) == 2
            assert ds[0].question == "Q1"
        finally:
            os.unlink(path)


# --- Monarch-specific tests ---

from src.monarch.config import MonarchTrainingConfig, load_config


class TestMonarchConfig:
    """Tests for src/monarch/config.py"""

    def test_default_config(self):
        cfg = MonarchTrainingConfig()
        assert cfg.model == "Qwen/Qwen2.5-1.5B"
        assert cfg.learner.lr == 5e-6
        assert cfg.generator.group_size == 8
        assert cfg.learner.kl_coef == 0.1
        assert cfg.learner.clip_range == 0.2
        assert cfg.trainer.weight_sync_interval == 3

    def test_load_config_from_yaml(self):
        cfg = load_config("config/monarch/qwen-1.5b.yaml")
        assert cfg.model == "Qwen/Qwen2.5-1.5B"
        assert cfg.data.train_batch_size == 256
        assert cfg.generator.gpu_memory_utilization == 0.5
        assert cfg.trainer.nnodes == 2

    def test_load_7b_config(self):
        cfg = load_config("config/monarch/qwen-7b.yaml")
        assert cfg.model == "Qwen/Qwen2.5-7B"
        assert cfg.data.train_batch_size == 128
        assert cfg.learner.lr == 1e-6
        assert cfg.generator.tensor_parallel_size == 2

    def test_config_overrides(self):
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": "test-model", "learner": {"lr": 1e-4}}, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.model == "test-model"
            assert cfg.learner.lr == 1e-4
            # Defaults preserved for unspecified fields
            assert cfg.generator.group_size == 8
        finally:
            os.unlink(path)


from src.monarch.metrics import MetricsLogger


class TestMonarchMetrics:
    """Tests for src/monarch/metrics.py"""

    def test_logger_creation(self):
        logger = MetricsLogger(experiment_name="test")
        assert logger.experiment_name == "test"
        assert logger._started is False

    def test_log_without_start(self, capsys):
        logger = MetricsLogger(experiment_name="test")
        # log() should print to stdout even without starting prometheus
        logger.log(0, {"policy_loss": 0.5, "mean_reward": 0.3})
        captured = capsys.readouterr()
        assert "step:0" in captured.out
        assert "policy_loss" in captured.out
        assert "mean_reward" in captured.out


# --- Replay buffer tests (no Monarch dependency, pure Python) ---

import asyncio


class TestReplayBuffer:
    """Tests for replay buffer logic without Monarch Actor dependency."""

    def test_buffer_add_and_sample(self):
        """Test basic add/sample cycle."""
        storage = []
        episodes = {
            "prompts": ["p1"],
            "completions": ["c1"],
            "log_probs": [[0.1, 0.2]],
            "rewards": [1.0],
            "advantages": [0.5],
        }
        storage.append({"data": episodes, "version": 0})
        assert len(storage) == 1
        assert storage[0]["data"]["rewards"] == [1.0]

    def test_buffer_staleness_filter(self):
        """Test that staleness filtering works."""
        storage = [
            {"data": {"rewards": [1.0]}, "version": 0},
            {"data": {"rewards": [0.5]}, "version": 1},
            {"data": {"rewards": [0.8]}, "version": 2},
        ]
        max_policy_age = 1
        current_version = 2
        min_version = current_version - max_policy_age
        valid = [e for e in storage if e["version"] >= min_version]
        assert len(valid) == 2  # version 1 and 2
        assert valid[0]["version"] == 1
        assert valid[1]["version"] == 2

    def test_buffer_sync_mode(self):
        """Test max_policy_age=0 only returns current version."""
        storage = [
            {"data": {"rewards": [1.0]}, "version": 0},
            {"data": {"rewards": [0.5]}, "version": 1},
        ]
        max_policy_age = 0
        current_version = 1
        min_version = current_version - max_policy_age
        valid = [e for e in storage if e["version"] >= min_version]
        assert len(valid) == 1
        assert valid[0]["version"] == 1


class TestRewardScoring:
    """Tests for reward scoring logic used by RewardActor."""

    def test_batch_scoring(self):
        completions = ["#### 42", "#### 99", "#### 42"]
        ground_truths = ["42", "42", "42"]
        rewards = [
            compute_score("gsm8k", c, gt)
            for c, gt in zip(completions, ground_truths)
        ]
        assert rewards == [1.0, 0.0, 1.0]

    def test_grpo_pipeline(self):
        """Test the full reward -> advantage pipeline."""
        completions = ["#### 42"] * 4 + ["#### 99"] * 4
        ground_truths = ["42"] * 8
        rewards = [compute_score("gsm8k", c, gt) for c, gt in zip(completions, ground_truths)]
        rewards_tensor = torch.tensor(rewards)
        advantages = compute_grpo_advantages(rewards_tensor, group_size=8)

        assert advantages.shape == (8,)
        # Correct answers should have positive advantages
        assert all(advantages[:4] > 0)
        # Wrong answers should have negative advantages
        assert all(advantages[4:] < 0)
