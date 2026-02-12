"""Unit tests for reward computation."""

import pytest
from src.environment.reward import RewardWorker


class TestRewardWorker:
    """Tests for RewardWorker class."""
    
    def setup_method(self):
        self.worker = RewardWorker()
    
    # extract_answer tests
    def test_extract_answer_gsm8k_format(self):
        """Test extraction from GSM8K #### format."""
        completion = "Step 1: 5 + 3 = 8\n#### 8"
        assert self.worker.extract_answer(completion) == "8"
    
    def test_extract_answer_the_answer_is(self):
        """Test extraction from 'the answer is X' format."""
        completion = "So the answer is 42"
        assert self.worker.extract_answer(completion) == "42"
    
    def test_extract_answer_with_commas(self):
        """Test extraction handles numbers with commas."""
        completion = "The total is 1,234\n#### 1,234"
        assert self.worker.extract_answer(completion) == "1234"
    
    def test_extract_answer_decimal(self):
        """Test extraction of decimal numbers."""
        completion = "The answer is 3.14"
        assert self.worker.extract_answer(completion) == "3.14"
    
    def test_extract_answer_negative(self):
        """Test extraction of negative numbers."""
        completion = "#### -5"
        assert self.worker.extract_answer(completion) == "-5"
    
    def test_extract_answer_empty(self):
        """Test extraction from empty completion."""
        assert self.worker.extract_answer("") is None
        assert self.worker.extract_answer("   ") is None
    
    # check_format tests
    def test_check_format_full_compliance(self):
        """Test format score with full compliance."""
        completion = "First, we add 5 + 3. Then we get 8. The answer is 8"
        score = self.worker.check_format(completion)
        assert score == 0.2
    
    def test_check_format_reasoning_only(self):
        """Test format score with reasoning but no answer marker."""
        completion = "First, we add 5 + 3. Then we get 8."
        score = self.worker.check_format(completion)
        assert score == 0.1
    
    def test_check_format_answer_only(self):
        """Test format score with answer marker but no reasoning."""
        completion = "8\n#### 8"
        score = self.worker.check_format(completion)
        assert score == 0.1
    
    def test_check_format_empty(self):
        """Test format score for empty completion."""
        assert self.worker.check_format("") == 0.0
    
    # compute_reward tests
    def test_compute_reward_correct_with_format(self):
        """Test reward for correct answer with good format."""
        completion = "First, 5 + 3 = 8. The answer is 8"
        reward = self.worker.compute_reward("", completion, "8")
        assert reward == 1.2  # 1.0 correctness + 0.2 format
    
    def test_compute_reward_correct_no_format(self):
        """Test reward for correct answer without format."""
        completion = "8"
        reward = self.worker.compute_reward("", completion, "8")
        assert reward == 1.0  # 1.0 correctness + 0.0 format
    
    def test_compute_reward_incorrect(self):
        """Test reward for incorrect answer."""
        completion = "First, 5 + 3 = 9. The answer is 9"
        reward = self.worker.compute_reward("", completion, "8")
        assert reward == 0.2  # 0.0 correctness + 0.2 format
    
    def test_compute_reward_no_answer(self):
        """Test reward when no answer can be extracted."""
        completion = "I don't know"
        reward = self.worker.compute_reward("", completion, "8")
        assert reward == 0.0
    
    def test_compute_reward_determinism(self):
        """Test that reward computation is deterministic."""
        completion = "First, 5 + 3 = 8. The answer is 8"
        reward1 = self.worker.compute_reward("", completion, "8")
        reward2 = self.worker.compute_reward("", completion, "8")
        assert reward1 == reward2
