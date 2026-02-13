"""
Property-based tests for reward computation.

Property 2: Reward Determinism
**Validates: Requirements 5.4**
For any trajectory (prompt, completion, ground_truth) tuple, calling the reward
computation function multiple times SHALL produce identical scalar reward values.

Property 3: Reward Correctness
**Validates: Requirements 5.1, 5.2, 5.3**
For any completion containing a final numeric answer:
- If the extracted answer equals the ground truth, correctness_score SHALL be 1.0
- If the extracted answer does not equal the ground truth, correctness_score SHALL be 0.0
- The total reward SHALL be correctness_score + format_score where 0.0 ≤ format_score ≤ 0.2
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from src.environment.reward import RewardWorker


# Strategy for generating prompts (math problem text)
prompt_strategy = st.text(min_size=0, max_size=500)

# Strategy for generating completions (model output with potential answers)
completion_strategy = st.one_of(
    # Empty or whitespace completions
    st.just(""),
    st.text(alphabet=" \t\n", min_size=1, max_size=10),
    # Completions with GSM8K format
    st.builds(
        lambda reasoning, answer: f"{reasoning}\n#### {answer}",
        reasoning=st.text(min_size=0, max_size=200),
        answer=st.one_of(
            st.integers(min_value=-10000, max_value=10000).map(str),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False).map(lambda x: f"{x:.2f}"),
        )
    ),
    # Completions with "the answer is" format
    st.builds(
        lambda reasoning, answer: f"{reasoning} The answer is {answer}",
        reasoning=st.text(min_size=0, max_size=200),
        answer=st.integers(min_value=-10000, max_value=10000).map(str),
    ),
    # Completions with reasoning indicators
    st.builds(
        lambda step1, step2, answer: f"First, {step1}. Then, {step2}. The answer is {answer}",
        step1=st.text(min_size=1, max_size=50),
        step2=st.text(min_size=1, max_size=50),
        answer=st.integers(min_value=-10000, max_value=10000).map(str),
    ),
    # Random text completions
    st.text(min_size=0, max_size=500),
)

# Strategy for generating ground truth answers
ground_truth_strategy = st.one_of(
    st.integers(min_value=-10000, max_value=10000).map(str),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False).map(lambda x: f"{x:.2f}"),
    st.text(min_size=1, max_size=20),  # Invalid ground truths
)


class TestRewardDeterminism:
    """
    Property-based tests for Reward Determinism.
    
    **Validates: Requirements 5.4**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up RewardWorker for each test."""
        self.worker = RewardWorker()
    
    @given(
        prompt=prompt_strategy,
        completion=completion_strategy,
        ground_truth=ground_truth_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_compute_reward_determinism(self, prompt, completion, ground_truth):
        """
        Property 2: Reward Determinism
        
        For any trajectory (prompt, completion, ground_truth) tuple,
        calling compute_reward multiple times SHALL produce identical results.
        
        **Validates: Requirements 5.4**
        """
        # Call compute_reward multiple times with identical inputs
        reward1 = self.worker.compute_reward(prompt, completion, ground_truth)
        reward2 = self.worker.compute_reward(prompt, completion, ground_truth)
        reward3 = self.worker.compute_reward(prompt, completion, ground_truth)
        
        # All calls must return identical values
        assert reward1 == reward2, (
            f"Reward changed between calls: {reward1} != {reward2}\n"
            f"Input: prompt={prompt!r}, completion={completion!r}, ground_truth={ground_truth!r}"
        )
        assert reward2 == reward3, (
            f"Reward changed between calls: {reward2} != {reward3}\n"
            f"Input: prompt={prompt!r}, completion={completion!r}, ground_truth={ground_truth!r}"
        )
    
    @given(
        completion=completion_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_extract_answer_determinism(self, completion):
        """
        Property: extract_answer SHALL be deterministic.
        
        For any completion, calling extract_answer multiple times
        SHALL produce identical results.
        
        **Validates: Requirements 5.4**
        """
        answer1 = self.worker.extract_answer(completion)
        answer2 = self.worker.extract_answer(completion)
        
        assert answer1 == answer2, (
            f"extract_answer changed between calls: {answer1!r} != {answer2!r}\n"
            f"Input: completion={completion!r}"
        )
    
    @given(
        completion=completion_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_check_format_determinism(self, completion):
        """
        Property: check_format SHALL be deterministic.
        
        For any completion, calling check_format multiple times
        SHALL produce identical results.
        
        **Validates: Requirements 5.4**
        """
        score1 = self.worker.check_format(completion)
        score2 = self.worker.check_format(completion)
        
        assert score1 == score2, (
            f"check_format changed between calls: {score1} != {score2}\n"
            f"Input: completion={completion!r}"
        )


class TestRewardCorrectness:
    """
    Property-based tests for Reward Correctness.
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    Property 3: Reward Correctness
    - If extracted answer equals ground truth, correctness_score SHALL be 1.0
    - If extracted answer does not equal ground truth, correctness_score SHALL be 0.0
    - Total reward SHALL be in valid range [0.0, 1.2]
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up RewardWorker for each test."""
        self.worker = RewardWorker()
    
    @given(
        answer=st.integers(min_value=-10000, max_value=10000),
        reasoning=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_correctness_score_is_one_when_answer_matches(self, answer, reasoning):
        """
        Property 3a: When extracted answer equals ground truth, correctness_score is 1.0.
        
        **Validates: Requirements 5.1**
        """
        # Create completion with known answer in GSM8K format
        completion = f"{reasoning}\n#### {answer}"
        ground_truth = str(answer)
        
        reward = self.worker.compute_reward("", completion, ground_truth)
        format_score = self.worker.check_format(completion)
        correctness_score = reward - format_score
        
        assert correctness_score == 1.0, (
            f"Expected correctness_score=1.0 when answer matches, got {correctness_score}\n"
            f"completion={completion!r}, ground_truth={ground_truth!r}"
        )
    
    @given(
        answer=st.integers(min_value=-10000, max_value=10000),
        wrong_offset=st.integers(min_value=1, max_value=1000),
        reasoning=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_correctness_score_is_zero_when_answer_differs(self, answer, wrong_offset, reasoning):
        """
        Property 3b: When extracted answer differs from ground truth, correctness_score is 0.0.
        
        **Validates: Requirements 5.1**
        """
        # Create completion with answer that differs from ground truth
        completion = f"{reasoning}\n#### {answer}"
        wrong_answer = answer + wrong_offset  # Guaranteed to be different
        ground_truth = str(wrong_answer)
        
        reward = self.worker.compute_reward("", completion, ground_truth)
        format_score = self.worker.check_format(completion)
        correctness_score = reward - format_score
        
        assert correctness_score == 0.0, (
            f"Expected correctness_score=0.0 when answer differs, got {correctness_score}\n"
            f"completion={completion!r}, ground_truth={ground_truth!r}"
        )
    
    @given(
        prompt=prompt_strategy,
        completion=completion_strategy,
        ground_truth=ground_truth_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_total_reward_in_valid_range(self, prompt, completion, ground_truth):
        """
        Property 3c: Total reward SHALL be in valid range [0.0, 1.2].
        
        **Validates: Requirements 5.2, 5.3**
        """
        reward = self.worker.compute_reward(prompt, completion, ground_truth)
        
        assert 0.0 <= reward <= 1.2, (
            f"Reward {reward} outside valid range [0.0, 1.2]\n"
            f"prompt={prompt!r}, completion={completion!r}, ground_truth={ground_truth!r}"
        )
    
    @given(
        completion=completion_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_format_score_in_valid_range(self, completion):
        """
        Property 3d: Format score SHALL be in valid range [0.0, 0.2].
        
        **Validates: Requirements 5.2**
        """
        format_score = self.worker.check_format(completion)
        
        assert 0.0 <= format_score <= 0.2, (
            f"Format score {format_score} outside valid range [0.0, 0.2]\n"
            f"completion={completion!r}"
        )
    
    @given(
        prompt=prompt_strategy,
        completion=completion_strategy,
        ground_truth=ground_truth_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_correctness_score_is_binary(self, prompt, completion, ground_truth):
        """
        Property 3e: Correctness score SHALL be binary (0.0 or 1.0).
        
        **Validates: Requirements 5.1**
        """
        reward = self.worker.compute_reward(prompt, completion, ground_truth)
        format_score = self.worker.check_format(completion)
        correctness_score = reward - format_score
        
        assert correctness_score in (0.0, 1.0), (
            f"Correctness score {correctness_score} is not binary (0.0 or 1.0)\n"
            f"reward={reward}, format_score={format_score}\n"
            f"prompt={prompt!r}, completion={completion!r}, ground_truth={ground_truth!r}"
        )
    
    @given(
        answer=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        reasoning=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_correctness_with_float_answers(self, answer, reasoning):
        """
        Property 3f: Correctness works correctly with floating point answers.
        
        **Validates: Requirements 5.1**
        """
        # Format answer with 2 decimal places
        answer_str = f"{answer:.2f}"
        completion = f"{reasoning}\nThe answer is {answer_str}"
        
        reward = self.worker.compute_reward("", completion, answer_str)
        format_score = self.worker.check_format(completion)
        correctness_score = reward - format_score
        
        # When answer matches exactly, correctness should be 1.0
        assert correctness_score == 1.0, (
            f"Expected correctness_score=1.0 for matching float, got {correctness_score}\n"
            f"completion={completion!r}, ground_truth={answer_str!r}"
        )
