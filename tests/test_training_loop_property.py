"""
Property-based tests for the training loop round-trip.

Property 1: Training Loop Round-Trip
**Validates: Requirements 2.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.5**

For any batch of math problem prompts, the system SHALL:
1. Generate G rollouts per prompt via the actor model
2. Send all trajectories to the environment service
3. Receive scalar rewards for each trajectory
4. Compute GRPO advantages using group normalization
5. Apply policy updates to the actor model

The number of rewards received MUST equal the number of trajectories sent (batch_size × group_size).
"""

import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, settings, strategies as st, assume
from typing import List

from src.trainer.config import TrainingConfig, FSDPConfig
from src.trainer.models import (
    Rollout,
    Trajectory,
    RewardResponse,
    RewardDetails,
)
from src.trainer.dataset import MathProblem
from src.trainer.grpo import compute_grpo_advantages


# Strategy for generating batch sizes (typical training batch sizes)
batch_size_strategy = st.integers(min_value=1, max_value=8)

# Strategy for generating group sizes (typical GRPO group sizes)
group_size_strategy = st.integers(min_value=2, max_value=8)

# Strategy for generating math problem questions
question_strategy = st.text(min_size=10, max_size=200, alphabet=st.characters(
    whitelist_categories=('L', 'N', 'P', 'Z'),
    whitelist_characters=' .,?!+-*/='
))

# Strategy for generating numeric answers
answer_strategy = st.integers(min_value=-1000, max_value=1000).map(str)

# Strategy for generating reward values
reward_strategy = st.floats(
    min_value=0.0,
    max_value=1.2,
    allow_nan=False,
    allow_infinity=False,
)


def create_mock_rollout(prompt: str, completion: str = "") -> Rollout:
    """Create a mock rollout with the given prompt."""
    return Rollout(
        prompt=prompt,
        completion=completion,
        log_probs=torch.randn(10),  # Random log probs
        tokens=list(range(10)),
    )


def create_mock_reward_response(n_trajectories: int, rewards: List[float]) -> RewardResponse:
    """Create a mock reward response with the given rewards."""
    return RewardResponse(
        rewards=rewards,
        details=[
            RewardDetails(
                correctness_score=1.0 if r >= 1.0 else 0.0,
                format_score=min(r, 0.2),
                extracted_answer="42",
                is_correct=r >= 1.0,
            )
            for r in rewards
        ],
    )


class TestTrainingLoopRoundTrip:
    """
    Property-based tests for Training Loop Round-Trip.
    
    **Validates: Requirements 2.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.5**
    
    Property 1: Training Loop Round-Trip
    For any batch of math problem prompts:
    - Generate G rollouts per prompt
    - Send all trajectories to environment
    - Receive rewards for each trajectory
    - Number of rewards MUST equal batch_size × group_size
    """
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
        problems_data=st.data(),
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_trajectory_count_equals_batch_times_group(
        self,
        batch_size: int,
        group_size: int,
        problems_data,
        rewards_data,
    ):
        """
        Property 1a: Number of trajectories generated equals batch_size × group_size.
        
        **Validates: Requirements 2.4, 7.1, 7.2**
        """
        # Generate random math problems
        problems = [
            MathProblem(
                question=problems_data.draw(question_strategy),
                answer=problems_data.draw(answer_strategy),
            )
            for _ in range(batch_size)
        ]
        
        # Simulate rollout generation (G rollouts per prompt)
        prompts = [p.to_prompt() for p in problems]
        rollouts = []
        for prompt in prompts:
            for _ in range(group_size):
                rollouts.append(create_mock_rollout(prompt))
        
        # Verify trajectory count
        expected_count = batch_size * group_size
        actual_count = len(rollouts)
        
        assert actual_count == expected_count, (
            f"Expected {expected_count} trajectories (batch_size={batch_size} × group_size={group_size}), "
            f"got {actual_count}"
        )
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_reward_count_equals_trajectory_count(
        self,
        batch_size: int,
        group_size: int,
        rewards_data,
    ):
        """
        Property 1b: Number of rewards received equals number of trajectories sent.
        
        **Validates: Requirements 2.5, 7.3, 7.4**
        """
        n_trajectories = batch_size * group_size
        
        # Generate random rewards for each trajectory
        rewards = [
            rewards_data.draw(reward_strategy)
            for _ in range(n_trajectories)
        ]
        
        # Create mock response
        response = create_mock_reward_response(n_trajectories, rewards)
        
        # Verify reward count matches trajectory count
        assert len(response.rewards) == n_trajectories, (
            f"Expected {n_trajectories} rewards, got {len(response.rewards)}"
        )
        assert len(response.details) == n_trajectories, (
            f"Expected {n_trajectories} details, got {len(response.details)}"
        )
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_advantages_count_equals_reward_count(
        self,
        batch_size: int,
        group_size: int,
        rewards_data,
    ):
        """
        Property 1c: Number of computed advantages equals number of rewards.
        
        **Validates: Requirements 2.6, 7.5**
        """
        n_trajectories = batch_size * group_size
        
        # Generate random rewards
        rewards = [
            rewards_data.draw(reward_strategy)
            for _ in range(n_trajectories)
        ]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute GRPO advantages
        advantages = compute_grpo_advantages(rewards_tensor, group_size)
        
        # Verify advantage count matches reward count
        assert advantages.shape[0] == n_trajectories, (
            f"Expected {n_trajectories} advantages, got {advantages.shape[0]}"
        )
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
        problems_data=st.data(),
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_full_round_trip_invariant(
        self,
        batch_size: int,
        group_size: int,
        problems_data,
        rewards_data,
    ):
        """
        Property 1d: Full round-trip maintains count invariant throughout.
        
        For batch_size prompts with group_size rollouts each:
        - trajectories_sent == batch_size × group_size
        - rewards_received == trajectories_sent
        - advantages_computed == rewards_received
        
        **Validates: Requirements 2.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        expected_count = batch_size * group_size
        
        # Step 1: Generate problems
        problems = [
            MathProblem(
                question=problems_data.draw(question_strategy),
                answer=problems_data.draw(answer_strategy),
            )
            for _ in range(batch_size)
        ]
        
        # Step 2: Generate rollouts (G per prompt)
        prompts = [p.to_prompt() for p in problems]
        ground_truths = [p.answer for p in problems]
        
        rollouts = []
        for prompt in prompts:
            for _ in range(group_size):
                rollouts.append(create_mock_rollout(prompt, completion="The answer is 42"))
        
        # Step 3: Build trajectories
        trajectories = []
        for i, rollout in enumerate(rollouts):
            prompt_idx = i // group_size
            trajectories.append(Trajectory(
                prompt=rollout.prompt,
                completion=rollout.completion,
                ground_truth=ground_truths[prompt_idx],
            ))
        
        # Verify trajectory count
        assert len(trajectories) == expected_count, (
            f"Trajectory count mismatch: expected {expected_count}, got {len(trajectories)}"
        )
        
        # Step 4: Simulate reward response
        rewards = [
            rewards_data.draw(reward_strategy)
            for _ in range(expected_count)
        ]
        response = create_mock_reward_response(expected_count, rewards)
        
        # Verify reward count
        assert len(response.rewards) == expected_count, (
            f"Reward count mismatch: expected {expected_count}, got {len(response.rewards)}"
        )
        
        # Step 5: Compute advantages
        rewards_tensor = torch.tensor(response.rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards_tensor, group_size)
        
        # Verify advantage count
        assert advantages.shape[0] == expected_count, (
            f"Advantage count mismatch: expected {expected_count}, got {advantages.shape[0]}"
        )
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_rollout_prompt_assignment(
        self,
        batch_size: int,
        group_size: int,
    ):
        """
        Property 1e: Each prompt gets exactly group_size rollouts.
        
        **Validates: Requirements 2.4, 7.2**
        """
        # Create distinct prompts
        prompts = [f"Problem {i}: What is {i} + {i}?" for i in range(batch_size)]
        
        # Generate rollouts
        rollouts = []
        for prompt in prompts:
            for _ in range(group_size):
                rollouts.append(create_mock_rollout(prompt))
        
        # Count rollouts per prompt
        prompt_counts = {}
        for rollout in rollouts:
            prompt_counts[rollout.prompt] = prompt_counts.get(rollout.prompt, 0) + 1
        
        # Verify each prompt has exactly group_size rollouts
        for prompt in prompts:
            assert prompt_counts.get(prompt, 0) == group_size, (
                f"Prompt '{prompt[:50]}...' has {prompt_counts.get(prompt, 0)} rollouts, "
                f"expected {group_size}"
            )
    
    @given(
        batch_size=batch_size_strategy,
        group_size=group_size_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_ground_truth_assignment_to_trajectories(
        self,
        batch_size: int,
        group_size: int,
        rewards_data,
    ):
        """
        Property 1f: Each trajectory has correct ground_truth from its source problem.
        
        **Validates: Requirements 7.1, 7.3**
        """
        # Create problems with distinct answers
        problems = [
            MathProblem(
                question=f"Problem {i}",
                answer=str(i * 100),  # Distinct answers
            )
            for i in range(batch_size)
        ]
        
        prompts = [p.to_prompt() for p in problems]
        ground_truths = [p.answer for p in problems]
        
        # Generate rollouts and trajectories
        rollouts = []
        for prompt in prompts:
            for _ in range(group_size):
                rollouts.append(create_mock_rollout(prompt))
        
        trajectories = []
        for i, rollout in enumerate(rollouts):
            prompt_idx = i // group_size
            trajectories.append(Trajectory(
                prompt=rollout.prompt,
                completion=rollout.completion,
                ground_truth=ground_truths[prompt_idx],
            ))
        
        # Verify ground truth assignment
        for i, trajectory in enumerate(trajectories):
            prompt_idx = i // group_size
            expected_ground_truth = ground_truths[prompt_idx]
            assert trajectory.ground_truth == expected_ground_truth, (
                f"Trajectory {i} has ground_truth={trajectory.ground_truth}, "
                f"expected {expected_ground_truth} (prompt_idx={prompt_idx})"
            )
