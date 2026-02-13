"""
Property-based tests for GRPO advantage normalization.

Property 5: GRPO Advantage Normalization
**Validates: Requirements 2.6, 7.5**

For any group of G rollouts for a single prompt with rewards [r₁, r₂, ..., r_G]:
- The computed advantages SHALL have mean ≈ 0 (within floating point tolerance)
- The computed advantages SHALL have std ≈ 1 (within floating point tolerance)
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st, assume

from src.trainer.grpo import compute_grpo_advantages, compute_grpo_advantages_from_list


# Tolerance for floating point comparisons
EPSILON = 1e-5

# Strategy for generating reward values (reasonable range for RL rewards)
reward_strategy = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy for generating group sizes (typical GRPO group sizes)
group_size_strategy = st.integers(min_value=2, max_value=16)

# Strategy for generating number of groups (batch size)
num_groups_strategy = st.integers(min_value=1, max_value=8)


class TestGRPOAdvantageNormalization:
    """
    Property-based tests for GRPO Advantage Normalization.
    
    **Validates: Requirements 2.6, 7.5**
    
    Property 5: GRPO Advantage Normalization
    For any group of G rollouts with rewards [r₁, ..., r_G]:
    - advantages SHALL have mean ≈ 0
    - advantages SHALL have std ≈ 1
    """
    
    @given(
        group_size=group_size_strategy,
        num_groups=num_groups_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_advantages_mean_approximately_zero(self, group_size, num_groups, rewards_data):
        """
        Property 5a: Normalized advantages SHALL have mean ≈ 0 per group.
        
        **Validates: Requirements 2.6, 7.5**
        """
        # Generate rewards for all groups
        rewards_list = [
            rewards_data.draw(st.lists(reward_strategy, min_size=group_size, max_size=group_size))
            for _ in range(num_groups)
        ]
        
        # Flatten rewards into tensor
        flat_rewards = [r for group in rewards_list for r in group]
        rewards_tensor = torch.tensor(flat_rewards, dtype=torch.float32)
        
        # Compute advantages
        advantages = compute_grpo_advantages(rewards_tensor, group_size)
        
        # Reshape to check per-group statistics
        grouped_advantages = advantages.view(num_groups, group_size)
        
        for i in range(num_groups):
            group_advantages = grouped_advantages[i]
            group_rewards = rewards_tensor.view(num_groups, group_size)[i]
            
            # Check if this group has non-zero std (skip zero-std groups)
            if group_rewards.std() < EPSILON:
                # For zero-std groups, all advantages should be 0
                assert torch.allclose(group_advantages, torch.zeros_like(group_advantages), atol=EPSILON), (
                    f"Group {i} with zero std should have all-zero advantages, got {group_advantages}"
                )
            else:
                # For non-zero std groups, mean should be ≈ 0
                mean = group_advantages.mean().item()
                assert abs(mean) < EPSILON, (
                    f"Group {i} advantages mean {mean} not approximately 0\n"
                    f"Rewards: {group_rewards.tolist()}\n"
                    f"Advantages: {group_advantages.tolist()}"
                )
    
    @given(
        group_size=group_size_strategy,
        num_groups=num_groups_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_advantages_std_approximately_one(self, group_size, num_groups, rewards_data):
        """
        Property 5b: Normalized advantages SHALL have std ≈ 1 per group.
        
        **Validates: Requirements 2.6, 7.5**
        """
        # Generate rewards for all groups
        rewards_list = [
            rewards_data.draw(st.lists(reward_strategy, min_size=group_size, max_size=group_size))
            for _ in range(num_groups)
        ]
        
        # Flatten rewards into tensor
        flat_rewards = [r for group in rewards_list for r in group]
        rewards_tensor = torch.tensor(flat_rewards, dtype=torch.float32)
        
        # Compute advantages
        advantages = compute_grpo_advantages(rewards_tensor, group_size)
        
        # Reshape to check per-group statistics
        grouped_advantages = advantages.view(num_groups, group_size)
        
        for i in range(num_groups):
            group_advantages = grouped_advantages[i]
            group_rewards = rewards_tensor.view(num_groups, group_size)[i]
            
            # Check if this group has non-zero std
            if group_rewards.std() < EPSILON:
                # For zero-std groups, all advantages should be 0
                assert torch.allclose(group_advantages, torch.zeros_like(group_advantages), atol=EPSILON), (
                    f"Group {i} with zero std should have all-zero advantages"
                )
            else:
                # For non-zero std groups, std should be ≈ 1
                std = group_advantages.std().item()
                assert abs(std - 1.0) < EPSILON, (
                    f"Group {i} advantages std {std} not approximately 1\n"
                    f"Rewards: {group_rewards.tolist()}\n"
                    f"Advantages: {group_advantages.tolist()}"
                )
    
    @given(
        rewards=st.lists(reward_strategy, min_size=2, max_size=16),
    )
    @settings(max_examples=100, deadline=None)
    def test_single_group_normalization(self, rewards):
        """
        Property 5c: Single group normalization produces mean ≈ 0 and std ≈ 1.
        
        **Validates: Requirements 2.6, 7.5**
        """
        # Use list-based function for single group
        advantages_list = compute_grpo_advantages_from_list([rewards])
        advantages = advantages_list[0]
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        
        if rewards_tensor.std() < EPSILON:
            # Zero-std case: all advantages should be 0
            assert all(a == 0.0 for a in advantages), (
                f"Zero-std rewards should produce all-zero advantages, got {advantages}"
            )
        else:
            # Non-zero std case: check normalization
            mean = advantages_tensor.mean().item()
            std = advantages_tensor.std().item()
            
            assert abs(mean) < EPSILON, (
                f"Advantages mean {mean} not approximately 0\n"
                f"Rewards: {rewards}\n"
                f"Advantages: {advantages}"
            )
            assert abs(std - 1.0) < EPSILON, (
                f"Advantages std {std} not approximately 1\n"
                f"Rewards: {rewards}\n"
                f"Advantages: {advantages}"
            )
    
    @given(
        base_reward=reward_strategy,
        group_size=group_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_identical_rewards_produce_zero_advantages(self, base_reward, group_size):
        """
        Property 5d: When all rewards in a group are identical, advantages SHALL be 0.
        
        **Validates: Requirements 2.6, 7.5**
        """
        # Create group with identical rewards
        rewards = [base_reward] * group_size
        
        advantages_list = compute_grpo_advantages_from_list([rewards])
        advantages = advantages_list[0]
        
        assert all(a == 0.0 for a in advantages), (
            f"Identical rewards {rewards} should produce all-zero advantages, got {advantages}"
        )
    
    @given(
        group_size=group_size_strategy,
        num_groups=num_groups_strategy,
        rewards_data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_tensor_and_list_functions_equivalent(self, group_size, num_groups, rewards_data):
        """
        Property 5e: Tensor and list-based functions SHALL produce equivalent results.
        
        **Validates: Requirements 2.6, 7.5**
        """
        # Generate rewards for all groups
        rewards_list = [
            rewards_data.draw(st.lists(reward_strategy, min_size=group_size, max_size=group_size))
            for _ in range(num_groups)
        ]
        
        # Compute using list function
        advantages_from_list = compute_grpo_advantages_from_list(rewards_list)
        
        # Compute using tensor function
        flat_rewards = [r for group in rewards_list for r in group]
        rewards_tensor = torch.tensor(flat_rewards, dtype=torch.float32)
        advantages_tensor = compute_grpo_advantages(rewards_tensor, group_size)
        
        # Compare results
        for i, (list_advs, tensor_advs) in enumerate(zip(
            advantages_from_list,
            advantages_tensor.view(num_groups, group_size).tolist()
        )):
            for j, (la, ta) in enumerate(zip(list_advs, tensor_advs)):
                assert abs(la - ta) < EPSILON, (
                    f"Mismatch at group {i}, position {j}: list={la}, tensor={ta}\n"
                    f"Rewards: {rewards_list[i]}"
                )
