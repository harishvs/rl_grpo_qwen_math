"""GRPO (Group Relative Policy Optimization) advantage computation."""

import torch
from typing import List


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute GRPO advantages using group normalization.
    
    For each group of G rollouts from the same prompt, compute:
        A_i = (r_i - mean(R)) / std(R)
    
    Args:
        rewards: Tensor of shape (batch_size * group_size,) containing rewards
        group_size: Number of rollouts per prompt (G)
        eps: Small constant for numerical stability when std ≈ 0
        
    Returns:
        Tensor of normalized advantages with same shape as rewards
    """
    num_groups = rewards.shape[0] // group_size
    
    # Reshape to (num_groups, group_size) for per-group normalization
    grouped_rewards = rewards.view(num_groups, group_size)
    
    # Compute per-group statistics
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)
    
    # Handle edge case where std = 0 (all rewards identical in group)
    # In this case, all advantages should be 0
    group_stds = torch.where(
        group_stds < eps,
        torch.ones_like(group_stds),
        group_stds,
    )
    
    # Normalize within each group
    advantages = (grouped_rewards - group_means) / group_stds
    
    # For groups with zero std, set advantages to 0
    zero_std_mask = grouped_rewards.std(dim=1, keepdim=True) < eps
    advantages = torch.where(zero_std_mask, torch.zeros_like(advantages), advantages)
    
    # Flatten back to original shape
    return advantages.view(-1)


def compute_grpo_advantages_from_list(
    reward_groups: List[List[float]],
    eps: float = 1e-8,
) -> List[List[float]]:
    """Compute GRPO advantages from a list of reward groups.
    
    Convenience function for when rewards are organized as nested lists.
    
    Args:
        reward_groups: List of reward lists, one per prompt
        eps: Small constant for numerical stability
        
    Returns:
        List of advantage lists with same structure as input
    """
    advantages = []
    
    for rewards in reward_groups:
        if len(rewards) == 0:
            advantages.append([])
            continue
            
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        
        # Check if all rewards are effectively identical using range check
        # This handles floating point precision issues better than std alone
        reward_range = rewards_tensor.max() - rewards_tensor.min()
        
        if std < eps or reward_range < eps:
            # All rewards identical, advantages are 0
            advantages.append([0.0] * len(rewards))
        else:
            normalized = ((rewards_tensor - mean) / std).tolist()
            advantages.append(normalized)
    
    return advantages
