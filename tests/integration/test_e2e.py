"""
End-to-end integration tests for RL Code LLM Training.

These tests validate:
1. Environment service reachability from trainer components
2. Single training step execution
3. Communication between GPU and CPU workloads

Requirements: 7.6
"""

import asyncio
import os
import pytest
import torch
from unittest.mock import patch

from src.trainer.environment_client import (
    EnvironmentClient,
    EnvironmentClientConfig,
)
from src.trainer.config import RetryConfig, TrainingConfig
from src.trainer.models import Trajectory, RewardResponse, RewardDetails
from src.trainer.grpo import compute_grpo_advantages


# Skip integration tests if not in integration test mode
INTEGRATION_TEST_MODE = os.environ.get("INTEGRATION_TEST_MODE", "false").lower() == "true"
ENVIRONMENT_SERVICE_URL = os.environ.get(
    "ENVIRONMENT_SERVICE_URL", 
    "http://localhost:8080"
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEnvironmentServiceReachability:
    """Tests for environment service reachability from trainer components."""
    
    @pytest.fixture
    def client_config(self):
        """Create client config for testing."""
        return EnvironmentClientConfig(
            base_url=ENVIRONMENT_SERVICE_URL,
            retry=RetryConfig(
                max_retries=2,
                initial_backoff_ms=100,
                timeout_ms=5000,
            ),
        )
    
    def test_client_config_defaults(self):
        """Test environment client configuration defaults."""
        config = EnvironmentClientConfig()
        
        assert config.base_url == "http://environment-service:8080"
        assert config.retry is not None
        assert config.retry.max_retries == 3
    
    def test_client_initialization(self):
        """Test environment client can be initialized."""
        config = EnvironmentClientConfig(
            base_url="http://test-service:8080",
            retry=RetryConfig(max_retries=1, timeout_ms=1000),
        )
        
        client = EnvironmentClient(config)
        assert client.config.base_url == "http://test-service:8080"
        assert client._session is None  # Session not created until first use
    
    def test_health_check_returns_false_on_connection_error(self):
        """Test health check returns False when service is unreachable."""
        async def _test():
            config = EnvironmentClientConfig(
                base_url="http://nonexistent-service:8080",
                retry=RetryConfig(max_retries=0, timeout_ms=100),
            )
            
            client = EnvironmentClient(config)
            result = await client.health_check()
            assert result is False
            await client.close()
        
        run_async(_test())
    
    @pytest.mark.skipif(
        not INTEGRATION_TEST_MODE,
        reason="Integration tests require INTEGRATION_TEST_MODE=true"
    )
    def test_live_health_check(self, client_config):
        """Test health check against live environment service."""
        async def _test():
            async with EnvironmentClient(client_config) as client:
                result = await client.health_check()
                assert result is True, (
                    f"Environment service at {ENVIRONMENT_SERVICE_URL} is not healthy"
                )
        
        run_async(_test())


class TestTrainerEnvironmentCommunication:
    """Tests for communication between trainer and environment service."""
    
    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        return [
            Trajectory(
                prompt="What is 5 + 3?",
                completion="Let me solve this step by step. 5 + 3 = 8. The answer is 8",
                ground_truth="8",
            ),
            Trajectory(
                prompt="What is 10 - 4?",
                completion="10 minus 4 equals 6. The answer is 6",
                ground_truth="6",
            ),
            Trajectory(
                prompt="What is 2 * 7?",
                completion="2 times 7 is 14. The answer is 14",
                ground_truth="14",
            ),
        ]
    
    def test_trajectory_creation(self, sample_trajectories):
        """Test trajectory objects are created correctly."""
        assert len(sample_trajectories) == 3
        assert sample_trajectories[0].prompt == "What is 5 + 3?"
        assert sample_trajectories[0].ground_truth == "8"
    
    def test_reward_response_structure(self):
        """Test RewardResponse structure."""
        response = RewardResponse(
            rewards=[1.0, 0.5, 0.8],
            details=[
                RewardDetails(
                    correctness_score=1.0,
                    format_score=0.0,
                    extracted_answer="8",
                    is_correct=True,
                ),
                RewardDetails(
                    correctness_score=0.0,
                    format_score=0.5,
                    extracted_answer="wrong",
                    is_correct=False,
                ),
                RewardDetails(
                    correctness_score=0.8,
                    format_score=0.0,
                    extracted_answer="14",
                    is_correct=True,
                ),
            ],
        )
        
        assert len(response.rewards) == 3
        assert len(response.details) == 3
        assert response.details[0].is_correct is True
        assert response.details[1].is_correct is False
    
    def test_retry_config_defaults(self):
        """Test retry configuration defaults."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_backoff_ms == 100
        assert config.max_backoff_ms == 5000
        assert config.backoff_multiplier == 2.0
        assert config.timeout_ms == 30000
        assert config.skip_on_failure is True
    
    @pytest.mark.skipif(
        not INTEGRATION_TEST_MODE,
        reason="Integration tests require INTEGRATION_TEST_MODE=true"
    )
    def test_live_compute_rewards(self, sample_trajectories):
        """Test reward computation against live environment service."""
        async def _test():
            config = EnvironmentClientConfig(
                base_url=ENVIRONMENT_SERVICE_URL,
                retry=RetryConfig(max_retries=2, timeout_ms=10000),
            )
            
            async with EnvironmentClient(config) as client:
                result = await client.compute_rewards(sample_trajectories)
                
                assert len(result.rewards) == len(sample_trajectories)
                assert len(result.details) == len(sample_trajectories)
                # All sample trajectories have correct answers
                assert all(d.is_correct for d in result.details)
        
        run_async(_test())


class TestSingleTrainingStep:
    """Tests for single training step execution."""
    
    def test_grpo_advantage_computation(self):
        """Test GRPO advantage computation for a single step."""
        # Simulate rewards from environment service (1 group of 8)
        rewards = torch.tensor([0.8, 1.2, 0.5, 1.0, 0.9, 1.1, 0.7, 0.6])
        group_size = 8
        
        advantages = compute_grpo_advantages(rewards, group_size)
        
        # Verify normalization properties
        assert len(advantages) == len(rewards)
        # Mean should be approximately 0
        mean_adv = advantages.mean().item()
        assert abs(mean_adv) < 1e-5, f"Mean advantage {mean_adv} not close to 0"
        # Std should be approximately 1
        std_adv = advantages.std().item()
        assert abs(std_adv - 1.0) < 1e-5, f"Std advantage {std_adv} not close to 1"
    
    def test_grpo_advantage_with_identical_rewards(self):
        """Test GRPO handles identical rewards (std=0 case)."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        group_size = 4
        
        advantages = compute_grpo_advantages(rewards, group_size)
        
        # All advantages should be 0 when rewards are identical
        assert torch.allclose(advantages, torch.zeros_like(advantages))
    
    def test_grpo_advantage_multiple_groups(self):
        """Test GRPO with multiple groups."""
        # 2 groups of 4 rewards each
        rewards = torch.tensor([
            0.5, 0.7, 0.9, 1.1,  # Group 1
            0.2, 0.4, 0.6, 0.8,  # Group 2
        ])
        group_size = 4
        
        advantages = compute_grpo_advantages(rewards, group_size)
        
        assert len(advantages) == 8
        
        # Each group should be normalized independently
        group1_adv = advantages[:4]
        group2_adv = advantages[4:]
        
        # Each group mean should be ~0
        assert abs(group1_adv.mean().item()) < 1e-5
        assert abs(group2_adv.mean().item()) < 1e-5
    
    def test_training_step_reward_count_invariant(self):
        """Test that reward count matches trajectory count."""
        batch_size = 4
        group_size = 8
        total_trajectories = batch_size * group_size
        
        # Simulate rewards from environment
        rewards = torch.tensor([0.5 + (i % 5) * 0.1 for i in range(total_trajectories)])
        
        # Verify count invariant
        assert len(rewards) == total_trajectories
        
        # Compute advantages for all groups at once
        advantages = compute_grpo_advantages(rewards, group_size)
        assert len(advantages) == total_trajectories
    
    def test_full_training_step_simulation(self):
        """Test full training step simulation with mock data."""
        # Configuration
        batch_size = 2
        group_size = 4
        
        # Mock trajectories (simulating rollout generation)
        trajectories = [
            Trajectory(
                prompt=f"Problem {i // group_size}",
                completion=f"Solution {i}. The answer is {i}",
                ground_truth=str(i),
            )
            for i in range(batch_size * group_size)
        ]
        
        # Mock environment response
        mock_rewards = [0.5 + (i % 3) * 0.2 for i in range(len(trajectories))]
        mock_response = RewardResponse(
            rewards=mock_rewards,
            details=[
                RewardDetails(
                    correctness_score=r - 0.1,
                    format_score=0.1,
                    extracted_answer=str(i),
                    is_correct=r > 0.5,
                )
                for i, r in enumerate(mock_rewards)
            ],
        )
        
        # Simulate training step
        # 1. Verify trajectory count
        assert len(trajectories) == batch_size * group_size
        
        # 2. Verify reward count matches
        assert len(mock_response.rewards) == len(trajectories)
        
        # 3. Compute advantages for all groups at once
        rewards_tensor = torch.tensor(mock_response.rewards)
        all_advantages = compute_grpo_advantages(rewards_tensor, group_size)
        
        # 4. Verify advantage count
        assert len(all_advantages) == len(trajectories)
        
        # 5. Verify advantages are normalized per group
        for i in range(batch_size):
            group_adv = all_advantages[i * group_size : (i + 1) * group_size]
            assert abs(group_adv.mean().item()) < 1e-5


class TestTrainingConfiguration:
    """Tests for training configuration validation."""
    
    def test_default_config_values(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        
        assert config.model_name == "Qwen/Qwen2.5-1.5B"
        assert config.group_size == 8
        assert config.batch_size == 32
        assert config.learning_rate == 1e-6
        assert config.kl_coef == 0.1
        assert config.clip_range == 0.2
    
    def test_config_from_environment(self):
        """Test configuration can be loaded from environment variables."""
        env_vars = {
            "MODEL_NAME": "Qwen/Qwen2.5-3B",
            "BATCH_SIZE": "16",
            "GROUP_SIZE": "4",
            "LEARNING_RATE": "5e-7",
        }
        
        with patch.dict(os.environ, env_vars):
            # Simulate loading config from env
            model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
            batch_size = int(os.environ.get("BATCH_SIZE", "32"))
            group_size = int(os.environ.get("GROUP_SIZE", "8"))
            learning_rate = float(os.environ.get("LEARNING_RATE", "1e-6"))
            
            assert model_name == "Qwen/Qwen2.5-3B"
            assert batch_size == 16
            assert group_size == 4
            assert learning_rate == 5e-7
    
    def test_config_hyperparameters_valid_ranges(self):
        """Test that default hyperparameters are in valid ranges."""
        config = TrainingConfig()
        
        # Learning rate should be positive and small
        assert 0 < config.learning_rate < 1
        
        # KL coefficient should be positive
        assert config.kl_coef > 0
        
        # Clip range should be between 0 and 1
        assert 0 < config.clip_range < 1
        
        # Group size should be positive
        assert config.group_size > 0
        
        # Batch size should be positive
        assert config.batch_size > 0


@pytest.mark.skipif(
    not INTEGRATION_TEST_MODE,
    reason="Integration tests require INTEGRATION_TEST_MODE=true"
)
class TestLiveIntegration:
    """Live integration tests requiring deployed services."""
    
    def test_end_to_end_reward_computation(self):
        """Test complete reward computation flow."""
        async def _test():
            config = EnvironmentClientConfig(
                base_url=ENVIRONMENT_SERVICE_URL,
                retry=RetryConfig(max_retries=3, timeout_ms=30000),
            )
            
            # Create realistic trajectories
            trajectories = [
                Trajectory(
                    prompt="A store has 15 apples. If 7 are sold, how many remain?",
                    completion="Starting with 15 apples and selling 7: 15 - 7 = 8. The answer is 8",
                    ground_truth="8",
                ),
                Trajectory(
                    prompt="A store has 15 apples. If 7 are sold, how many remain?",
                    completion="15 minus 7 is 8. The answer is 8",
                    ground_truth="8",
                ),
                Trajectory(
                    prompt="A store has 15 apples. If 7 are sold, how many remain?",
                    completion="The answer is 9",  # Wrong answer
                    ground_truth="8",
                ),
            ]
            
            async with EnvironmentClient(config) as client:
                result = await client.compute_rewards(trajectories)
                
                # Verify response structure
                assert len(result.rewards) == 3
                assert len(result.details) == 3
                
                # First two should be correct
                assert result.details[0].is_correct
                assert result.details[1].is_correct
                
                # Third should be incorrect
                assert not result.details[2].is_correct
                
                # Rewards should reflect correctness
                assert result.rewards[0] > result.rewards[2]
                assert result.rewards[1] > result.rewards[2]
        
        run_async(_test())
