"""
Property-based tests for the environment service contract.

**Validates: Requirements 3.2, 3.3, 3.5**

Property 4: Environment Service Contract
For any valid RewardRequest containing N trajectories:
- The environment service SHALL accept the request
- The environment service SHALL return a RewardResponse with exactly N rewards
- The response time SHALL be bounded (< timeout threshold)
"""

import time
import pytest
from hypothesis import given, settings, strategies as st
from fastapi.testclient import TestClient

from src.environment.service import app


# Timeout threshold in seconds (from service config)
TIMEOUT_THRESHOLD_SECONDS = 30.0


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


# Strategy for generating valid trajectory data
trajectory_strategy = st.fixed_dictionaries({
    "prompt": st.text(min_size=1, max_size=200),
    "completion": st.text(min_size=0, max_size=500),
    "ground_truth": st.text(min_size=1, max_size=20),
})

# Strategy for generating lists of trajectories (1 to 20 items)
trajectories_strategy = st.lists(trajectory_strategy, min_size=1, max_size=20)


class TestEnvironmentServiceContract:
    """
    Property-based tests for Environment Service Contract.
    
    **Validates: Requirements 3.2, 3.3, 3.5**
    """
    
    @given(trajectories=trajectories_strategy)
    @settings(max_examples=100, deadline=None)
    def test_response_length_matches_request(self, client, trajectories):
        """
        Property 4: Environment Service Contract
        
        For any valid RewardRequest containing N trajectories,
        the response SHALL contain exactly N rewards.
        
        **Validates: Requirements 3.2, 3.3, 3.5**
        """
        n = len(trajectories)
        
        start_time = time.time()
        response = client.post(
            "/compute_rewards",
            json={"trajectories": trajectories}
        )
        elapsed_time = time.time() - start_time
        
        # Service SHALL accept the request
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        
        # Response SHALL contain exactly N rewards
        assert len(data["rewards"]) == n, (
            f"Expected {n} rewards, got {len(data['rewards'])}"
        )
        
        # Response SHALL contain exactly N details
        assert len(data["details"]) == n, (
            f"Expected {n} details, got {len(data['details'])}"
        )
        
        # Response time SHALL be bounded
        assert elapsed_time < TIMEOUT_THRESHOLD_SECONDS, (
            f"Response time {elapsed_time}s exceeded threshold {TIMEOUT_THRESHOLD_SECONDS}s"
        )
    
    @given(trajectories=trajectories_strategy)
    @settings(max_examples=100, deadline=None)
    def test_rewards_are_valid_floats(self, client, trajectories):
        """
        Property: All returned rewards SHALL be valid floats in range [0.0, 1.2].
        
        **Validates: Requirements 3.3**
        """
        response = client.post(
            "/compute_rewards",
            json={"trajectories": trajectories}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for i, reward in enumerate(data["rewards"]):
            assert isinstance(reward, (int, float)), (
                f"Reward {i} is not a number: {reward}"
            )
            assert 0.0 <= reward <= 1.2, (
                f"Reward {i} out of range [0.0, 1.2]: {reward}"
            )
    
    @given(trajectories=trajectories_strategy)
    @settings(max_examples=100, deadline=None)
    def test_details_structure_valid(self, client, trajectories):
        """
        Property: All returned details SHALL have valid structure.
        
        **Validates: Requirements 3.3**
        """
        response = client.post(
            "/compute_rewards",
            json={"trajectories": trajectories}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for i, detail in enumerate(data["details"]):
            # correctness_score must be 0.0 or 1.0
            assert detail["correctness_score"] in [0.0, 1.0], (
                f"Detail {i} correctness_score invalid: {detail['correctness_score']}"
            )
            
            # format_score must be in [0.0, 0.2]
            assert 0.0 <= detail["format_score"] <= 0.2, (
                f"Detail {i} format_score out of range: {detail['format_score']}"
            )
            
            # is_correct must be boolean
            assert isinstance(detail["is_correct"], bool), (
                f"Detail {i} is_correct not boolean: {detail['is_correct']}"
            )
