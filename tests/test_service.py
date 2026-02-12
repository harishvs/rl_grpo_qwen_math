"""Unit tests for the FastAPI environment service."""

import pytest
from fastapi.testclient import TestClient

from src.environment.service import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestEnvironmentService:
    """Tests for the environment service endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_compute_rewards_empty(self, client):
        """Test compute_rewards with empty trajectories."""
        response = client.post(
            "/compute_rewards",
            json={"trajectories": []}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rewards"] == []
        assert data["details"] == []
    
    def test_compute_rewards_single(self, client):
        """Test compute_rewards with single trajectory."""
        response = client.post(
            "/compute_rewards",
            json={
                "trajectories": [{
                    "prompt": "What is 5 + 3?",
                    "completion": "First, 5 + 3 = 8. The answer is 8",
                    "ground_truth": "8"
                }]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rewards"]) == 1
        assert data["rewards"][0] == 1.2  # 1.0 correct + 0.2 format
        assert data["details"][0]["is_correct"] is True
    
    def test_compute_rewards_batch(self, client):
        """Test compute_rewards with multiple trajectories."""
        response = client.post(
            "/compute_rewards",
            json={
                "trajectories": [
                    {
                        "prompt": "What is 5 + 3?",
                        "completion": "The answer is 8",
                        "ground_truth": "8"
                    },
                    {
                        "prompt": "What is 10 - 4?",
                        "completion": "The answer is 5",
                        "ground_truth": "6"
                    }
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rewards"]) == 2
        assert len(data["details"]) == 2
    
    def test_compute_rewards_response_length_matches_request(self, client):
        """Test that response length matches request length."""
        trajectories = [
            {"prompt": f"Q{i}", "completion": f"A{i}", "ground_truth": str(i)}
            for i in range(5)
        ]
        response = client.post(
            "/compute_rewards",
            json={"trajectories": trajectories}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rewards"]) == len(trajectories)
        assert len(data["details"]) == len(trajectories)
