"""FastAPI environment service for reward computation.

Re-exports the shared reward service. The custom trainer's K8s deployment
and Dockerfile reference this path, so it must remain here.

The shared service at src/shared/reward_service.py provides both:
- POST /compute_rewards (custom trainer compatible)
- POST /score (simpler Monarch interface)
"""
from src.shared.reward_service import app

__all__ = ["app"]
