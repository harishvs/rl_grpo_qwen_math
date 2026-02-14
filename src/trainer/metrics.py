"""
Metrics logging for GRPO training.

Provides structured logging of training metrics to stdout (for CloudWatch Logs)
and optionally to CloudWatch Metrics via the CloudWatch agent.

Requirements: 6.2
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training step metrics."""
    step: int
    epoch: int
    policy_loss: float
    kl_divergence: float
    mean_reward: float
    clip_fraction: float
    advantages_mean: float
    advantages_std: float
    learning_rate: float
    gradient_norm: Optional[float] = None
    environment_latency_ms: Optional[float] = None
    rollout_throughput: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MetricsConfig:
    """Configuration for metrics logging."""
    enabled: bool = True
    log_interval_steps: int = 10
    cloudwatch_namespace: str = "RLCodeLLMTraining"
    environment: str = "dev"
    model_name: str = "Qwen2.5-1.5B"
    log_format: str = "json"
    
    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.environ.get("METRICS_ENABLED", "true").lower() == "true",
            log_interval_steps=int(os.environ.get("METRICS_LOG_INTERVAL", "10")),
            cloudwatch_namespace=os.environ.get("CLOUDWATCH_NAMESPACE", "RLCodeLLMTraining"),
            environment=os.environ.get("ENVIRONMENT", "dev"),
            model_name=os.environ.get("MODEL_NAME", "Qwen2.5-1.5B"),
            log_format=os.environ.get("LOG_FORMAT", "json"),
        )


class MetricsLogger:
    """Logger for training metrics.
    
    Outputs structured JSON logs that can be collected by CloudWatch Logs
    and parsed into CloudWatch Metrics.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig.from_env()
        self._step_times: List[float] = []
        self._last_log_step = 0
        
        # Configure JSON logging
        if self.config.log_format == "json":
            self._setup_json_logging()
    
    def _setup_json_logging(self):
        """Configure JSON formatted logging."""
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        
        # Get root logger for metrics
        metrics_logger = logging.getLogger("training.metrics")
        metrics_logger.setLevel(logging.INFO)
        metrics_logger.addHandler(handler)
        metrics_logger.propagate = False
        
        self._metrics_logger = metrics_logger
    
    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step.
        
        Args:
            metrics: Training metrics for the current step
        """
        if not self.config.enabled:
            return
        
        # Only log at configured intervals
        if metrics.step - self._last_log_step < self.config.log_interval_steps:
            return
        
        self._last_log_step = metrics.step
        
        # Create structured log entry
        log_entry = {
            "type": "training_metrics",
            "namespace": self.config.cloudwatch_namespace,
            "dimensions": {
                "Environment": self.config.environment,
                "Model": self.config.model_name,
            },
            "metrics": metrics.to_dict(),
        }
        
        if self.config.log_format == "json":
            self._metrics_logger.info(json.dumps(log_entry))
        else:
            logger.info(f"Step {metrics.step}: loss={metrics.policy_loss:.4f}, "
                       f"reward={metrics.mean_reward:.4f}, kl={metrics.kl_divergence:.4f}")
    
    def log_epoch_summary(
        self,
        epoch: int,
        total_steps: int,
        avg_loss: float,
        avg_reward: float,
        avg_kl: float,
        duration_seconds: float,
    ):
        """Log summary metrics for an epoch.
        
        Args:
            epoch: Epoch number
            total_steps: Total steps in the epoch
            avg_loss: Average policy loss
            avg_reward: Average reward
            avg_kl: Average KL divergence
            duration_seconds: Epoch duration in seconds
        """
        if not self.config.enabled:
            return
        
        log_entry = {
            "type": "epoch_summary",
            "namespace": self.config.cloudwatch_namespace,
            "dimensions": {
                "Environment": self.config.environment,
                "Model": self.config.model_name,
            },
            "metrics": {
                "epoch": epoch,
                "total_steps": total_steps,
                "avg_policy_loss": avg_loss,
                "avg_reward": avg_reward,
                "avg_kl_divergence": avg_kl,
                "duration_seconds": duration_seconds,
                "steps_per_second": total_steps / duration_seconds if duration_seconds > 0 else 0,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        
        if self.config.log_format == "json":
            self._metrics_logger.info(json.dumps(log_entry))
        else:
            logger.info(f"Epoch {epoch} complete: {total_steps} steps, "
                       f"avg_loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}")
    
    def log_checkpoint(self, step: int, checkpoint_path: str, duration_seconds: float):
        """Log checkpoint save event.
        
        Args:
            step: Training step
            checkpoint_path: Path where checkpoint was saved
            duration_seconds: Time taken to save checkpoint
        """
        if not self.config.enabled:
            return
        
        log_entry = {
            "type": "checkpoint",
            "namespace": self.config.cloudwatch_namespace,
            "metrics": {
                "step": step,
                "checkpoint_path": checkpoint_path,
                "save_duration_seconds": duration_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        
        if self.config.log_format == "json":
            self._metrics_logger.info(json.dumps(log_entry))
        else:
            logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
    
    def log_environment_latency(self, latency_ms: float, batch_size: int):
        """Log environment service latency.
        
        Args:
            latency_ms: Latency in milliseconds
            batch_size: Number of trajectories in the batch
        """
        if not self.config.enabled:
            return
        
        log_entry = {
            "type": "environment_latency",
            "namespace": self.config.cloudwatch_namespace,
            "metrics": {
                "latency_ms": latency_ms,
                "batch_size": batch_size,
                "latency_per_trajectory_ms": latency_ms / batch_size if batch_size > 0 else 0,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        
        if self.config.log_format == "json":
            self._metrics_logger.info(json.dumps(log_entry))


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # If message is already JSON, return as-is
        if record.msg.startswith("{"):
            return record.msg
        
        # Otherwise wrap in JSON structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000
