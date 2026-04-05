"""Prometheus metrics exporter for Monarch GRPO training.

Exposes training metrics on port 9090 for scraping by the existing
Prometheus/Grafana monitoring stack. Same metric pattern as veRL's
cloudwatch_metrics.py.
"""
from prometheus_client import Gauge, start_http_server

METRICS_PORT = 9090

# Training metrics
_gauges = {}

METRIC_NAMES = [
    "monarch_policy_loss",
    "monarch_kl_divergence",
    "monarch_mean_reward",
    "monarch_clip_fraction",
    "monarch_grad_norm",
    "monarch_step_time_seconds",
    "monarch_throughput_samples_per_sec",
    "monarch_global_step",
    "monarch_epoch",
    "monarch_replay_buffer_size",
    "monarch_generator_policy_version",
    "monarch_learner_policy_version",
]


def _get_gauge(name: str) -> Gauge:
    """Get or create a Prometheus Gauge."""
    if name not in _gauges:
        _gauges[name] = Gauge(name, name.replace("_", " "), ["run"])
    return _gauges[name]


class MetricsLogger:
    """Logs training metrics to Prometheus and stdout."""

    def __init__(self, experiment_name: str = "default", port: int = METRICS_PORT):
        self.experiment_name = experiment_name
        self.port = port
        self._started = False

    def start(self):
        """Start the Prometheus HTTP server."""
        if not self._started:
            start_http_server(self.port)
            self._started = True
            print(f"Prometheus metrics server on :{self.port}", flush=True)

    def log(self, step: int, metrics: dict):
        """Log metrics for a training step.

        Args:
            step: Current training step
            metrics: Dict of metric name -> value
        """
        run = self.experiment_name

        # Always set global step
        _get_gauge("monarch_global_step").labels(run=run).set(step)

        # Set all provided metrics
        for key, value in metrics.items():
            gauge_name = f"monarch_{key}" if not key.startswith("monarch_") else key
            if gauge_name in METRIC_NAMES or gauge_name.startswith("monarch_"):
                _get_gauge(gauge_name).labels(run=run).set(float(value))

        # Print to stdout for log collection
        metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"step:{step} | {metrics_str}", flush=True)
