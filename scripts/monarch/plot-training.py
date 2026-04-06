#!/usr/bin/env python3
"""Parse Monarch GRPO training log and generate metric plots as PNGs.

Usage:
    python scripts/monarch/plot-training.py <training.log> <output_dir>

Reads lines like:
    step:0 | policy_loss=-0.0003 | kl_divergence=0.0005 | mean_reward=0.1523 | ...
"""
import re
import sys
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    metrics = defaultdict(list)
    steps = []
    pattern = re.compile(r"(\w+)=([-\d.]+)")

    with open(path) as f:
        for line in f:
            if not line.startswith("step:"):
                continue
            # Parse step number
            step_match = re.match(r"step:(\d+)", line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            steps.append(step)

            # Parse all key=value pairs
            for key, val in pattern.findall(line):
                if key == "epoch":
                    continue
                try:
                    metrics[key].append(float(val))
                except ValueError:
                    pass

    return steps, dict(metrics)


def plot_metric(steps, values, name, ylabel, output_path, color="tab:blue"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, values, color=color, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Monarch GRPO — {name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot-training.py <training.log> <output_dir>")
        sys.exit(1)

    log_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    steps, metrics = parse_log(log_path)
    print(f"Parsed {len(steps)} steps, metrics: {list(metrics.keys())}")

    plots = [
        ("mean_reward", "Mean Reward", "Reward (fraction correct)", "tab:green"),
        ("policy_loss", "Policy Loss", "Loss", "tab:red"),
        ("kl_divergence", "KL Divergence", "KL", "tab:orange"),
        ("clip_fraction", "Clip Fraction", "Fraction clipped", "tab:purple"),
        ("grad_norm", "Gradient Norm", "Grad norm", "tab:brown"),
        ("throughput_samples_per_sec", "Throughput", "Samples/sec", "tab:blue"),
        ("step_time_seconds", "Step Time", "Seconds", "tab:gray"),
    ]

    for key, name, ylabel, color in plots:
        if key in metrics and len(metrics[key]) == len(steps):
            plot_metric(steps, metrics[key], name, ylabel,
                        os.path.join(output_dir, f"{key}.png"), color)

    # Combined overview plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    overview = [
        (axes[0, 0], "mean_reward", "Mean Reward", "tab:green"),
        (axes[0, 1], "policy_loss", "Policy Loss", "tab:red"),
        (axes[1, 0], "kl_divergence", "KL Divergence", "tab:orange"),
        (axes[1, 1], "grad_norm", "Gradient Norm", "tab:brown"),
    ]
    for ax, key, title, color in overview:
        if key in metrics:
            ax.plot(steps, metrics[key], color=color, linewidth=1.0)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Monarch GRPO Training Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overview.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(output_dir, 'overview.png')}")


if __name__ == "__main__":
    main()
