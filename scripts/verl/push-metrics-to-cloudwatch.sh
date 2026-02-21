#!/bin/bash
# Scrapes veRL training metrics from Ray worker log and pushes to CloudWatch.
# Run locally — uses your AWS credentials. Safe to run repeatedly (idempotent).
#
# Usage: ./scripts/verl/push-metrics-to-cloudwatch.sh [--loop]
#   --loop: run every 5 minutes until training finishes

set -euo pipefail

HEAD_POD="${HEAD_POD:-verl-grpo-head-724lx}"
NAMESPACE="GRPO-Training"
RUN="7b-grpo-$(date -u +%Y%m%d)"
REGION="us-east-1"

# Find the worker log with step metrics
LOG_FILE=$(kubectl exec "$HEAD_POD" -- bash -c 'ls -t /tmp/ray/session_latest/logs/worker-*.out 2>/dev/null | head -1')
if [ -z "$LOG_FILE" ]; then
    echo "No worker log found"
    exit 1
fi

push_metrics() {
    kubectl exec "$HEAD_POD" -- grep "^step:" "$LOG_FILE" 2>/dev/null | \
    python3 -c "
import sys, json, subprocess, datetime

for line in sys.stdin:
    parts = dict(p.split(':',1) for p in line.strip().split(' - ') if ':' in p)
    step = parts.get('training/global_step', parts.get('step','0'))
    ts = datetime.datetime.utcnow().isoformat() + 'Z'

    metrics = []
    key_map = {
        'critic/score/mean': 'Reward',
        'actor/ppo_kl': 'KL',
        'actor/pg_loss': 'PolicyLoss',
        'actor/entropy': 'Entropy',
        'actor/grad_norm': 'GradNorm',
        'timing_s/step': 'StepTime',
        'response_length/mean': 'ResponseLength',
    }
    # Validation metrics
    for k, v in parts.items():
        if k.startswith('val-core'):
            key_map[k] = 'ValReward'

    for src, name in key_map.items():
        if src in parts:
            metrics.append({
                'MetricName': name,
                'Value': float(parts[src]),
                'Timestamp': ts,
                'Dimensions': [{'Name':'Run','Value':'$RUN'},{'Name':'Step','Value':str(step)}],
                'Unit': 'Seconds' if name == 'StepTime' else 'None'
            })

    if metrics:
        # CloudWatch accepts max 1000 metrics per call
        print(json.dumps(metrics))
" | while read -r batch; do
        aws cloudwatch put-metric-data \
            --region "$REGION" \
            --namespace "$NAMESPACE" \
            --metric-data "$batch" 2>&1
        echo "Pushed $(echo "$batch" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))')  metrics for step"
    done
}

push_metrics

if [ "${1:-}" = "--loop" ]; then
    echo "Looping every 5 minutes. Ctrl+C to stop."
    while true; do
        sleep 300
        push_metrics 2>/dev/null || echo "Push failed, retrying..."
    done
fi
