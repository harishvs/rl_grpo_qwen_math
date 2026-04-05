#!/bin/bash
# Collect training logs from CloudWatch + live pods, then push to GitHub.
# CloudWatch logs persist after pod termination (via Fluent Bit DaemonSet).
# This script downloads them so we have a permanent record in the repo.
#
# Usage: ./scripts/monarch/collect-logs.sh [--run-id RUN_ID]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
AWS_REGION="${AWS_REGION:-us-west-2}"
LOG_GROUP="/aws/eks/rl-code-llm-training-dev/containers"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d-%H%M)}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RUN_DIR="${PROJECT_ROOT}/logs/monarch/run-${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "=== Collecting Monarch GRPO Training Logs ==="
echo "Run ID:  ${RUN_ID}"
echo "Output:  ${RUN_DIR}"

# --- 1. Download from CloudWatch (survives pod termination) ---
echo ""
echo "--- CloudWatch Logs ---"

# Find log streams for monarch pods (last 24 hours)
START_TIME=$(( $(date +%s) * 1000 - 86400000 ))

for FILTER in "monarch-grpo" "monarch-reward"; do
    echo "Searching CloudWatch for '${FILTER}' streams..."
    STREAMS=$(aws logs describe-log-streams \
        --log-group-name "${LOG_GROUP}" \
        --log-stream-name-prefix "${FILTER}" \
        --order-by LastEventTime \
        --descending \
        --limit 10 \
        --region "${AWS_REGION}" \
        --query 'logStreams[*].logStreamName' \
        --output text 2>/dev/null || echo "")

    if [[ -z "${STREAMS}" ]]; then
        echo "  No CloudWatch streams found for '${FILTER}'"
        continue
    fi

    for STREAM in ${STREAMS}; do
        SAFE_NAME=$(echo "${STREAM}" | tr '/' '_')
        echo "  Downloading: ${STREAM}"
        aws logs get-log-events \
            --log-group-name "${LOG_GROUP}" \
            --log-stream-name "${STREAM}" \
            --start-time "${START_TIME}" \
            --region "${AWS_REGION}" \
            --query 'events[*].message' \
            --output text \
            > "${RUN_DIR}/cloudwatch-${SAFE_NAME}.log" 2>/dev/null || true
    done
done

# --- 2. Collect from live pods (if still running) ---
echo ""
echo "--- Live Pod Logs ---"

for POD in $(kubectl get pods -l app=monarch-grpo --no-headers 2>/dev/null | awk '{print $1}'); do
    echo "  Collecting from ${POD}..."
    kubectl logs "${POD}" > "${RUN_DIR}/${POD}.log" 2>&1 || true
    kubectl exec "${POD}" -- cat /tmp/training.log > "${RUN_DIR}/${POD}-training.log" 2>&1 || true
done

for POD in $(kubectl get pods -l app=monarch-reward --no-headers 2>/dev/null | awk '{print $1}'); do
    echo "  Collecting from ${POD}..."
    kubectl logs "${POD}" > "${RUN_DIR}/${POD}.log" 2>&1 || true
done

# --- 3. Collect pod status ---
echo ""
echo "--- Pod Status ---"
kubectl get pods -l app=monarch-grpo -o wide > "${RUN_DIR}/pod-status.txt" 2>&1 || true
kubectl describe pods -l app=monarch-grpo > "${RUN_DIR}/pod-describe.txt" 2>&1 || true

# --- 4. Summary ---
echo ""
echo "--- Files Collected ---"
ls -lh "${RUN_DIR}/"
TOTAL_SIZE=$(du -sh "${RUN_DIR}" | awk '{print $1}')
echo "Total: ${TOTAL_SIZE}"

echo ""
echo "Done. Logs saved locally at: ${RUN_DIR}"
echo "Review for sensitive info before sharing."
