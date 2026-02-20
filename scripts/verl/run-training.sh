#!/bin/bash
# Run veRL GRPO training on EKS
# Usage: ./scripts/verl/run-training.sh [--model MODEL] [--follow]
#
# Options:
#   --model MODEL   HuggingFace model path (default: Qwen/Qwen2.5-1.5B)
#   --follow        Follow training logs after launch

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL="Qwen/Qwen2.5-1.5B"
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --follow) FOLLOW=true; shift ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

echo -e "${GREEN}=== veRL GRPO Training ===${NC}"
echo "Model: $MODEL"
echo ""

# Verify cluster access
if ! kubectl cluster-info &>/dev/null; then
    echo -e "${RED}Error: kubectl not configured or cluster unreachable${NC}"
    exit 1
fi

# Deploy configmap and raycluster
echo -e "${YELLOW}Deploying veRL manifests...${NC}"
kubectl apply -f "${PROJECT_ROOT}/k8s/verl/configmap.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/verl/raycluster.yaml"
echo -e "${GREEN}✓ Manifests applied${NC}"
echo ""

# Wait for head pod
echo -e "${YELLOW}Waiting for head pod...${NC}"
HEAD_POD=""
for i in {1..60}; do
    HEAD_POD=$(kubectl get pods -l ray.io/cluster=verl-grpo,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "$HEAD_POD" ]]; then
        STATUS=$(kubectl get pod "$HEAD_POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)
        if [[ "$STATUS" == "Running" ]]; then
            echo -e "${GREEN}✓ Head pod running: $HEAD_POD${NC}"
            break
        fi
    fi
    echo -n "."
    sleep 5
done
echo ""

if [[ -z "$HEAD_POD" || "$STATUS" != "Running" ]]; then
    echo -e "${RED}Error: Head pod not ready after 5 minutes${NC}"
    kubectl get pods -l ray.io/cluster=verl-grpo
    exit 1
fi

# Wait for worker
echo -e "${YELLOW}Waiting for worker pod...${NC}"
for i in {1..60}; do
    WORKER_READY=$(kubectl get pods -l ray.io/cluster=verl-grpo,ray.io/node-type=worker -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    if [[ "$WORKER_READY" == "Running" ]]; then
        echo -e "${GREEN}✓ Worker pod running${NC}"
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

# Check Ray cluster has all nodes
echo -e "${YELLOW}Checking Ray cluster...${NC}"
RAY_NODES=$(kubectl exec "$HEAD_POD" -- ray status 2>/dev/null | grep -c "node_" || true)
echo "Ray nodes detected: $RAY_NODES"
echo ""

# Install veRL on all pods
echo -e "${YELLOW}Installing veRL on all pods...${NC}"
kubectl exec "$HEAD_POD" -- pip install verl==0.6.1 2>&1 | tail -1
WORKER_POD=$(kubectl get pods -l ray.io/cluster=verl-grpo,ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [[ -n "$WORKER_POD" ]]; then
    kubectl exec "$WORKER_POD" -- pip install verl==0.6.1 2>&1 | tail -1
fi
echo -e "${GREEN}✓ veRL installed on all nodes${NC}"
echo ""

# If model is not default, patch the run script
if [[ "$MODEL" != "Qwen/Qwen2.5-1.5B" ]]; then
    echo -e "${YELLOW}Patching model to: $MODEL${NC}"
    kubectl exec "$HEAD_POD" -- sed -i "s|Qwen/Qwen2.5-1.5B|${MODEL}|g" /scripts/run_grpo.sh
fi

# Launch training
echo -e "${GREEN}=== Launching training ===${NC}"
kubectl exec "$HEAD_POD" -- bash -c "nohup bash /scripts/run_grpo.sh > /tmp/training.log 2>&1 &"
echo -e "${GREEN}✓ Training started in background${NC}"
echo ""

echo -e "${YELLOW}To follow logs:${NC}"
echo "  kubectl exec $HEAD_POD -- tail -f /tmp/training.log"
echo ""
echo -e "${YELLOW}To check Ray dashboard:${NC}"
echo "  kubectl port-forward $HEAD_POD 8265:8265"

if [[ "$FOLLOW" == true ]]; then
    echo ""
    echo -e "${GREEN}Following training logs...${NC}"
    sleep 5
    kubectl exec "$HEAD_POD" -- tail -f /tmp/training.log
fi
