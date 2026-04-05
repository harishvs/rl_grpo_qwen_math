#!/bin/bash
# Deploy Monarch GRPO training on EKS
# Usage: ./scripts/monarch/run-training.sh [--config CONFIG] [--follow]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s/monarch"

# Defaults
CONFIG="qwen-1.5b.yaml"
FOLLOW=false
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --follow) FOLLOW=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Monarch GRPO Training ==="
echo "Config: ${CONFIG}"

# Check cluster connectivity
echo "Checking cluster..."
kubectl cluster-info > /dev/null 2>&1 || { echo "ERROR: kubectl not connected to cluster"; exit 1; }

# Install MonarchMesh operator if not present
if ! kubectl get crd monarchmeshes.monarch.meta.com > /dev/null 2>&1; then
    echo "Installing MonarchMesh operator..."
    helm repo add monarch-operator https://meta-pytorch.github.io/monarch-kubernetes 2>/dev/null || true
    helm repo update
    helm upgrade --install monarch-operator monarch-operator/monarch-operator \
        --namespace monarch-system --create-namespace
    echo "Waiting for operator to be ready..."
    kubectl wait --for=condition=available deployment/monarch-controller-manager \
        --namespace monarch-system --timeout=120s
    echo "MonarchMesh operator installed"
else
    echo "MonarchMesh operator already installed"
fi

# Apply manifests (substitute ACCOUNT_ID and REGION placeholders)
echo "Deploying manifests..."
echo "  ECR Registry: ${ECR_REGISTRY}"
for MANIFEST in serviceaccount.yaml configmap.yaml monarchmesh.yaml reward-deployment.yaml; do
    sed -e "s|ACCOUNT_ID|${AWS_ACCOUNT_ID}|g" \
        -e "s|us-east-1\.amazonaws\.com|${AWS_REGION}.amazonaws.com|g" \
        "${K8S_DIR}/${MANIFEST}" | kubectl apply -f -
done

# Wait for GPU pods
echo "Waiting for Monarch worker pods..."
for i in $(seq 1 60); do
    READY=$(kubectl get pods -l monarch.pytorch.org/mesh-name=grpo-monarch --no-headers 2>/dev/null | grep -c Running || true)
    if [[ "${READY}" -ge 2 ]]; then
        echo "All GPU pods running (${READY}/2)"
        break
    fi
    echo "  Waiting for pods... (${READY}/2 running, attempt ${i}/60)"
    sleep 5
done

# Wait for reward pod
echo "Waiting for reward pod..."
kubectl wait --for=condition=available deployment/monarch-reward --timeout=120s 2>/dev/null || true

# Identify controller pod (first GPU pod)
CONTROLLER_POD=$(kubectl get pods -l monarch.pytorch.org/mesh-name=grpo-monarch --no-headers | head -1 | awk '{print $1}')
echo "Controller pod: ${CONTROLLER_POD}"

# Download model on all pods
echo "Pre-downloading model..."
kubectl exec "${CONTROLLER_POD}" -- python -c \
    "from transformers import AutoModelForCausalLM, AutoTokenizer; m='$(grep 'model:' "${PROJECT_ROOT}/config/monarch/${CONFIG}" | head -1 | awk '{print $2}')'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m); print('Model cached')"

# Prep data
echo "Preparing data..."
kubectl exec "${CONTROLLER_POD}" -- python /scripts/prep_data.py

# Launch training
echo "Launching training..."
kubectl exec "${CONTROLLER_POD}" -- bash -c \
    "nohup python /scripts/run_training.py /config/${CONFIG%.yaml}.yaml > /tmp/training.log 2>&1 &"

echo "Training launched on ${CONTROLLER_POD}"
echo "View logs: kubectl exec ${CONTROLLER_POD} -- tail -f /tmp/training.log"

if [[ "${FOLLOW}" == "true" ]]; then
    echo "Following logs..."
    sleep 5
    kubectl exec "${CONTROLLER_POD}" -- tail -f /tmp/training.log

    # After training completes (tail exits), collect logs from CloudWatch
    echo ""
    echo "Training finished. Collecting logs from CloudWatch..."
    "${SCRIPT_DIR}/collect-logs.sh" --run-id "$(date +%Y-%m-%d-%H%M)"
fi
