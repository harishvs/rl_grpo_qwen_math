#!/bin/bash
# Build and push Monarch GRPO Docker images to ECR
# Builds two images: GPU (training) and reward (lightweight CPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
TAG="${1:-latest}"

# Authenticate to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# --- 1. GPU image (training + generation) ---
GPU_IMAGE="rl-code-llm-training-dev/monarch-grpo"
echo "=== Building GPU image: ${GPU_IMAGE}:${TAG} ==="
aws ecr describe-repositories --repository-names "${GPU_IMAGE}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${GPU_IMAGE}" --region "${AWS_REGION}"

# --network=host: required on EC2 with systemd-resolved, otherwise DNS fails inside build containers
docker build \
    --network=host \
    -t "${ECR_REGISTRY}/${GPU_IMAGE}:${TAG}" \
    -f "${PROJECT_ROOT}/docker/monarch/Dockerfile" \
    "${PROJECT_ROOT}"
docker push "${ECR_REGISTRY}/${GPU_IMAGE}:${TAG}"

echo "GPU image pushed: ${ECR_REGISTRY}/${GPU_IMAGE}:${TAG}"

# --- 2. Reward image (lightweight CPU) ---
REWARD_IMAGE="rl-code-llm-training-dev/monarch-reward"
echo "=== Building reward image: ${REWARD_IMAGE}:${TAG} ==="
aws ecr describe-repositories --repository-names "${REWARD_IMAGE}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${REWARD_IMAGE}" --region "${AWS_REGION}"

docker build \
    --network=host \
    -t "${ECR_REGISTRY}/${REWARD_IMAGE}:${TAG}" \
    -f "${PROJECT_ROOT}/docker/monarch/Dockerfile.reward" \
    "${PROJECT_ROOT}"
docker push "${ECR_REGISTRY}/${REWARD_IMAGE}:${TAG}"

echo "Reward image pushed: ${ECR_REGISTRY}/${REWARD_IMAGE}:${TAG}"

echo "=== All images built and pushed ==="
