#!/bin/bash
# Build and push Docker images to ECR
# Usage: ./scripts/build-and-push-images.sh [environment|trainer|all]
#
# IMPORTANT: This script builds for linux/amd64 platform to ensure compatibility
# with EKS GPU nodes (x86_64 architecture). Do NOT use default platform builds
# when building from ARM-based machines (e.g., Apple Silicon Macs).

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Target platform for EKS GPU nodes (x86_64)
TARGET_PLATFORM="linux/amd64"

# Get AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-$(aws configure get region 2>/dev/null || echo "us-east-1")}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
PROJECT_NAME="rl-code-llm-training-dev"

# Image names
ENVIRONMENT_IMAGE="${ECR_REGISTRY}/${PROJECT_NAME}/environment"
TRAINER_IMAGE="${ECR_REGISTRY}/${PROJECT_NAME}/trainer"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default to building all images
BUILD_TARGET=${1:-all}

echo -e "${GREEN}=== Docker Image Build Script ===${NC}"
echo -e "AWS Account:     ${AWS_ACCOUNT_ID}"
echo -e "AWS Region:      ${AWS_REGION}"
echo -e "ECR Registry:    ${ECR_REGISTRY}"
echo -e "Target Platform: ${TARGET_PLATFORM}"
echo -e "Build Target:    ${BUILD_TARGET}"
echo ""

# Verify docker buildx is available for cross-platform builds
if ! docker buildx version &>/dev/null; then
    echo -e "${RED}Error: docker buildx is required for cross-platform builds${NC}"
    echo -e "Install with: docker buildx install"
    exit 1
fi

# Create/use buildx builder for cross-platform support
BUILDER_NAME="eks-multiplatform"
if ! docker buildx inspect ${BUILDER_NAME} &>/dev/null; then
    echo -e "${YELLOW}Creating buildx builder for cross-platform builds...${NC}"
    docker buildx create --name ${BUILDER_NAME} --use
else
    docker buildx use ${BUILDER_NAME}
fi
echo -e "${GREEN}✓ Using buildx builder: ${BUILDER_NAME}${NC}"
echo ""

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
echo -e "${GREEN}✓ ECR login successful${NC}"
echo ""

build_environment() {
    echo -e "${YELLOW}Building environment service image for ${TARGET_PLATFORM}...${NC}"
    
    # Create a temporary build context
    BUILD_DIR=$(mktemp -d)
    trap "rm -rf ${BUILD_DIR}" EXIT
    
    # Copy necessary files
    cp -r "${PROJECT_ROOT}/src" "${BUILD_DIR}/"
    cp "${PROJECT_ROOT}/docker/environment/Dockerfile" "${BUILD_DIR}/"
    
    # Create requirements.txt for environment service
    cat > "${BUILD_DIR}/requirements.txt" << 'EOF'
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
httpx>=0.25.0
EOF
    
    # Build and push the image for target platform
    docker buildx build \
        --platform ${TARGET_PLATFORM} \
        -t "${ENVIRONMENT_IMAGE}:latest" \
        -f "${BUILD_DIR}/Dockerfile" \
        --push \
        "${BUILD_DIR}"
    
    echo -e "${GREEN}✓ Environment image built and pushed: ${ENVIRONMENT_IMAGE}:latest${NC}"
}

build_trainer() {
    echo -e "${YELLOW}Building trainer image for ${TARGET_PLATFORM}...${NC}"
    
    # Build from project root to preserve directory structure expected by Dockerfile
    docker buildx build \
        --platform ${TARGET_PLATFORM} \
        -t "${TRAINER_IMAGE}:latest" \
        -f "${PROJECT_ROOT}/docker/trainer/Dockerfile" \
        --push \
        "${PROJECT_ROOT}"
    
    echo -e "${GREEN}✓ Trainer image built and pushed: ${TRAINER_IMAGE}:latest${NC}"
}

case ${BUILD_TARGET} in
    environment)
        build_environment
        ;;
    trainer)
        build_trainer
        ;;
    all)
        build_environment
        build_trainer
        ;;
    *)
        echo -e "${RED}Unknown build target: ${BUILD_TARGET}${NC}"
        echo "Usage: $0 [environment|trainer|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "Environment Image: ${ENVIRONMENT_IMAGE}:latest"
echo -e "Trainer Image:     ${TRAINER_IMAGE}:latest"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Restart deployments: kubectl rollout restart deployment/environment-service"
echo -e "  2. Check pod status: kubectl get pods -l app=environment-service"
