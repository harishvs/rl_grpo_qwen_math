#!/bin/bash
# Build and push Docker images to ECR
# Builds for linux/amd64 platform to match EKS nodes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Building Docker Images for EKS ===${NC}"

# Get AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_BASE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Repository names
ENV_REPO="rl-code-llm-training-dev/environment"
TRAINER_REPO="rl-code-llm-training-dev/trainer"

echo -e "${YELLOW}AWS Account: ${AWS_ACCOUNT_ID}${NC}"
echo -e "${YELLOW}ECR Base: ${ECR_BASE}${NC}"

# Login to ECR
echo -e "\n${YELLOW}Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_BASE}

# Build environment image for linux/amd64
echo -e "\n${YELLOW}Building environment image for linux/amd64...${NC}"
docker buildx build \
    --platform linux/amd64 \
    --push \
    -t ${ECR_BASE}/${ENV_REPO}:latest \
    -f docker/environment/Dockerfile \
    .

echo -e "${GREEN}✓ Environment image pushed: ${ECR_BASE}/${ENV_REPO}:latest${NC}"

# Build trainer image for linux/amd64
echo -e "\n${YELLOW}Building trainer image for linux/amd64...${NC}"
docker buildx build \
    --platform linux/amd64 \
    --push \
    -t ${ECR_BASE}/${TRAINER_REPO}:latest \
    -f docker/trainer/Dockerfile \
    .

echo -e "${GREEN}✓ Trainer image pushed: ${ECR_BASE}/${TRAINER_REPO}:latest${NC}"

echo -e "\n${GREEN}=== Build Complete ===${NC}"
echo -e "Environment Image: ${ECR_BASE}/${ENV_REPO}:latest"
echo -e "Trainer Image:     ${ECR_BASE}/${TRAINER_REPO}:latest"
echo -e "\nNext steps:"
echo -e "  1. Restart deployments: kubectl rollout restart deployment/environment-service"
echo -e "  2. Check pod status: kubectl get pods -l app=environment-service"
