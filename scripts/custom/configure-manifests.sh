#!/bin/bash
# Configure Kubernetes manifests with AWS account-specific values
# This script fetches values from AWS and updates placeholder values in manifests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Configuring Kubernetes manifests...${NC}"

# Get AWS Account ID
echo -e "${YELLOW}Fetching AWS Account ID...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Error: Could not fetch AWS Account ID. Ensure AWS CLI is configured.${NC}"
    exit 1
fi
echo -e "  Account ID: ${AWS_ACCOUNT_ID}"

# Get AWS Region (default to us-east-1 if not set)
AWS_REGION=${AWS_REGION:-$(aws configure get region 2>/dev/null || echo "us-east-1")}
echo -e "  Region: ${AWS_REGION}"

# Derive ECR repository URI
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/rl-grpo-training"
echo -e "  ECR Repo: ${ECR_REPO}"

# Derive S3 bucket name (using a convention)
S3_BUCKET="rl-grpo-checkpoints-${AWS_ACCOUNT_ID}-${AWS_REGION}"
echo -e "  S3 Bucket: ${S3_BUCKET}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "\n${YELLOW}Updating manifests...${NC}"

# Update serviceaccount.yaml
SA_FILE="${PROJECT_ROOT}/k8s/custom/trainer/serviceaccount.yaml"
if [ -f "$SA_FILE" ]; then
    sed -i.bak "s/ACCOUNT_ID/${AWS_ACCOUNT_ID}/g" "$SA_FILE"
    rm -f "${SA_FILE}.bak"
    echo -e "  ${GREEN}✓${NC} Updated serviceaccount.yaml"
else
    echo -e "  ${RED}✗${NC} serviceaccount.yaml not found"
fi

# Update secrets.yaml
SECRETS_FILE="${PROJECT_ROOT}/k8s/custom/config/secrets.yaml"
if [ -f "$SECRETS_FILE" ]; then
    sed -i.bak "s/your-checkpoint-bucket-name/${S3_BUCKET}/g" "$SECRETS_FILE"
    rm -f "${SECRETS_FILE}.bak"
    echo -e "  ${GREEN}✓${NC} Updated secrets.yaml"
else
    echo -e "  ${RED}✗${NC} secrets.yaml not found"
fi

# Update job.yaml
JOB_FILE="${PROJECT_ROOT}/k8s/custom/trainer/job.yaml"
if [ -f "$JOB_FILE" ]; then
    sed -i.bak "s|\${ECR_REPO}|${ECR_REPO}|g" "$JOB_FILE"
    rm -f "${JOB_FILE}.bak"
    echo -e "  ${GREEN}✓${NC} Updated job.yaml"
else
    echo -e "  ${RED}✗${NC} job.yaml not found"
fi

# Update environment deployment.yaml
ENV_DEPLOY_FILE="${PROJECT_ROOT}/k8s/custom/environment/deployment.yaml"
if [ -f "$ENV_DEPLOY_FILE" ]; then
    sed -i.bak "s|\${ECR_REPO}|${ECR_REPO}|g" "$ENV_DEPLOY_FILE"
    rm -f "${ENV_DEPLOY_FILE}.bak"
    echo -e "  ${GREEN}✓${NC} Updated environment deployment.yaml"
else
    echo -e "  ${RED}✗${NC} environment deployment.yaml not found"
fi

echo -e "\n${GREEN}Configuration complete!${NC}"
echo -e "\nConfigured values:"
echo -e "  AWS Account ID: ${AWS_ACCOUNT_ID}"
echo -e "  AWS Region:     ${AWS_REGION}"
echo -e "  ECR Repository: ${ECR_REPO}"
echo -e "  S3 Bucket:      ${S3_BUCKET}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Create ECR repository: aws ecr create-repository --repository-name rl-grpo-training"
echo -e "  2. Create S3 bucket: aws s3 mb s3://${S3_BUCKET}"
echo -e "  3. Build and push Docker images"
echo -e "  4. Deploy with: ./scripts/deploy-services.sh"
