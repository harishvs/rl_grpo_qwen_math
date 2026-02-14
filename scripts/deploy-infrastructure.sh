#!/bin/bash
# Deploy Infrastructure Script
# Provisions AWS infrastructure using Terraform
# Requirements: 8.7

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TERRAFORM_DIR="${PROJECT_ROOT}/terraform/environments/dev"
BACKEND_CONFIG="${TERRAFORM_DIR}/backend.hcl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy AWS infrastructure for RL Code LLM Training using Terraform.

OPTIONS:
    -h, --help          Show this help message
    -p, --plan          Run terraform plan only (no apply)
    -d, --destroy       Destroy infrastructure
    -a, --auto-approve  Auto-approve terraform apply
    -e, --env ENV       Environment to deploy (default: dev)

EXAMPLES:
    $(basename "$0")                    # Interactive apply
    $(basename "$0") --plan             # Plan only
    $(basename "$0") --auto-approve     # Apply without confirmation
    $(basename "$0") --destroy          # Destroy infrastructure
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check terraform
    if ! command -v terraform &> /dev/null; then
        log_error "terraform is not installed. Please install terraform first."
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

init_terraform() {
    log_info "Initializing Terraform..."
    cd "${TERRAFORM_DIR}"
    
    if [[ -f "${BACKEND_CONFIG}" ]]; then
        terraform init -backend-config="${BACKEND_CONFIG}"
    else
        log_warn "Backend config not found at ${BACKEND_CONFIG}, using local state."
        terraform init
    fi
}

validate_terraform() {
    log_info "Validating Terraform configuration..."
    cd "${TERRAFORM_DIR}"
    terraform validate
}

plan_terraform() {
    log_info "Running Terraform plan..."
    cd "${TERRAFORM_DIR}"
    terraform plan -out=tfplan
}

apply_terraform() {
    local auto_approve=$1
    log_info "Applying Terraform configuration..."
    cd "${TERRAFORM_DIR}"
    
    if [[ "${auto_approve}" == "true" ]]; then
        terraform apply -auto-approve tfplan
    else
        terraform apply tfplan
    fi
}

destroy_terraform() {
    local auto_approve=$1
    log_warn "Destroying infrastructure..."
    cd "${TERRAFORM_DIR}"
    
    if [[ "${auto_approve}" == "true" ]]; then
        terraform destroy -auto-approve
    else
        terraform destroy
    fi
}

configure_kubectl() {
    log_info "Configuring kubectl for EKS cluster..."
    cd "${TERRAFORM_DIR}"
    
    # Get cluster name from Terraform output
    local cluster_name
    cluster_name=$(terraform output -raw cluster_name 2>/dev/null || echo "")
    
    if [[ -z "${cluster_name}" ]]; then
        log_warn "Could not get cluster name from Terraform output. Skipping kubectl configuration."
        return
    fi
    
    local region
    region=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
    
    aws eks update-kubeconfig --name "${cluster_name}" --region "${region}"
    log_info "kubectl configured for cluster: ${cluster_name}"
}

main() {
    local plan_only=false
    local destroy=false
    local auto_approve=false
    local environment="dev"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -p|--plan)
                plan_only=true
                shift
                ;;
            -d|--destroy)
                destroy=true
                shift
                ;;
            -a|--auto-approve)
                auto_approve=true
                shift
                ;;
            -e|--env)
                environment="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Update paths for environment
    TERRAFORM_DIR="${PROJECT_ROOT}/terraform/environments/${environment}"
    BACKEND_CONFIG="${TERRAFORM_DIR}/backend.hcl"
    
    if [[ ! -d "${TERRAFORM_DIR}" ]]; then
        log_error "Environment directory not found: ${TERRAFORM_DIR}"
        exit 1
    fi
    
    log_info "Deploying infrastructure for environment: ${environment}"
    
    check_prerequisites
    init_terraform
    validate_terraform
    
    if [[ "${destroy}" == "true" ]]; then
        destroy_terraform "${auto_approve}"
        log_info "Infrastructure destroyed successfully."
        exit 0
    fi
    
    plan_terraform
    
    if [[ "${plan_only}" == "true" ]]; then
        log_info "Plan complete. Review the plan above."
        exit 0
    fi
    
    apply_terraform "${auto_approve}"
    configure_kubectl
    
    log_info "Infrastructure deployment complete!"
    log_info "Next steps:"
    log_info "  1. Run ./scripts/deploy-services.sh to deploy Kubernetes services"
    log_info "  2. Run ./scripts/run-training.sh to start the training job"
}

main "$@"
