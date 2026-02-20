#!/bin/bash
# Deploy Services Script
# Deploys Kubernetes services for RL Code LLM Training
# Requirements: 8.7

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s"

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

Deploy Kubernetes services for RL Code LLM Training.

OPTIONS:
    -h, --help              Show this help message
    -n, --namespace NS      Kubernetes namespace (default: default)
    -d, --dry-run           Run kubectl apply with --dry-run=client
    --delete                Delete all services
    --env-only              Deploy environment service only
    --config-only           Deploy config and secrets only

EXAMPLES:
    $(basename "$0")                    # Deploy all services
    $(basename "$0") --dry-run          # Validate manifests only
    $(basename "$0") --env-only         # Deploy environment service only
    $(basename "$0") --delete           # Delete all services
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please configure kubectl first."
        log_error "Run: aws eks update-kubeconfig --name <cluster-name> --region <region>"
        exit 1
    fi
    
    log_info "All prerequisites met."
}

create_namespace() {
    local namespace=$1
    
    if [[ "${namespace}" != "default" ]]; then
        log_info "Creating namespace: ${namespace}"
        kubectl create namespace "${namespace}" --dry-run=client -o yaml | kubectl apply -f -
    fi
}

deploy_config() {
    local namespace=$1
    local dry_run=$2
    
    log_info "Deploying configuration..."
    
    local dry_run_flag=""
    if [[ "${dry_run}" == "true" ]]; then
        dry_run_flag="--dry-run=client"
    fi
    
    # Deploy training config
    kubectl apply -f "${K8S_DIR}/config/training-config.yaml" \
        -n "${namespace}" ${dry_run_flag}
    
    log_info "Configuration deployed."
}

deploy_environment_service() {
    local namespace=$1
    local dry_run=$2
    
    log_info "Deploying environment service..."
    
    local dry_run_flag=""
    if [[ "${dry_run}" == "true" ]]; then
        dry_run_flag="--dry-run=client"
    fi
    
    # Deploy environment service
    kubectl apply -f "${K8S_DIR}/environment/deployment.yaml" \
        -n "${namespace}" ${dry_run_flag}
    kubectl apply -f "${K8S_DIR}/environment/service.yaml" \
        -n "${namespace}" ${dry_run_flag}
    
    if [[ "${dry_run}" != "true" ]]; then
        log_info "Waiting for environment service to be ready..."
        kubectl rollout status deployment/environment-service \
            -n "${namespace}" --timeout=300s || {
            log_warn "Environment service rollout timed out. Check pod status."
        }
    fi
    
    log_info "Environment service deployed."
}

deploy_monitoring() {
    local namespace=$1
    local dry_run=$2
    
    log_info "Deploying monitoring configuration..."
    
    local dry_run_flag=""
    if [[ "${dry_run}" == "true" ]]; then
        dry_run_flag="--dry-run=client"
    fi
    
    # Check if monitoring directory exists
    if [[ -d "${K8S_DIR}/monitoring" ]]; then
        kubectl apply -f "${K8S_DIR}/monitoring/" \
            -n "${namespace}" ${dry_run_flag} || {
            log_warn "Monitoring deployment failed or not configured."
        }
    else
        log_warn "Monitoring directory not found. Skipping monitoring deployment."
    fi
}

delete_services() {
    local namespace=$1
    
    log_warn "Deleting all services..."
    
    # Delete trainer job if exists
    kubectl delete -f "${K8S_DIR}/trainer/job.yaml" \
        -n "${namespace}" --ignore-not-found=true
    
    # Delete environment service
    kubectl delete -f "${K8S_DIR}/environment/service.yaml" \
        -n "${namespace}" --ignore-not-found=true
    kubectl delete -f "${K8S_DIR}/environment/deployment.yaml" \
        -n "${namespace}" --ignore-not-found=true
    
    # Delete config
    kubectl delete -f "${K8S_DIR}/config/training-config.yaml" \
        -n "${namespace}" --ignore-not-found=true
    
    # Delete monitoring if exists
    if [[ -d "${K8S_DIR}/monitoring" ]]; then
        kubectl delete -f "${K8S_DIR}/monitoring/" \
            -n "${namespace}" --ignore-not-found=true
    fi
    
    log_info "All services deleted."
}

show_status() {
    local namespace=$1
    
    log_info "Current deployment status:"
    echo ""
    
    echo "=== Pods ==="
    kubectl get pods -n "${namespace}" -l "app in (environment-service,grpo-trainer)" 2>/dev/null || echo "No pods found"
    echo ""
    
    echo "=== Services ==="
    kubectl get services -n "${namespace}" -l "app=environment-service" 2>/dev/null || echo "No services found"
    echo ""
    
    echo "=== ConfigMaps ==="
    kubectl get configmaps -n "${namespace}" -l "app=grpo-trainer" 2>/dev/null || echo "No configmaps found"
    echo ""
}

main() {
    local namespace="default"
    local dry_run=false
    local delete=false
    local env_only=false
    local config_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -n|--namespace)
                namespace="$2"
                shift 2
                ;;
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            --delete)
                delete=true
                shift
                ;;
            --env-only)
                env_only=true
                shift
                ;;
            --config-only)
                config_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    
    if [[ "${delete}" == "true" ]]; then
        delete_services "${namespace}"
        exit 0
    fi
    
    create_namespace "${namespace}"
    
    if [[ "${config_only}" == "true" ]]; then
        deploy_config "${namespace}" "${dry_run}"
        exit 0
    fi
    
    if [[ "${env_only}" == "true" ]]; then
        deploy_environment_service "${namespace}" "${dry_run}"
        exit 0
    fi
    
    # Deploy all services
    deploy_config "${namespace}" "${dry_run}"
    deploy_environment_service "${namespace}" "${dry_run}"
    deploy_monitoring "${namespace}" "${dry_run}"
    
    if [[ "${dry_run}" != "true" ]]; then
        show_status "${namespace}"
        
        log_info "Services deployment complete!"
        log_info "Next steps:"
        log_info "  1. Verify environment service is running: kubectl get pods"
        log_info "  2. Run ./scripts/run-training.sh to start the training job"
    else
        log_info "Dry run complete. All manifests are valid."
    fi
}

main "$@"
