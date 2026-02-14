#!/bin/bash
# Run Training Script
# Launches the GRPO training job on EKS
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
BLUE='\033[0;34m'
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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Launch GRPO training job on EKS cluster.

OPTIONS:
    -h, --help              Show this help message
    -n, --namespace NS      Kubernetes namespace (default: default)
    -d, --dry-run           Run kubectl apply with --dry-run=client
    -f, --follow            Follow job logs after starting
    -w, --wait              Wait for job completion
    --delete                Delete existing training job
    --restart               Delete existing job and start new one
    --status                Show current job status

EXAMPLES:
    $(basename "$0")                    # Start training job
    $(basename "$0") --follow           # Start and follow logs
    $(basename "$0") --restart          # Restart training job
    $(basename "$0") --status           # Check job status
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
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

check_environment_service() {
    local namespace=$1
    
    log_info "Checking environment service..."
    
    # Check if environment service is running
    local ready_pods
    ready_pods=$(kubectl get pods -n "${namespace}" \
        -l "app=environment-service" \
        -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    
    if [[ -z "${ready_pods}" ]] || [[ "${ready_pods}" != *"True"* ]]; then
        log_error "Environment service is not ready."
        log_error "Please run ./scripts/deploy-services.sh first."
        exit 1
    fi
    
    log_info "Environment service is ready."
}

check_config() {
    local namespace=$1
    
    log_info "Checking training configuration..."
    
    # Check if training config exists
    if ! kubectl get configmap training-config -n "${namespace}" &> /dev/null; then
        log_error "Training config not found."
        log_error "Please run ./scripts/deploy-services.sh first."
        exit 1
    fi
    
    log_info "Training configuration found."
}

delete_job() {
    local namespace=$1
    
    log_info "Deleting existing training job..."
    
    kubectl delete -f "${K8S_DIR}/trainer/job.yaml" \
        -n "${namespace}" --ignore-not-found=true
    
    # Wait for pod termination
    log_info "Waiting for pods to terminate..."
    kubectl wait --for=delete pod \
        -l "app=grpo-trainer" \
        -n "${namespace}" \
        --timeout=120s 2>/dev/null || true
    
    log_info "Training job deleted."
}

start_job() {
    local namespace=$1
    local dry_run=$2
    
    log_info "Starting training job..."
    
    local dry_run_flag=""
    if [[ "${dry_run}" == "true" ]]; then
        dry_run_flag="--dry-run=client"
    fi
    
    kubectl apply -f "${K8S_DIR}/trainer/job.yaml" \
        -n "${namespace}" ${dry_run_flag}
    
    if [[ "${dry_run}" != "true" ]]; then
        log_info "Training job submitted."
        
        # Wait for pod to be created
        log_info "Waiting for trainer pod to be scheduled..."
        sleep 5
        
        # Show initial status
        kubectl get pods -n "${namespace}" -l "app=grpo-trainer"
    fi
}

follow_logs() {
    local namespace=$1
    
    log_info "Following training logs..."
    log_info "Press Ctrl+C to stop following (job will continue running)"
    echo ""
    
    # Wait for pod to be running
    local max_wait=300
    local waited=0
    while [[ ${waited} -lt ${max_wait} ]]; do
        local phase
        phase=$(kubectl get pods -n "${namespace}" \
            -l "app=grpo-trainer" \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
        
        if [[ "${phase}" == "Running" ]]; then
            break
        elif [[ "${phase}" == "Succeeded" ]] || [[ "${phase}" == "Failed" ]]; then
            log_info "Job already completed with status: ${phase}"
            kubectl logs -n "${namespace}" -l "app=grpo-trainer" --tail=100
            return
        fi
        
        log_debug "Waiting for pod to start... (${phase:-Pending})"
        sleep 5
        waited=$((waited + 5))
    done
    
    if [[ ${waited} -ge ${max_wait} ]]; then
        log_warn "Timeout waiting for pod to start. Check pod status:"
        kubectl describe pods -n "${namespace}" -l "app=grpo-trainer"
        return
    fi
    
    # Follow logs
    kubectl logs -n "${namespace}" -l "app=grpo-trainer" -f
}

wait_for_completion() {
    local namespace=$1
    
    log_info "Waiting for training job to complete..."
    
    kubectl wait --for=condition=complete job/grpo-trainer \
        -n "${namespace}" \
        --timeout=86400s || {
        local status
        status=$(kubectl get job grpo-trainer -n "${namespace}" \
            -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown")
        
        if [[ "${status}" == "Failed" ]]; then
            log_error "Training job failed."
            kubectl logs -n "${namespace}" -l "app=grpo-trainer" --tail=100
            exit 1
        fi
    }
    
    log_info "Training job completed successfully!"
}

show_status() {
    local namespace=$1
    
    echo ""
    echo "=== Training Job Status ==="
    kubectl get job grpo-trainer -n "${namespace}" 2>/dev/null || echo "No training job found"
    echo ""
    
    echo "=== Trainer Pods ==="
    kubectl get pods -n "${namespace}" -l "app=grpo-trainer" 2>/dev/null || echo "No trainer pods found"
    echo ""
    
    # Show recent events
    echo "=== Recent Events ==="
    kubectl get events -n "${namespace}" \
        --field-selector involvedObject.name=grpo-trainer \
        --sort-by='.lastTimestamp' 2>/dev/null | tail -10 || echo "No events found"
    echo ""
    
    # Show resource usage if pod is running
    local pod_name
    pod_name=$(kubectl get pods -n "${namespace}" \
        -l "app=grpo-trainer" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "${pod_name}" ]]; then
        echo "=== Resource Usage ==="
        kubectl top pod "${pod_name}" -n "${namespace}" 2>/dev/null || echo "Metrics not available"
        echo ""
    fi
}

main() {
    local namespace="default"
    local dry_run=false
    local follow=false
    local wait=false
    local delete=false
    local restart=false
    local status=false
    
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
            -f|--follow)
                follow=true
                shift
                ;;
            -w|--wait)
                wait=true
                shift
                ;;
            --delete)
                delete=true
                shift
                ;;
            --restart)
                restart=true
                shift
                ;;
            --status)
                status=true
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
    
    if [[ "${status}" == "true" ]]; then
        show_status "${namespace}"
        exit 0
    fi
    
    if [[ "${delete}" == "true" ]]; then
        delete_job "${namespace}"
        exit 0
    fi
    
    if [[ "${restart}" == "true" ]]; then
        delete_job "${namespace}"
    fi
    
    # Pre-flight checks
    check_environment_service "${namespace}"
    check_config "${namespace}"
    
    # Start the job
    start_job "${namespace}" "${dry_run}"
    
    if [[ "${dry_run}" == "true" ]]; then
        log_info "Dry run complete. Job manifest is valid."
        exit 0
    fi
    
    if [[ "${follow}" == "true" ]]; then
        follow_logs "${namespace}"
    elif [[ "${wait}" == "true" ]]; then
        wait_for_completion "${namespace}"
    else
        log_info "Training job started!"
        log_info "Monitor progress with:"
        log_info "  kubectl logs -f -l app=grpo-trainer"
        log_info "  ./scripts/run-training.sh --status"
    fi
}

main "$@"
