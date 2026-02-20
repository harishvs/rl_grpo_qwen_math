#!/bin/bash
# Start the GRPO training job on EKS
# Usage: ./scripts/start-training.sh [--rebuild]
#
# Options:
#   --rebuild    Rebuild and push the trainer image before starting
#
# This script:
#   1. Optionally rebuilds the trainer image for linux/amd64
#   2. Deletes any existing training job
#   3. Starts a new training job
#   4. Monitors the job until it starts running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

REBUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--rebuild]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== GRPO Training Job Launcher ===${NC}"
echo ""

# Verify kubectl is configured
if ! kubectl cluster-info &>/dev/null; then
    echo -e "${RED}Error: kubectl is not configured or cluster is unreachable${NC}"
    exit 1
fi

# Optionally rebuild the image
if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Rebuilding trainer image...${NC}"
    "${SCRIPT_DIR}/build-and-push-images.sh" trainer
    echo ""
fi

# Delete existing job if present
echo -e "${YELLOW}Cleaning up any existing training job...${NC}"
kubectl delete job grpo-trainer --ignore-not-found
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Start the new job
echo -e "${YELLOW}Starting training job...${NC}"
kubectl apply -f "${PROJECT_ROOT}/k8s/custom/trainer/job.yaml"
echo -e "${GREEN}✓ Job created${NC}"
echo ""

# Wait for pod to be scheduled and show status
echo -e "${YELLOW}Waiting for pod to start...${NC}"
for i in {1..30}; do
    POD_STATUS=$(kubectl get pods -l app=grpo-trainer -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
    CONTAINER_STATUS=$(kubectl get pods -l app=grpo-trainer -o jsonpath='{.items[0].status.containerStatuses[0].state}' 2>/dev/null || echo "{}")
    
    if [[ "$POD_STATUS" == "Running" ]]; then
        echo -e "${GREEN}✓ Training pod is running${NC}"
        break
    elif [[ "$CONTAINER_STATUS" == *"ImagePullBackOff"* ]] || [[ "$CONTAINER_STATUS" == *"ErrImagePull"* ]]; then
        echo -e "${RED}Error: Image pull failed${NC}"
        echo -e "${YELLOW}The image may not exist or may be built for the wrong platform.${NC}"
        echo -e "${YELLOW}Run with --rebuild flag to rebuild for linux/amd64:${NC}"
        echo -e "  $0 --rebuild"
        echo ""
        kubectl describe pod -l app=grpo-trainer | tail -20
        exit 1
    elif [[ "$POD_STATUS" == "Failed" ]]; then
        echo -e "${RED}Error: Pod failed to start${NC}"
        kubectl describe pod -l app=grpo-trainer | tail -20
        exit 1
    fi
    
    echo -n "."
    sleep 2
done
echo ""

# Show final status
echo -e "${GREEN}=== Training Job Status ===${NC}"
kubectl get pods -l app=grpo-trainer
echo ""

echo -e "${YELLOW}To follow logs:${NC}"
echo "  kubectl logs -f -l app=grpo-trainer"
echo ""
echo -e "${YELLOW}To check job status:${NC}"
echo "  kubectl get jobs grpo-trainer"
