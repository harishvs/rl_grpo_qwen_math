#!/bin/bash
# Run veRL GRPO training on EKS
# Usage: ./scripts/verl/run-training.sh [--config CONFIG] [--follow]
#
# Options:
#   --config CONFIG   Config file from config/verl/ (default: qwen-1.5b.yaml)
#   --follow          Follow training logs after launch

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

CONFIG="qwen-1.5b.yaml"
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --follow) FOLLOW=true; shift ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

CONFIG_FILE="${PROJECT_ROOT}/config/verl/${CONFIG}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Config not found: $CONFIG_FILE${NC}"
    echo "Available configs:"
    ls "${PROJECT_ROOT}/config/verl/"
    exit 1
fi

# Parse YAML config
read_config() {
    python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    c = yaml.safe_load(f)
def flat(d, prefix=''):
    for k, v in d.items():
        key = f'{prefix}_{k}' if prefix else k
        if isinstance(v, dict):
            flat(v, key)
        else:
            print(f'CFG_{key}={v}')
flat(c)
"
}

eval "$(read_config)"

MODEL="${CFG_model}"
echo -e "${GREEN}=== veRL GRPO Training ===${NC}"
echo "Config: $CONFIG"
echo "Model:  $MODEL"
echo ""

if ! kubectl cluster-info &>/dev/null; then
    echo -e "${RED}Error: kubectl not configured or cluster unreachable${NC}"
    exit 1
fi

# Deploy manifests
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
        [[ "$STATUS" == "Running" ]] && echo -e "${GREEN}✓ Head pod running: $HEAD_POD${NC}" && break
    fi
    echo -n "."
    sleep 5
done
echo ""
[[ -z "$HEAD_POD" || "$STATUS" != "Running" ]] && echo -e "${RED}Error: Head pod not ready${NC}" && exit 1

# Wait for worker
echo -e "${YELLOW}Waiting for worker pod...${NC}"
for i in {1..60}; do
    WORKER_READY=$(kubectl get pods -l ray.io/cluster=verl-grpo,ray.io/node-type=worker -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    [[ "$WORKER_READY" == "Running" ]] && echo -e "${GREEN}✓ Worker pod running${NC}" && break
    echo -n "."
    sleep 5
done
echo ""

# Install veRL on all pods
echo -e "${YELLOW}Installing veRL on all pods...${NC}"
kubectl exec "$HEAD_POD" -- pip install verl==0.6.1 2>&1 | tail -1
WORKER_POD=$(kubectl get pods -l ray.io/cluster=verl-grpo,ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[[ -n "$WORKER_POD" ]] && kubectl exec "$WORKER_POD" -- pip install verl==0.6.1 2>&1 | tail -1
echo -e "${GREEN}✓ veRL installed on all nodes${NC}"
echo ""

# Build env vars string from config
ENV_VARS="VERL_MODEL=${CFG_model}"
ENV_VARS+=" VERL_EXPERIMENT=${CFG_experiment_name}"
ENV_VARS+=" VERL_TRAIN_BATCH_SIZE=${CFG_data_train_batch_size}"
ENV_VARS+=" VERL_MAX_PROMPT_LENGTH=${CFG_data_max_prompt_length}"
ENV_VARS+=" VERL_MAX_RESPONSE_LENGTH=${CFG_data_max_response_length}"
ENV_VARS+=" VERL_MINI_BATCH_SIZE=${CFG_actor_ppo_mini_batch_size}"
ENV_VARS+=" VERL_MICRO_BATCH_SIZE=${CFG_actor_ppo_micro_batch_size_per_gpu}"
ENV_VARS+=" VERL_LR=${CFG_actor_lr}"
ENV_VARS+=" VERL_ACTOR_PARAM_OFFLOAD=${CFG_actor_fsdp_param_offload}"
ENV_VARS+=" VERL_ACTOR_OPTIMIZER_OFFLOAD=${CFG_actor_fsdp_optimizer_offload}"
ENV_VARS+=" VERL_GRADIENT_CHECKPOINTING=${CFG_actor_gradient_checkpointing}"
ENV_VARS+=" VERL_GPU_MEM_UTIL=${CFG_rollout_gpu_memory_utilization}"
ENV_VARS+=" VERL_ROLLOUT_N=${CFG_rollout_n}"
ENV_VARS+=" VERL_ROLLOUT_LOG_PROB_MICRO_BATCH=${CFG_rollout_log_prob_micro_batch_size_per_gpu}"
ENV_VARS+=" VERL_ROLLOUT_TP=${CFG_rollout_tensor_model_parallel_size}"
ENV_VARS+=" VERL_REF_LOG_PROB_MICRO_BATCH=${CFG_ref_log_prob_micro_batch_size_per_gpu}"
ENV_VARS+=" VERL_REF_PARAM_OFFLOAD=${CFG_ref_fsdp_param_offload}"
ENV_VARS+=" VERL_N_GPUS=${CFG_trainer_n_gpus_per_node}"
ENV_VARS+=" VERL_NNODES=${CFG_trainer_nnodes}"
ENV_VARS+=" VERL_TOTAL_EPOCHS=${CFG_trainer_total_epochs}"
ENV_VARS+=" VERL_SAVE_FREQ=${CFG_trainer_save_freq}"
ENV_VARS+=" VERL_TEST_FREQ=${CFG_trainer_test_freq}"

# Launch training
echo -e "${GREEN}=== Launching training ===${NC}"
kubectl exec "$HEAD_POD" -- bash -c "nohup bash -c '${ENV_VARS} bash /scripts/run_grpo.sh' > /tmp/training.log 2>&1 &"
echo -e "${GREEN}✓ Training started in background${NC}"
echo ""

echo -e "${YELLOW}To follow logs:${NC}"
echo "  kubectl exec $HEAD_POD -- tail -f /tmp/training.log"

if [[ "$FOLLOW" == true ]]; then
    echo ""
    sleep 5
    kubectl exec "$HEAD_POD" -- tail -f /tmp/training.log
fi
