#!/bin/bash
# Entrypoint script for GRPO trainer with distributed training support

set -e

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Multi-node: NNODES and JOB_COMPLETION_INDEX set by k8s indexed job
NNODES=${NNODES:-1}
NODE_RANK=${JOB_COMPLETION_INDEX:-0}

if [ "$NNODES" -gt 1 ]; then
    # Non-master nodes wait until master's rendezvous port is reachable
    if [ "$NODE_RANK" -gt 0 ]; then
        echo "Node $NODE_RANK: waiting for master rendezvous at $MASTER_ADDR:${MASTER_PORT:-29400}..."
        for i in $(seq 1 120); do
            if python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('$MASTER_ADDR', ${MASTER_PORT:-29400}))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
                echo "Master rendezvous reachable after ${i}s"
                # Extra 5s buffer to let master fully initialize
                sleep 5
                break
            fi
            sleep 1
        done
    fi
    echo "Starting multi-node training: node $NODE_RANK of $NNODES, master=$MASTER_ADDR:$MASTER_PORT"
    exec torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NUM_GPUS \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=${MASTER_PORT:-29400} \
        --tee=3 \
        --local_ranks_filter=0 \
        -m src.custom.trainer.main "$@"
elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting distributed training with torchrun on $NUM_GPUS GPUs"
    exec torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        -m src.custom.trainer.main "$@"
else
    echo "Starting single GPU training"
    exec python -m src.custom.trainer.main "$@"
fi
