#!/bin/bash
# Entrypoint script for GRPO trainer with distributed training support

set -e

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting distributed training with torchrun on $NUM_GPUS GPUs"
    exec torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        -m src.trainer.main "$@"
else
    echo "Starting single GPU training"
    exec python -m src.trainer.main "$@"
fi
