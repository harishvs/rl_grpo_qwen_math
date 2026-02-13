# GRPO Trainer Dockerfile
# Uses NVIDIA NGC PyTorch container as base for GPU training

FROM nvcr.io/nvidia/pytorch:25.01-py3

LABEL maintainer="ML Platform Team"
LABEL description="GRPO Trainer for RL-based LLM training"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-trainer.txt /app/requirements-trainer.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-trainer.txt

# Copy source code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app

# Default environment variables
ENV MODEL_NAME="Qwen/Qwen2.5-1.5B"
ENV ENVIRONMENT_SERVICE_URL="http://environment-service:8080"
ENV CHECKPOINT_DIR="/checkpoints"

# Create checkpoint directory
RUN mkdir -p /checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Entrypoint
ENTRYPOINT ["python", "-m", "src.trainer.main"]
