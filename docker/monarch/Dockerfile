FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

# System dependencies + AWS EFA installer
# EFA installer needs pciutils, environment-modules, tcl -- must keep apt cache until after install
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl pciutils environment-modules tcl && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && \
    tar -xf aws-efa-installer-latest.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/aws-efa-installer* && \
    rm -rf /var/lib/apt/lists/*

# Install torchmonarch first (may upgrade PyTorch)
RUN pip install --no-cache-dir torchmonarch==0.4.0 kubernetes>=28.0.0

# Reinstall vLLM against the current PyTorch version
# (torchmonarch may have upgraded PyTorch, breaking pre-compiled extensions)
RUN pip install --no-cache-dir vllm --force-reinstall

# Remaining Python dependencies
RUN pip install --no-cache-dir \
    datasets>=2.14.0 \
    transformers>=4.40.0 \
    accelerate>=0.25.0 \
    boto3>=1.34.0 \
    pyyaml>=6.0 \
    prometheus_client>=0.20.0 \
    "numpy<2"

# Copy source (shared + monarch)
COPY src/shared/ /app/src/shared/
COPY src/monarch/ /app/src/monarch/

ENV PYTHONPATH=/app
ENV NCCL_DEBUG=WARN
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_PROVIDER=efa

RUN mkdir -p /checkpoints /data

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import monarch; print(torch.cuda.is_available())" || exit 1
