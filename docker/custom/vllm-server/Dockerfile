# vLLM Inference Server Dockerfile
# Separate pod for generation, receives weights from trainer via NCCL/EFA

FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

COPY docker/custom/vllm-server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Remove incompatible flash_attn from NGC base — vLLM uses its own attention backends
RUN pip uninstall -y flash-attn 2>/dev/null; true

COPY src/custom/ /app/src/custom/

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.custom.vllm_server.server"]
