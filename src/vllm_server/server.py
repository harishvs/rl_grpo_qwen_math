"""vLLM inference server with HTTP weight sync from trainer.

Runs as a separate pod on 1 GPU. Receives updated weights from trainer rank 0
via HTTP POST (binary tensor), then serves generation requests via HTTP.
"""

import io
import logging
import os
import threading
import time
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


class GenerateRequest(BaseModel):
    prompts: List[str]
    group_size: int = 8
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0


class CompletionResult(BaseModel):
    prompt: str
    completion: str
    token_ids: List[int]
    log_probs: List[float]


class GenerateResponse(BaseModel):
    completions: List[CompletionResult]


# Global state
_engine: Optional["VLLMEngine"] = None
_pending_state_dict = None


def _apply_state_dict(model):
    """Module-level function for apply_model (must be picklable)."""
    global _pending_state_dict
    if _pending_state_dict is None:
        return 0
    model_params = dict(model.named_parameters())
    updated = 0
    for name, new_param in _pending_state_dict.items():
        if name in model_params:
            model_params[name].data.copy_(new_param)
            updated += 1
    return updated


class VLLMEngine:
    """Wraps vLLM LLM with HTTP weight update capability."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._llm: Optional[LLM] = None
        self._lock = threading.Lock()
        self._sync_count = 0
        self._last_sync_duration_ms: Optional[float] = None

    def setup(self):
        logger.info(f"Loading vLLM model: {self.model_name}")
        self._llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.80,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
        )
        logger.info("vLLM model loaded")

    def update_weights(self, state_dict_bytes: bytes):
        """Update model weights from serialized state dict via apply_model."""
        t0 = time.time()
        buf = io.BytesIO(state_dict_bytes)
        # Load to CPU to avoid GPU OOM (model + KV cache already use most GPU memory)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)

        with self._lock:
            global _pending_state_dict
            _pending_state_dict = state_dict
            results = self._llm.apply_model(_apply_state_dict)
            updated = results[0] if results else 0
            _pending_state_dict = None

        elapsed_ms = (time.time() - t0) * 1000
        self._sync_count += 1
        self._last_sync_duration_ms = elapsed_ms
        logger.info(f"Weight sync #{self._sync_count}: {updated} params in {elapsed_ms:.1f}ms")

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        sampling_params = SamplingParams(
            n=req.group_size,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            logprobs=1,
        )

        with self._lock:
            outputs = self._llm.generate(req.prompts, sampling_params)

        completions = []
        for output in outputs:
            for seq in output.outputs:
                token_ids = list(seq.token_ids)
                log_probs = []
                if seq.logprobs:
                    for lp_dict in seq.logprobs:
                        if lp_dict:
                            chosen = next(iter(lp_dict.values()))
                            log_probs.append(chosen.logprob)
                        else:
                            log_probs.append(0.0)

                completions.append(CompletionResult(
                    prompt=output.prompt,
                    completion=seq.text,
                    token_ids=token_ids,
                    log_probs=log_probs,
                ))

        return GenerateResponse(completions=completions)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    return _engine.generate(req)


@app.post("/update_weights")
async def update_weights(request: Request):
    body = await request.body()
    _engine.update_weights(body)
    return {"status": "ok", "sync_count": _engine._sync_count, "duration_ms": _engine._last_sync_duration_ms}


@app.get("/health")
async def health():
    return {"status": "ok", "syncs": _engine._sync_count if _engine else 0}


def main():
    global _engine
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
    port = int(os.environ.get("VLLM_SERVER_PORT", "8000"))

    _engine = VLLMEngine(model_name=model_name)
    _engine.setup()

    logger.info(f"Starting vLLM server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
