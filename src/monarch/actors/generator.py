"""GeneratorActor -- vLLM-powered text generation on dedicated GPUs.

Runs vLLM as an in-process engine (no HTTP). Generates G completions per
prompt with per-token log probabilities. Handles weight updates when the
Learner pushes new weights after training steps.
"""
import io
from typing import Dict, List, Optional

import torch
from monarch.actor import Actor, endpoint


class GeneratorActor(Actor):
    """Generates completions using vLLM on dedicated GPUs.

    Each instance owns its GPU allocation via the Monarch proc mesh.
    vLLM runs in-process -- no HTTP serialization overhead.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_tokens: int = 1024,
        trust_remote_code: bool = True,
    ):
        import os
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.policy_version = 0

        self.engine = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype="bfloat16",
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.SamplingParams = SamplingParams

    @endpoint
    async def generate(
        self,
        prompts: List[str],
        group_size: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> dict:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of formatted prompt strings
            group_size: Number of completions per prompt (G)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Dict with 'completions', 'log_probs', 'token_ids', 'prompts_expanded'
        """
        sampling_params = self.SamplingParams(
            n=group_size,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1,
        )

        outputs = self.engine.generate(prompts, sampling_params)

        completions = []
        log_probs = []
        token_ids = []
        prompts_expanded = []

        for output in outputs:
            prompt_text = output.prompt
            for completion_output in output.outputs:
                completions.append(completion_output.text)
                prompts_expanded.append(prompt_text)

                # Extract per-token log probs
                if completion_output.logprobs:
                    token_logps = [
                        list(lp.values())[0].logprob
                        for lp in completion_output.logprobs
                    ]
                else:
                    token_logps = []
                log_probs.append(token_logps)
                token_ids.append(list(completion_output.token_ids))

        return {
            "completions": completions,
            "log_probs": log_probs,
            "token_ids": token_ids,
            "prompts_expanded": prompts_expanded,
            "policy_version": self.policy_version,
        }

    @endpoint
    async def set_weight_handles(self, rdma_info: dict) -> None:
        """Store RDMA buffer handle and param metadata from the learner.

        Args:
            rdma_info: Dict with 'buffer' (tensor, RDMABuffer) and 'meta' [(name, shape, dtype)]
        """
        self._rdma_tensor, self._rdma_buf = rdma_info["buffer"]
        self._rdma_meta = rdma_info["meta"]

    @endpoint
    async def sync_weights_rdma(self, version: int) -> None:
        """Pull latest weights from learner via single RDMA read.

        One 3GB read over EFA, then slice into individual params.
        """
        # Read the entire flat buffer in one RDMA call
        local_flat = torch.empty_like(self._rdma_tensor)
        await self._rdma_buf.read_into(local_flat.view(torch.uint8).flatten())

        # Slice back into individual params
        local_state_dict = {}
        offset = 0
        for name, shape, dtype_str in self._rdma_meta:
            numel = 1
            for s in shape:
                numel *= s
            param = local_flat[offset:offset + numel].reshape(shape)
            local_state_dict[name] = param
            offset += numel

        def _load(model):
            model.load_state_dict(local_state_dict, strict=False)

        self.engine.apply_model(_load)
        self.policy_version = version

    @endpoint
    async def update_weights(self, state_dict_bytes: bytes, version: int) -> None:
        """Update model weights from serialized state dict (fallback, non-RDMA)."""
        buffer = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        clean_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, 'full_tensor'):
                clean_state_dict[k] = v.full_tensor().cpu()
            elif hasattr(v, '_local_tensor'):
                clean_state_dict[k] = v._local_tensor.cpu()
            else:
                clean_state_dict[k] = v.cpu() if hasattr(v, 'cpu') else v

        def _load(model):
            model.load_state_dict(clean_state_dict, strict=False)

        self.engine.apply_model(_load)
        self.policy_version = version

    @endpoint
    async def get_stats(self) -> dict:
        """Return generator statistics."""
        return {
            "model_name": self.model_name,
            "policy_version": self.policy_version,
        }
