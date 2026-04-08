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
        # Make all GPUs visible for tensor parallelism — Monarch may have
        # restricted CUDA_VISIBLE_DEVICES to 1 GPU for this process
        if tensor_parallel_size > 1:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
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
    async def init_weight_sync_pg(self, store_host: str, store_port: int) -> str:
        """Join a gloo process group for direct tensor recv from learner."""
        import torch.distributed as dist
        from torch.distributed import TCPStore, ProcessGroupGloo
        store = TCPStore(store_host, store_port, 2, False, timeout=dist.default_pg_timeout)
        self._sync_pg = ProcessGroupGloo(store, 1, 2)
        return f"generator joined weight_sync gloo pg on {store_host}:{store_port}"

    @endpoint
    async def recv_weights_direct(self, meta: dict, version: int) -> float:
        """Receive gathered weight tensor from learner via gloo, load into vLLM.

        No serialization — raw tensor recv + reshape into state_dict.
        """
        import torch.distributed as dist
        import time
        t0 = time.time()

        shard_size = meta["shard_size"]
        world_size = meta["world_size"]
        dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
        total_numel = shard_size * world_size

        # Receive CPU tensor from learner rank 0 — use PG's own recv method
        recv_buf = torch.empty(total_numel, dtype=dtype)
        self._sync_pg.recv([recv_buf], 0, 0).wait()

        # Reconstruct state_dict
        state_dict = {}
        for name, full_shape, shard_numel, shard_offset in meta["meta"]:
            param_shards = []
            for r in range(world_size):
                start = r * shard_size + shard_offset
                param_shards.append(recv_buf[start:start + shard_numel])
            full_param = torch.cat(param_shards)
            full_numel = 1
            for s in full_shape:
                full_numel *= s
            state_dict[name] = full_param[:full_numel].reshape(full_shape)

        def _load(model):
            model.load_state_dict(state_dict, strict=False)

        self.engine.apply_model(_load)
        self.policy_version = version
        return time.time() - t0

    @endpoint
    async def set_ps_handle(self, rdma_info: dict) -> None:
        """Store param server RDMA handle and layout.

        Args:
            rdma_info: Dict with 'buffer' (tensor, RDMABuffer) and 'layout'
        """
        self._ps_flat, self._ps_rdma_buf = rdma_info["buffer"]
        self._ps_layout = rdma_info["layout"]

    @endpoint
    async def sync_weights_from_ps(self, version: int) -> None:
        """Read full model weights from parameter server via one RDMA read.

        One ~3GB read over EFA, then slice into individual params using layout.
        """
        # One RDMA read for entire flat buffer
        local_flat = torch.empty_like(self._ps_flat)
        await self._ps_rdma_buf.read_into(local_flat)

        # Reconstruct state_dict from layout
        state_dict = {}
        for name, (offset, nbytes, dtype_str, shape) in self._ps_layout.items():
            if "__rank" in name:
                continue  # skip per-rank entries, use full param entries only
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            param_bytes = local_flat[offset:offset + nbytes]
            state_dict[name] = param_bytes.view(dtype).reshape(shape)

        def _load(model):
            model.load_state_dict(state_dict, strict=False)

        self.engine.apply_model(_load)
        self.policy_version = version

    @endpoint
    async def set_shard_handles(self, shard_list: list) -> None:
        """Store RDMA shard handles from all learner ranks.

        Args:
            shard_list: List of 8 dicts, each with {rank, buffer: (tensor, RDMABuffer), meta}
        """
        # Sort by rank to ensure correct shard ordering
        self._shards = sorted(shard_list, key=lambda s: s["rank"])
        self._shard_meta = self._shards[0]["meta"]  # meta is same across ranks

    @endpoint
    async def sync_weights_rdma(self, version: int) -> None:
        """Pull shards from all 8 learner ranks via RDMA, reconstruct full tensors.

        8 parallel RDMA reads of ~444 MB each, then concatenate per-param.
        No collectives on the learner side.
        """
        import asyncio

        # Read all 8 shards in parallel
        local_flats = []
        for shard_info in self._shards:
            remote_tensor, rdma_buf = shard_info["buffer"]
            local = torch.empty_like(remote_tensor)
            await rdma_buf.read_into(local.view(torch.uint8).flatten())
            local_flats.append(local)

        # Reconstruct full tensors by concatenating shards per param
        state_dict = {}
        offsets = [0] * len(self._shards)  # track position in each rank's flat buffer

        for param_name, full_shape, local_shape in self._shard_meta:
            local_numel = 1
            for s in local_shape:
                local_numel *= s

            # Gather this param's shard from each rank
            param_shards = []
            for r, flat in enumerate(local_flats):
                shard = flat[offsets[r]:offsets[r] + local_numel]
                param_shards.append(shard)
                offsets[r] += local_numel

            # Concatenate shards → full param (FSDP shards along flattened dim)
            full_flat = torch.cat(param_shards)
            full_numel = 1
            for s in full_shape:
                full_numel *= s
            state_dict[param_name] = full_flat[:full_numel].reshape(full_shape)

        def _load(model):
            model.load_state_dict(state_dict, strict=False)

        self.engine.apply_model(_load)
        self.policy_version = version

    @endpoint
    async def update_weights(self, state_dict_bytes: bytes, version: int) -> None:
        """Update model weights from serialized state dict.

        Saves to HF-format tmp dir, then uses vLLM's reload_weights
        which works with TP>1 (no closure pickling needed).
        """
        import os, tempfile, shutil
        from transformers import AutoConfig

        buf = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)

        # Clean DTensors if any
        clean = {}
        for k, v in state_dict.items():
            if hasattr(v, 'full_tensor'):
                clean[k] = v.full_tensor().cpu()
            elif hasattr(v, '_local_tensor'):
                clean[k] = v._local_tensor.cpu()
            else:
                clean[k] = v.cpu() if hasattr(v, 'cpu') else v

        # Save as safetensors in a tmp dir (HF checkpoint format)
        tmp_dir = "/tmp/weight_update"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save config so vLLM can find it
        config = AutoConfig.from_pretrained(self.model_name)
        config.save_pretrained(tmp_dir)

        # Save weights
        from safetensors.torch import save_file
        save_file(clean, os.path.join(tmp_dir, "model.safetensors"))

        # vLLM reload_weights: works with TP>1, reloads from disk
        self.engine.collective_rpc("reload_weights", kwargs={
            "weights_path": tmp_dir,
        })

        self.policy_version = version

    @endpoint
    async def get_stats(self) -> dict:
        """Return generator statistics."""
        return {
            "model_name": self.model_name,
            "policy_version": self.policy_version,
        }
