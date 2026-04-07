"""ParameterServerActor — holds a flat CPU buffer with RDMA handle.

Lives in the worker process (not in FSDP rank child processes) so the
IbvManagerActor can see and negotiate the RDMA buffer. Learner ranks
push their local FSDP shards here via messages. Generator reads via RDMA.
"""
import torch
from monarch.actor import Actor, endpoint
from monarch.rdma import RDMABuffer


class ParameterServerActor(Actor):
    """Flat CPU buffer with RDMA exposure for cross-node weight sync.

    Each learner rank copies its local FSDP shard into this buffer.
    The generator reads the full buffer via one RDMA read_into call.
    """

    def __init__(self, total_bytes: int):
        self.flat = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        self.rdma_buf = RDMABuffer(self.flat)
        self.layout = {}  # name -> (offset, num_bytes, dtype_str, full_shape)

    @endpoint
    async def set_layout(self, layout: dict) -> None:
        """Set the parameter layout mapping.

        Args:
            layout: Dict of {name: (offset, num_bytes, dtype_str, full_shape)}
        """
        self.layout = layout

    @endpoint
    async def push_shard(self, offset: int, shard_bytes: bytes) -> None:
        """Receive a shard from a learner rank and copy into flat buffer.

        Args:
            offset: Byte offset in the flat buffer
            shard_bytes: Raw bytes of the shard tensor
        """
        src = torch.frombuffer(bytearray(shard_bytes), dtype=torch.uint8)
        self.flat[offset:offset + src.numel()].copy_(src)

    @endpoint
    async def get_rdma_handle(self) -> dict:
        """Return RDMA buffer handle and layout for the generator."""
        return {
            "buffer": (self.flat, self.rdma_buf),
            "layout": self.layout,
        }

    @endpoint
    async def get_stats(self) -> dict:
        return {
            "buffer_size_mb": self.flat.numel() / 1e6,
            "num_params": len(self.layout),
        }
