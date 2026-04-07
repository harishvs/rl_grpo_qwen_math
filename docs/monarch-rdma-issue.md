# Bug Report: RDMABuffer.read_into() fails between actors on same node (EKS/EFA)

## Summary

`RDMABuffer.read_into()` fails with `delivery timeout` when called between two actors, even on the same node. The buffer creation succeeds (ibverbs backend confirmed), but the actual read operation hangs and times out.

## Environment

- torchmonarch 0.4.0
- PyTorch 2.10.0+cu128
- EKS 1.30, us-west-2
- 2x p4d.24xlarge with 4x EFA interfaces each
- EFA device plugin: aws-efa-k8s-device-plugin v0.5.7 (Helm chart)
- Pods request `vpc.amazonaws.com/efa: 4`
- `/dev/infiniband/` visible inside pods with 4 uverbs devices
- `RDMABuffer` reports `ibverbs` backend (not TCP fallback)

## Minimum reproduction

```python
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer
import torch, asyncio

class Server(Actor):
    def __init__(self):
        self.data = torch.arange(100, dtype=torch.float32)
        self.buf = RDMABuffer(self.data.view(torch.uint8).flatten())

    @endpoint
    async def get_buf(self):
        return (self.data, self.buf)

class Client(Actor):
    @endpoint
    async def read_from(self, remote_data, remote_buf):
        local = torch.empty_like(remote_data)
        await remote_buf.read_into(local.view(torch.uint8).flatten())
        return local[:5].tolist()

async def main():
    procs = this_host().spawn_procs(per_host={"cpus": 1})
    server = procs.spawn("server", Server)
    client = procs.spawn("client", Client)
    data, buf = await server.get_buf.call_one()
    result = await client.read_from.call_one(data, buf)
    print(f"Result: {result}")

asyncio.run(main())
```

## Expected behavior

`read_into()` copies data from the server's buffer to the client's local tensor. Output: `Result: [0.0, 1.0, 2.0, 3.0, 4.0]`

## Actual behavior

```
undeliverable message error:
    error: broken link: failed to enqueue in MailboxClient when processing buffer:
    channel closed with reason Some("session unix:/.../anon_0-....XXXXX: delivery timeout")
```

The `IbvManagerActor` on the server side never responds to the `RequestBuffer` message from the client side.

## What we verified

- `RDMABuffer(tensor)` succeeds — ibverbs backend, not TCP fallback
- `buf.backend` returns `"ibverbs"`
- `/dev/infiniband/uverbs{0,1,2,3}` visible in pods
- `ibv_devices` shows devices (though no GUID)
- EFA kernel module loaded on host (`efa`, `ib_uverbs`, `ib_core` all active)
- All 4 infiniband ports in ACTIVE state
- `Max locked memory: unlimited` in process limits
- HugePages configured: 10561 × 2MB

## Context

We're trying to use RDMA for weight sync between FSDP learner and vLLM generator in a GRPO training setup. This is blocking us from achieving fast weight transfer (~sub-second) — currently falling back to serialized torch.save/load over Monarch RPC (~65s for 3GB model).

## Questions

1. Is there additional configuration needed for `RDMABuffer.read_into()` to work on EKS with EFA?
2. Does `read_into()` require specific security context or capabilities beyond `privileged: true`?
3. Are there known issues with ibverbs RDMA on AWS EFA with p4d.24xlarge?
