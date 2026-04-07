# Bug Report: RDMABuffer.read_into() fails between actors on same node (EKS/EFA)

## Summary

`RDMABuffer.read_into()` fails with a `delivery timeout` when called between two actors, even when both actors run on the **same node** and within the same Monarch host. The buffer creation succeeds (ibverbs backend confirmed), but the actual read operation hangs and times out.

According to the `monarch.rdma` docs, CPU tensors with the ibverbs backend should be supported out of the box; we’re following that usage pattern.[web:108]

## Environment

- torchmonarch 0.4.0
- PyTorch 2.10.0+cu128
- EKS 1.30, us-west-2
- 2× p4d.24xlarge (8× A100, 4× EFA per node)
- EFA device plugin: `aws-efa-k8s-device-plugin` v0.5.7 (Helm)
- Pods request `vpc.amazonaws.com/efa: 4`
- `/dev/infiniband/` visible inside pods with 4 uverbs devices
- `RDMABuffer` reports `"ibverbs"` backend (not TCP fallback)
- EFA kernel modules loaded on host (`efa`, `ib_uverbs`, `ib_core`)
- All 4 InfiniBand/EFA ports in ACTIVE state
- `Max locked memory: unlimited`
- HugePages: 10561 × 2 MiB

This reproduces with **server and client actors running on the same node** in the same EKS cluster.

## Minimum reproduction

Both actors are spawned from the same host using Monarch:

```python
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer
import torch, asyncio


class Server(Actor):
    def __init__(self):
        self.data = torch.arange(100, dtype=torch.float32)
        # CPU tensor; docs say this is the supported path
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
    # Single host, two actors: Server and Client
    procs = this_host().spawn_procs(per_host={"cpus": 1})
    server = procs.spawn("server", Server)
    client = procs.spawn("client", Client)

    data, buf = await server.get_buf.call_one()
    result = await client.read_from.call_one(data, buf)
    print(f"Result: {result}")


asyncio.run(main())
```

## Expected behavior

`read_into()` copies data from the server's buffer to the client's local tensor, so the script prints:

```text
Result: [0.0, 1.0, 2.0, 3.0, 4.0]
```

## Actual behavior

Instead, we see:

```text
undeliverable message error:
    error: broken link: failed to enqueue in MailboxClient when processing buffer:
    channel closed with reason Some("session unix:/.../anon_0-....XXXXX: delivery timeout")
```

From the logs, the client’s RDMA manager sends a `RequestBuffer` to the `IbvManagerActor`, but the server-side `IbvManagerActor` never replies, and the session times out.

## What we verified

- `RDMABuffer(tensor)` succeeds — backend is `"ibverbs"`, not TCP.
- `buf.backend` returns `"ibverbs"`.
- `/dev/infiniband/uverbs{0,1,2,3}` visible inside the pod.
- `ibv_devices` shows devices (EFA) from inside the pod.
- EFA kernel modules present on host (`efa`, `ib_uverbs`, `ib_core`).
- All 4 EFA/IB ports in ACTIVE state.
- `ulimit -l` is `unlimited`.
- HugePages configured (10561 × 2 MiB).
- Actors are on the **same node**; this is not a cross-node connectivity issue.

## Context

We’re trying to use RDMA for weight sync between an FSDP learner and a vLLM generator in a GRPO training setup. Without RDMA, we fall back to serialized `torch.save`/`torch.load` over Monarch RPC (~65 seconds for a 3 GB model), which is too slow for our use case.

## EKS cluster setup (Terraform)

The full infrastructure is defined in Terraform at [`terraform/`](https://github.com/harishvs/rl_grpo_qwen_math/tree/feat/monarch-grpo/terraform):

```
terraform/
├── environments/dev/main.tf    # EKS cluster, node groups, EFA, FSx, CloudWatch
├── modules/eks/                 # EKS cluster and OIDC
├── modules/node-groups/         # GPU (p4d.24xlarge) + CPU node groups
└── modules/vpc/                 # VPC with private/public subnets
```

Key EFA configuration in `modules/node-groups/main.tf`:
- Launch template with 4 EFA interfaces (`interface_type = "efa"`) on network cards 0-3
- EFA security group allowing all intra-node traffic

EFA device plugin in `environments/dev/main.tf`:
```hcl
resource "helm_release" "aws_efa_k8s_device_plugin" {
  name       = "aws-efa-k8s-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  version    = "0.5.7"
  namespace  = "kube-system"
}
```

Pods request EFA in the MonarchMesh manifest (`k8s/monarch/monarchmesh.yaml`):
```yaml
resources:
  limits:
    vpc.amazonaws.com/efa: "4"
    nvidia.com/gpu: "8"
```

## EFA diagnostics output

```
$ ibv_devinfo (inside pod)
hca_id: rdmap16s27
    transport:          unspecified (4)
    fw_ver:             0.0.0.0
    node_guid:          0000:0000:0000:0000    ← all zeros
    sys_image_guid:     0000:0000:0000:0000    ← all zeros
    vendor_id:          0x1d0f                  (Amazon)
    vendor_part_id:     61344                   (EFA)
    hw_ver:             0xEFA0
    port 1: PORT_ACTIVE, max_mtu=4096

$ fi_info -p efa (inside pod)
provider: efa
    fabric: efa-direct
    domain: rdmap16s27-rdm
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
(4 devices total, all active)

$ ls /dev/infiniband/
uverbs0  uverbs1  uverbs2  uverbs3
```

**Note**: All `node_guid` and `sys_image_guid` values are `0000:0000:0000:0000`. EFA devices on AWS report zero GUIDs since they use a different addressing scheme than traditional InfiniBand. This may be relevant if Monarch's ibverbs backend relies on GUIDs for connection establishment.

NCCL uses the 1 and works correctly:
```
NCCL INFO NET/Plugin: Loaded net plugin AWS Libfabric (v8)
NCCL INFO Successfully loaded external plugin aws-ofi
```

## Questions

1. Is there any additional configuration required for `RDMABuffer.read_into()` to work on EKS with EFA and the ibverbs backend?
2. Does `read_into()` require specific Kubernetes security context or capabilities beyond `privileged: true` and `IPC_LOCK` (for locked memory)?
3. Are there any known issues or limitations with Monarch’s ibverbs RDMA on AWS EFA (p4d.24xlarge, multiple EFA interfaces per node)?
4. Is there a way to enable more detailed logging for `IbvManagerActor` / RDMA negotiation to help debug why the `RequestBuffer` never completes?

