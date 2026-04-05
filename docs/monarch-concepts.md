# Monarch Concepts Guide

> **Monarch** is Meta's distributed programming framework for PyTorch, built on **actor messaging**.
> Think of it as "Erlang meets PyTorch" -- isolated actors communicate via async messages, spread across GPUs and hosts.

**Version**: 0.4.0 (experimental, Linux-only)
**Install**: `pip install torchmonarch`
**Repo**: `github.com/meta-pytorch/monarch` (BSD-3)

---

## 1. The Big Picture

Traditional distributed PyTorch (torchrun, DDP, FSDP) gives you one program replicated across GPUs.
Monarch gives you **different programs on different GPUs** that talk to each other via messages.

This matters for RL training where you need fundamentally different roles:
- **Generators** producing rollouts on some GPUs
- **Learners** updating the policy on other GPUs
- **Scorers** evaluating rewards
- All running concurrently, communicating asynchronously

```
Traditional (torchrun):     Monarch (actors):
  GPU 0: same code            GPU 0: Learner
  GPU 1: same code            GPU 1: Scorer
  GPU 2: same code            GPU 2: Generator
  GPU 3: same code            GPU 3: Generator
  (all synchronized)          (async, message-passing)
```

**Analogy**: torchrun is like a synchronized swim team -- everyone does the same moves at the same time.
Monarch is like a restaurant kitchen -- the chef, sous chef, and waiter each do different jobs,
coordinating via orders (messages) rather than everyone cooking the same dish.

---

## 2. Actors

An **Actor** is an isolated unit of computation with private state. Like a microservice:
- Has its own memory (no shared state with other actors)
- Processes messages one at a time (sequential, no locks needed)
- Communicates only via async messages
- Location-transparent (same API whether local or on another machine)

### Defining an Actor

```python
from monarch.actor import Actor, endpoint

class Scorer(Actor):
    def __init__(self, threshold: float):
        # Private state -- only this actor can access it
        self.threshold = threshold
        self.scores = []

    @endpoint
    async def score(self, value: float) -> bool:
        """Public API -- other actors can call this."""
        passed = value > self.threshold
        self.scores.append(value)
        return passed

    @endpoint
    async def get_stats(self) -> dict:
        return {"count": len(self.scores), "mean": sum(self.scores) / len(self.scores)}
```

Key points:
- Inherit from `Actor`
- Mark public methods with `@endpoint` -- these are the actor's API
- Endpoints can be `async` or sync
- `__init__` receives parameters passed at spawn time
- Private state is truly private -- no other actor can touch `self.scores`

### Actor Lifecycle

```
Creation --> Construction (__init__) --> Running (processing messages) --> Termination
```

Messages queue up in the actor's **mailbox**. Processed one at a time, in FIFO order.
No race conditions, no locks, no shared memory bugs.

**Analogy**: An actor is like a person who checks their inbox. They read one email at a time,
do the work, send a reply, then move to the next email. They never read two emails at once.

---

## 3. Process Meshes (ProcMesh)

A **ProcMesh** is a grid of processes, organized by named dimensions (like "hosts" and "gpus").
This is where actors live.

```python
from monarch.actor import this_host

# Get a handle to the current machine
host = this_host()

# Spawn 4 GPU processes on this host
procs = host.spawn_procs(per_host={"gpus": 4})
# Result: a 1x4 mesh (1 host, 4 gpus)

# With 2 hosts in a cluster:
# procs = hosts.spawn_procs(per_host={"gpus": 4})
# Result: a 2x4 mesh (2 hosts, 4 gpus each = 8 total processes)
```

**Analogy**: A ProcMesh is a seating chart. It says "these 8 GPUs across 2 machines
are reserved for this job." Actors get assigned seats on this chart.

---

## 4. Actor Meshes (Spawning Actors)

When you spawn an actor on a ProcMesh, you get **one actor instance per process**.

```python
# Spawn a Scorer on each of the 4 GPU processes
scorers = procs.spawn("scorers", Scorer, threshold=0.5)
# --> 4 Scorer instances, one per GPU
```

This returns an **ActorMesh** -- a collection of actors you can address together or individually.

### Slicing

Target specific actors in a mesh:

```python
first_two = scorers.slice(gpus=slice(0, 2))   # First 2 GPUs
just_one  = scorers.slice(gpus=3)              # GPU 3 only
```

**Analogy**: The ActorMesh is a team. You can give orders to the whole team,
or pull aside specific members.

---

## 5. Messaging Patterns

How actors talk to each other. This is the core of Monarch.

### call_one() -- Ask one actor, wait for answer
```python
result = await scorers.score.call_one(0.75)
# Sends to one actor, waits for and returns the result
```

### call() -- Ask everyone, collect all answers
```python
results = await scorers.score.call(0.75)
# Sends to ALL 4 scorers, returns a list of 4 results
```
**Use case**: When each actor processes its shard of data (like FSDP training step).

### broadcast() -- Tell everyone, don't wait
```python
scorers.score.broadcast(0.75)
# Sends to all actors, returns immediately (fire-and-forget)
```
**Use case**: Sending a "stop" signal or weight update notification.

### stream() -- Get results as they arrive
```python
async for result in scorers.score.stream(0.75):
    print(result)
# Results arrive in completion order, not rank order
```

### Summary Table

| Pattern | Targets | Waits? | Returns | When to use |
|---------|---------|--------|---------|-------------|
| `call_one()` | 1 actor | Yes | Single result | Point-to-point operations |
| `call()` | All actors | Yes | All results | Collective operations (FSDP step) |
| `broadcast()` | All actors | No | Nothing | Notifications, signals |
| `stream()` | All actors | Incremental | Results as ready | Progress monitoring |

---

## 6. RDMA Buffers (Fast Weight Transfer)

**RDMA** = Remote Direct Memory Access. Lets one actor read/write another actor's memory
directly -- no serialization, no copies through the network stack.

```python
from monarch.rdma import RDMABuffer

# Learner side: expose model weights as RDMA buffers
class Learner(Actor):
    @endpoint
    async def weights_handle(self) -> dict:
        return {
            name: (param, RDMABuffer(param.view(torch.uint8).flatten()))
            for name, param in self.model.state_dict().items()
        }

# Generator side: pull weights directly from Learner's memory
class Generator(Actor):
    async def sync_weights(self):
        for name, (_, rdma_buf) in self.weight_buffers.items():
            await rdma_buf.read_into(
                self.model.state_dict()[name].view(torch.uint8).flatten()
            )
```

**Analogy**: Instead of the Learner packaging weights into a box, driving to the post office,
and mailing them (serialization + HTTP), the Generator reaches directly into the Learner's
filing cabinet and photocopies what it needs.

**Current limitation**: CPU tensors only. GPU RDMA is planned for future versions.
For our implementation, we'll use serialized state_dict via Monarch messaging instead --
simpler and works with GPU tensors.

### RDMA API Quick Reference

| Class/Method | What it does |
|-------------|-------------|
| `RDMABuffer(tensor)` | Register a CPU tensor for remote access |
| `buf.read_into(dst)` | Pull data from remote buffer into local tensor |
| `buf.write_from(src)` | Push local data into remote buffer |
| `is_ibverbs_available()` | Check if hardware RDMA is available |
| `get_rdma_backend()` | Returns `'ibverbs'`, `'tcp'`, or `'none'` |

---

## 7. Supervision (Fault Tolerance)

Actors form ownership trees. When a child actor fails, the parent gets notified
via a `__supervise__` method:

```python
class TrainingSupervisor(Actor):
    def __supervise__(self, failure):
        print(f"Child {failure.mesh_name} failed: {failure.report()}")
        # Return True = "I handled it, don't propagate"
        # Return False = "I can't handle it, tell my parent"
        return False
```

**Analogy**: Like a management chain. If a junior dev's code crashes,
their team lead is notified first. If the team lead can't fix it,
it escalates to the engineering manager, and so on.

This is borrowed from Erlang's "let it crash" philosophy -- instead of
trying to prevent every failure, you let things fail and handle it at
the right level.

---

## 8. Running on Kubernetes (EKS)

Monarch has first-class Kubernetes support via `KubernetesJob` and the **MonarchMesh CRD**.

### The MonarchMesh Operator

Install via Helm:
```bash
helm repo add monarch-operator https://meta-pytorch.github.io/monarch-kubernetes
helm upgrade --install monarch-operator monarch-operator/monarch-operator
```

This installs a Kubernetes operator that watches for `MonarchMesh` custom resources
and provisions pods (StatefulSets + Services) for Monarch workers.

### MonarchMesh CRD

```yaml
apiVersion: monarch.meta.com/v1alpha1
kind: MonarchMesh
metadata:
  name: grpo-training
spec:
  replicas: 2                    # 2 pods (one per node)
  template:
    spec:
      containers:
      - name: worker
        image: my-registry/monarch-grpo:latest
        resources:
          limits:
            nvidia.com/gpu: "8"  # 8 GPUs per pod
            memory: 800Gi
          requests:
            nvidia.com/gpu: "8"
            memory: 600Gi
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory         # In-memory for NCCL
          sizeLimit: 200Gi
```

**Analogy**: The MonarchMesh CRD is like a RayCluster CRD but for Monarch.
You declare "I want 2 pods with 8 GPUs each" and the operator handles provisioning.

### Connecting from Python

```python
from monarch.job import KubernetesJob

# Option 1: Provision pods from Python (creates MonarchMesh CRDs)
job = KubernetesJob()
state = job.apply()
hosts = state.workers  # HostMesh of allocated pods

# Option 2: Attach to pre-existing pods
job = KubernetesJob(mesh_name="grpo-training")
state = job.state()
hosts = state.workers

# Then spawn processes and actors
procs = hosts.spawn_procs(per_host={"gpus": 8})
trainer = procs.spawn("trainer", TrainerActor, config)
```

**Important**: The controller script must run **inside** the cluster (not from your laptop).
This is a current limitation of Monarch K8s support.

---

## 9. GRPO with Monarch Actors

Here's how the GRPO training pattern maps to Monarch actors for our Qwen + GSM8K setup:

```
Node 0 (8 GPUs)                          Node 1 (8 GPUs)
┌─────────────────────────┐              ┌─────────────────────────┐
│  Learner Actor Mesh     │              │  Generator Actor        │
│  (8 FSDP ranks)         │   weights    │  (vLLM engine)          │
│                         │ ──────────>  │                         │
│  - forward/backward     │              │  - generate completions │
│  - optimizer step       │              │  - update weights       │
│  - gradient clipping    │              │                         │
│  - checkpointing        │              │  Reference Model        │
├─────────────────────────┤              │  (CPU offloaded)        │
│  CPU Actors:            │              │  - compute ref logprobs │
│  - DatasetActor         │              └─────────────────────────┘
│  - RewardActor          │
│  - ReplayBufferActor    │
└─────────────────────────┘
```

### Data Flow (one training step)

```
1. DatasetActor.next_batch()
   └──> prompts + ground_truths

2. Generator.generate(prompts, group_size=8)
   └──> 8 completions per prompt + log_probs

3. RewardActor.score(completions, ground_truths)
   └──> binary rewards (0.0 or 1.0)

4. compute_grpo_advantages(rewards)
   └──> A_i = (r_i - mean(R)) / std(R) per group

5. Learner.train_step(completions, old_log_probs, advantages)
   └──> PPO-clipped loss + KL penalty, optimizer step

6. Learner.get_weights() --> Generator.update_weights()
   └──> every 3 steps, sync model weights
```

### Why Monarch helps here

With torchrun (our custom trainer), everything runs synchronously on the same GPU:
generate, score, train, repeat. The GPUs sit idle during generation (slow, autoregressive)
and during reward computation (CPU-bound).

With Monarch, generation and training can be on **different GPU pools**.
The Learner doesn't wait for the Generator, and vice versa.
A ReplayBuffer decouples them -- the Generator fills it, the Learner drains it.

---

## 10. Monarch vs Our Other Approaches

| Aspect | Custom Trainer | veRL (Ray) | Monarch |
|--------|---------------|------------|---------|
| **Orchestration** | torchrun | Ray Cluster | MonarchMesh CRD |
| **Programming model** | SPMD (same code everywhere) | Ray tasks + actors | Monarch actors + meshes |
| **Communication** | HTTP + NCCL | Ray object store | Monarch messages + RDMA |
| **Generation** | Separate vLLM pod via HTTP | Colocated vLLM | vLLM inside Generator actor |
| **Training** | Manual FSDP loop | veRL trainer | FSDP inside Learner actor |
| **Weight sync** | HTTP POST state_dict | Direct (colocated) | Monarch messaging (~3GB) |
| **Reward** | Separate FastAPI service | Inline function | RewardActor |
| **Async gen+train** | No (sequential) | Limited | Yes (actor-based) |
| **Fault tolerance** | Crash all | Ray restart | Supervision trees |
| **K8s resource** | Job (Indexed) | RayCluster CRD | MonarchMesh CRD |

---

## 11. Key Vocabulary

| Term | Meaning |
|------|---------|
| **Actor** | Isolated computation unit with private state and a mailbox |
| **@endpoint** | Decorator marking a method as callable by other actors |
| **ProcMesh** | Grid of processes organized by dimensions (hosts x gpus) |
| **ActorMesh** | Collection of actor instances spawned across a ProcMesh |
| **HostMesh** | Collection of hosts (machines) that can spawn processes |
| **RDMA Buffer** | Memory region exposed for direct remote read/write |
| **Supervision** | Parent actors handling child failures (Erlang-style) |
| **MonarchMesh** | Kubernetes CRD that provisions Monarch worker pods |
| **call_one** | Send message to one actor, wait for response |
| **call** | Broadcast to all actors in mesh, collect all responses |
| **broadcast** | Fire-and-forget message to all actors |
| **KubernetesJob** | Python class to provision or attach to MonarchMesh pods |

---

## 12. Installation & Runtime

```bash
# Install Monarch
pip install torchmonarch                    # stable
pip install torchmonarch[kubernetes]        # with K8s support

# Your script is a regular Python file
python my_monarch_script.py

# For Kubernetes:
# 1. Install operator
helm repo add monarch-operator https://meta-pytorch.github.io/monarch-kubernetes
helm upgrade --install monarch-operator monarch-operator/monarch-operator
# 2. Apply MonarchMesh manifest or provision from Python
# 3. Run controller script inside the cluster
```

**Python**: 3.10 - 3.13
**Platform**: Linux only (x86_64, aarch64)
**GPU**: Optional (CPU-only mode: `USE_TENSOR_ENGINE=0 pip install torchmonarch`)
