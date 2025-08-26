# MSCCL++ DSL Integration Guide

MSCCL++ DSL (domain‑specific language) enables concise expression of collective algorithms as Python functions. We provides pythonic interfaces to help users author, JIT‑compile, register, and select collective execution plans. Two modes of usage are supported: native MSCCL++ collective calls, and NCCL interposition (transparent acceleration of existing PyTorch code).

## User‑Facing API Surface

Author a collective algorithm in the Python DSL (or reuse presets), JIT‑compile it into a single canonical JSON execution plan, register it, and let a selector pick the plan per collective call.

### DSL Example
```python
def allreduce_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram (
        name,
        collective,
        gpu_size,
        instances=8,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ) as program:
        # Creating Channels
        nvls_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.input)
        channels = {}
        for gpu in range(gpu_size):
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)] = MemoryChannel(peer, gpu)

        # Synchronization to Ensure all the Gpus are Ready
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True, data_sync=SyncType.after)

        # Reducing and Storing the data
        for gpu in range(gpu_size):
            buffer_offset = gpu
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            nvls_chan.at_rank(gpu).reduce(
                buffer_offset=buffer_offset, size=1, dst_chunk=input_buffer[gpu : gpu + 1], tb=0
            )
            nvls_chan.at_rank(gpu).broadcast(
                src_chunk=input_buffer[gpu : gpu + 1], buffer_offset=buffer_offset, size=1, tb=0
            )

        # Synchronization to Ensure the Gpus finished
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True, data_sync=SyncType.before)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True)

    return program
```

### Typical Usage (MSCCL++ Native)
```python
from mscclpp.dsl import presets, jit
import mscclpp

plan = jit.compile(
    algo=allreduce_nvls,
    name="allreduce_nvls",
    collective="allreduce",
    nranks_per_node=8,
    world_size=world_size,
    instances=2,
    protocol="Simple",
    num_threads_per_block=1024,
    min_msg_size=1<<20,
    max_msg_size=48<<30,
    tags={"nvls"},
)
mscclpp.plan.register(plan)


def selector(plans: Dict[str, mscclpp.PlanHandle], req: mscclpp.Request):
    collective_plans = plans.get(req.collective)
    nvls = [p for p in collective_plans if "nvls" in p.tags]
    return nvls[0] if nvls else collective_plans[0]

mscclpp.plan.set_selector(selector)

# Use it
commGroup = mscclpp.comm.CommGroup(interfaceIpPortTrio=ifIpPortTrio, rank=rank, size=world_size)

# Users can define their own communication collectives
commGroup.all_reduce(x, out=y, op="sum", dtype="f16")
```

### Typical Usage (NCCL Interposition)
```python
import torch, torch.distributed as dist
from mscclpp.dsl import presets, jit
import mscclpp

dist.init_process_group(backend="nccl")

plan = jit.compile(
    algo=allreduce_nvls,
    name="allreduce_nvls",
    collective="allreduce",
    nranks_per_node=8,
    world_size=world_size,
    instances=2,
    protocol="Simple",
    num_threads_per_block=1024,
    min_msg_size=1<<20,
    max_msg_size=48<<30,
    tags={"nvls"},
)
mscclpp.plan.register(plan)

def selector(plans, req):
    collective_plans = plans.get(req.collective)
    nvls = [p for p in collective_plans if "nvls" in p.tags]
    return nvls[0] if nvls else collective_plans[0]

mscclpp.plan.set_selector(selector)

x = torch.randn(12<<20, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
```

### Compilation
```python
jit.compile(
    algo,                # DSL builder / preset
    name: str,           # logical name for this compiled variant
    collective: str,     # e.g. "allreduce"
    world_size: int,
    instances: int = 1,
    protocol: str = "Simple",
    num_threads_per_block=1024,
    min_msg_size = 0,
    max_msg_size = 1<<32,
    nranks_per_node: int,
    tags: set[str] | None = None
    rebuild: bool = False,
) -> PlanHandle
```
Returns PlanHandle with: id, name, collective, tags, constraints, executionPlan.

```python
@dataclass(frozen=True)
class PlanHandle:
    id: str                # unique plan id (hash)
    name: str              # user supplied name
    collective: str        # e.g. "allreduce"
    tags: set[str]         # user supplied tags
    constraints: dict      # e.g. {"min_msg_size": ..., "max_msg_size": ...}
    executionPlan: mscclpp.ExecutionPlan  # loaded from JSON
```

### Registry
```python
mscclpp.plan.register(plan: PlanHandle) -> None
mscclpp.plan.list(
    collective: str | None = None,
    tags: set[str] | None = None
) -> list[PlanHandle]
```

### Selector Interface
Deterministic, pure (no time, randomness, rank).
```python
@dataclass(frozen=True)
class Request:
    collective: str
    msg_bytes: int
    world_size: int
    nranks_per_node: int
    hints: dict

Selector = Callable[[Dict[str, PlanHandle], Request], PlanHandle | str]
mscclpp.plan.set_selector(selector: Selector) -> None
mscclpp.plan.clear_selector() -> None
```
If selector returns a string it is treated as a plan id.

### Collective Call
```python
mscclpp.comm.CommGroup.allreduce(send, out=recv, op="sum", dtype="f16", plan: str | None = None)
```
If plan id supplied, selector is bypassed.

## Implementation Details

### Plan JSON & Determinism
Canonical key ordering + identical lowering on every rank → identical bytes → identical hash.
```
plan_id = blake3(canonical_json({
  schema_version,
  compiler_version,
  algo_name,
  algo_src_hash,   // hash of the algo dsl code
  env_fingerprint,
})).base32()[:32]
```
env_fingerprint includes world_size, size_bucket(msg_size), nranks_per_node, instances, protocol, num_threads_per_block.

### Cache Layout
Root: $MSCCLPP_EXECUTION_PLAN_DIR or ~/.cache/mscclpp
```
<cache_root>/plans/<collective>/<plan_id>.json
```

### Compile Algorithm (Pseudo)
```python
def compile(...):
    key = compute_key(...)
    if registry.contains(key):
        return registry.get(key)
    path = path_for(key.plan_id)
    if file_exists(path) and not rebuild:
        handle = load_and_validate(path)
        registry.put(key, handle)
        return handle
    json_plan, canonical_inputs = lower(...)
    assert blake3(canonical_inputs)[:32] == key.plan_id
    atomic_write_with_lock(path, json_plan)
    handle = PlanHandle(...)
    registry.put(key, handle)
    return handle
```
Regenerate on: source change, compiler/schema bump, env fingerprint change, explicit rebuild, hash mismatch.

### Selection Precedence
1. Explicit plan id argument
2. User selector (if set)
3. Registered cuda kernel algorithm
4. NCCL/RCCL implementation


### NCCL Interposition
C shim builds a request struct mirroring Request, calls registered Python selector. If plan JSON absent, optional ensure callback triggers compilation; else loads JSON and launches executor.

Refer: https://nanobind.readthedocs.io/en/latest/functions.html#higher-order-functions

