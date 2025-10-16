# MSCCL++ DSL Integration Guide

MSCCL++ DSL (domain‑specific language) enables concise expression of collective algorithms as Python functions. We provides pythonic interfaces to help users author, JIT‑compile, register, and select collective execution plans. Two modes of usage are supported: native MSCCL++ collective calls, and NCCL interposition (transparent acceleration of existing PyTorch code).

## Initial Setup

After cloning the repository and completing the basic project setup, run the following at the mscclpp foulder:

1. Install Python dependencies
```bash
pip install -r ./python/<requirements_file>
```
Replace <requirements_file> with the appropriate file for your environment (e.g., requirements_cuda11.txt, requirements_cuda11.txt or requirements_rocm6.txt).

2. Install the module and generate default algorithm plans
```bash
pip install . && python3 -m mscclpp --install
```

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
When using MSCCL++ in native mode, you are not leveraging the existing NCCL communicator.
Instead, you create and manage your own MSCCL++ communicator directly.
To integrate this with PyTorch, it’s common to wrap the low-level communicator inside a high-level Python class that exposes a PyTorch-compatible interface for collective operations such as all_reduce.

The example below shows how to build a simple wrapper that performs an AllReduce using MSCCL++ natively. Please note that in the allreduce function, you need to use the ExecutionPlanRegister to select the registered algorithms that best match the request and then call the mscclpp executor directly.

```python
class CustomizedComm:
    """High-level MSCCL++ wrapper compatible with PyTorch-style collectives."""

    def __init__(self, comm: mscclpp_comm.CommGroup):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        
        # Initialize MSCCL++ components
        self.registry = mscclpp.ExecutionPlanRegistry()
        self.executor = mscclpp.Executor(comm.communicator)

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream: torch.cuda.Stream = None):
        """Performs an AllReduce operation using a native MSCCL++ plan."""
        assert op == torch.distributed.ReduceOp.SUM
        
        # Select an appropriate execution plan
        plan = self.registry.select(
            collective="allreduce",
            world_size=self.world_size,
            n_ranks_per_node=self.n_ranks_per_node,
            send_buffer=tensor.data_ptr(),
            recv_buffer=tensor.data_ptr(),
            message_size=tensor.numel() * tensor.element_size(),
        )
        if plan is None:
            raise ValueError(
                f"No suitable plan found for collective allreduce with message size {tensor.numel() * tensor.element_size()}"
            )
        
        # Execute the plan using the MSCCL++ executor
        self.executor.execute(
            self.rank,
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
            tensor.numel() * tensor.element_size(),
            dtype_to_mscclpp_dtype(tensor.dtype),
            plan.plan,
            stream.cuda_stream if stream is not None else 0,
        )
```

Once the high-level wrapper (CustomizedComm) is defined, performing collective operations such as AllReduce becomes just as simple as using PyTorch’s native torch.distributed interface. To perform all_reduce, simply use ```comm.all_reduce(x, op=torch.distributed.ReduceOp.SUM)```.

In native MSCCL++ mode, you explicitly create and manage your communicator, compile a DSL algorithm, and register it with the MSCCL++ runtime before launching collectives.

```python
from mscclpp.dsl import presets, jit
import mscclpp

# Step 1. Compile and register a DSL plan
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

# Step 2. Define a plan selector (choose algorithm based on tags, message size, etc.)
def selector(plans: Dict[str, mscclpp.PlanHandle], req: mscclpp.Request):
    collective_plans = plans.get(req.collective)
    nvls = [p for p in collective_plans if "nvls" in p.tags]
    return nvls[0] if nvls else collective_plans[0]

mscclpp.plan.set_selector(selector)

# Step 3. Initialize communicator and high-level wrapper
mscclpp_group = mscclpp.comm.CommGroup(interfaceIpPortTrio=ifIpPortTrio, rank=rank, size=world_size)
comm = CustomizedComm(mscclpp_group)

# Step 4. Perform the AllReduce operation
x = torch.randn(12<<20, dtype=torch.float16, device="cuda")
comm.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
```

You can launch the script just like any other PyTorch distributed program, for example: 
```bash
MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  cusotmized_comm.py
``` 

### Typical Usage (NCCL Interposition)
In this mode, PyTorch continues to use the nccl backend, but the underlying NCCL calls are intercepted and executed by MSCCL++.
This approach allows you to reuse the standard PyTorch distributed interface (dist.all_reduce, dist.all_gather, etc.) while transparently benefiting from optimized MSCCL++ collective algorithms.

This is ideal when you want drop-in acceleration with no code changes to your training scripts.

```python
import torch, torch.distributed as dist
from mscclpp.dsl import presets, jit
import mscclpp

# Step 1. Initialize the PyTorch distributed process group using the NCCL backend
dist.init_process_group(backend="nccl")

# Step 2. Compile and register an MSCCL++ DSL plan
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

# Step 3. Define and set a selector to choose the appropriate plan at runtime
def selector(plans, req):
    collective_plans = plans.get(req.collective)
    nvls = [p for p in collective_plans if "nvls" in p.tags]
    return nvls[0] if nvls else collective_plans[0]

mscclpp.plan.set_selector(selector)

# Step 4. Perform the AllReduce as usual
x = torch.randn(12<<20, dtype=torch.float16, device="cuda")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
```
To run with NCCL interposition, you preload the MSCCL++ shim so it transparently intercepts NCCL calls made by PyTorch’s nccl backend. In practice, nothing in your training script changes—PyTorch still calls dist.all_reduce, etc.—but execution is routed through your registered MSCCL++ DSL plans. Example single-node launch:
```bash
LD_PRELOAD=<MSCCLPP_REPO>/build/apps/nccl/libmscclpp_nccl.so torchrun --nnodes=1 --nproc_per_node=8 dsl-torch-integration/dsl_with_nccl_api.py
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

