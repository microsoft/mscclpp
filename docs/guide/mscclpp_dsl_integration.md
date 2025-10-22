# MSCCL++ DSL Integration Guide

MSCCL++ DSL (domain-specific language) enables concise expression of collective algorithms as Python functions.
MSCCL++ offers pythonic utilities to author, JIT-compile, register, and select execution plans. This guide walks through two integration paths: a customized MSCCL++ communicator and NCCL interposition that accelerates existing PyTorch `backend="nccl"` workloads.

## Initial Setup

Run the following from the repository root after completing the basic project setup:

1. Install Python dependencies.
   ```bash
   pip install -r ./python/<requirements_file>
   ```
   Replace `<requirements_file>` with the file that matches your environment (e.g., `requirements_cuda11.txt`, `requirements_cuda12.txt`, or `requirements_rocm6.txt`).

2. Install the module and generate default algorithm plans.
   ```bash
   pip install . && python3 -m mscclpp --install
   ```

## Integration Options

MSCCL++ DSL integrates into your training or inference workload in two ways:
1. **Custom MSCCL++ Communicator** — directly manage an MSCCL++ communicator and launch collectives with the MSCCL++ executor.
2. **NCCL Interposition** — keep using `backend="nccl"`; MSCCL++ intercepts NCCL calls at runtime for drop-in acceleration.

Both paths follow the same high-level flow:
1. Author (or reuse) a collective algorithm with the MSCCL++ DSL.
2. Compile it into an execution plan.
3. Register the plan with the MSCCL++ runtime.
4. Configure a selector to choose the plan for each collective call.

Below we show an AllReduce example and then detail each integration option.

### Example: AllReduce in the MSCCL++ DSL
The snippet defines an AllReduce that uses NVLS for intra-node reduce-scatter followed by broadcast.
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

        # Synchronization to ensure all GPUs are ready
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

        # Reduce then broadcast one chunk per GPU
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

        # Synchronization to ensure all GPUs finish
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

### Integrate with MSCCL++ customized communicator
Use when you want a PyTorch‑compatible interface with fine‑grained control. You manage the communicator, compile/register DSL plans, and invoke collectives via a thin wrapper. The example below shows an AllReduce built on the MSCCL++ communicator and executor.

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

#### Usage Example

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

#### Launch (single node)
```bash
MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm.py
```

### Integrate via NCCL Interposition
Keep your script as‑is: init PyTorch with backend="nccl"; MSCCL++ intercepts NCCL calls for drop‑in acceleration.

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

#### Launch with interposition
To run with NCCL interposition, you preload the MSCCL++ shim so it transparently intercepts NCCL calls made by PyTorch’s nccl backend.
```bash
LD_PRELOAD=<MSCCLPP_REPO>/build/apps/nccl/libmscclpp_nccl.so torchrun --nnodes=1 --nproc_per_node=8 dsl-torch-integration/dsl_with_nccl_api.py
```
## Notices:
 - When using NCCL interposition, the algorithm selection order is:
   1. Check for registered DSL plans matching the collective call.
   2. Check for a customized kernel implementation if no DSL plan fits.
   3. Fall back to the default NCCL implementation (set `MSCCLPP_NCCL_LIB_PATH` to the original NCCL library).
