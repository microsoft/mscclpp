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
def allreduce_nvls(spec: mscclpp.AlgoSpec) -> CollectiveProgram:
    gpu_size = spec.world_size
    with CollectiveProgram(
        spec.name,
        spec.collective,
        gpu_size,
        instances=8,
        protocol=spec.protocol,
        num_threads_per_block=spec.num_threads_per_block,
        min_message_size=spec.min_message_size,
        max_message_size=spec.max_message_size,
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

### Integrate with MSCCL++ customized communicator
Use when you want a PyTorch‑compatible interface with fine‑grained control. You manage the communicator, compile/register DSL plans, and invoke collectives via a thin wrapper. The example below shows an AllReduce built on the MSCCL++ communicator and executor.
Example source directory:
```
examples/torch-integration
```
Key file: `customized_comm.py`.


#### Launch (single node)
```bash
MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm.py
```

### Integrate via NCCL Interposition
Keep your script as‑is: init PyTorch with backend="nccl"; MSCCL++ intercepts NCCL calls for drop‑in acceleration.
Example source directory:
```
examples/torch-integration
```
Key file: `dsl_with_nccl_api.py`.

#### Launch with interposition
To run with NCCL interposition, you preload the MSCCL++ shim so it transparently intercepts NCCL calls made by PyTorch’s nccl backend.
```bash
LD_PRELOAD=<MSCCLPP_REPO>/build/apps/nccl/libmscclpp_nccl.so torchrun --nnodes=1 --nproc_per_node=8 dsl_with_nccl_api.py
```
## Notices:
 - When using NCCL interposition, the algorithm selection order is:
   1. Check for registered DSL plans matching the collective call.
   2. Check for a customized kernel implementation if no DSL plan fits.
   3. Fall back to the default NCCL implementation (set `MSCCLPP_NCCL_LIB_PATH` to the original NCCL library).
