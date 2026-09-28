(torchcomms)=
# TorchComms Integration

MSCCL++ integrates with [TorchComms](https://github.com/meta-pytorch/torchcomms), enabling PyTorch users to use MSCCL++ collectives through a standard API. This is the recommended way to use MSCCL++ in PyTorch training — particularly for mixed-backend setups where you want MSCCL++ for the hot-path collectives (allreduce, allgather) and NCCL/RCCL for everything else.

```python
import torch
import torchcomms
import mscclpp_torchcomms  # auto-registers the backend

comm = torchcomms.new_comm("mscclpp", torch.device(f"cuda:{local_rank}"), name="grad_sync")
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
comm.finalize()
```

## Why TorchComms

MSCCL++ provides GPU-driven collectives that are faster than NCCL for many workloads (especially allreduce on NVSwitch/H100 systems), but using them directly requires custom CUDA kernels and manual connection setup. The existing NCCL compatibility shim (`LD_PRELOAD`) works but prevents mixed-backend usage and masks MSCCL++'s identity.

TorchComms solves this:

- **Mixed-backend training**: Use MSCCL++ for gradient allreduce (~90% of communication time) and NCCL for broadcast, barrier, send/recv — no code changes.
- **Clean integration**: Training frameworks using TorchComms (torchtitan, FSDP2, etc.) swap in MSCCL++ with one line.
- **Proper identity**: MSCCL++ appears as its own backend, not masquerading as NCCL. This matters for debugging, profiling, and configuration.
- **Automatic algorithm selection**: The backend automatically selects the best algorithm (NVLS warp pipeline, packet, fullmesh, RS+AG, etc.) based on message size, topology, and hardware.

## Installation

### Prerequisites

| Dependency | Tested Version | Notes |
|---|---|---|
| PyTorch | 2.10.0+cu128 | Other versions with TorchComms support should work |
| torchcomms | 0.2.0 | `pip install --pre torchcomms` |
| pybind11 | 3.0.2 | Build dependency |
| glog | (any recent) | Build dependency |

**GPU support:** Tested on NVIDIA GPUs with CUDA 12.8. AMD ROCm GPUs are supported at the build level (MSCCL++ uses a CUDA/HIP translation layer), but the TorchComms backend has not been validated on ROCm yet.

### pip install (recommended)

```bash
$ python -m pip install ./python/mscclpp_torchcomms
```

This builds and installs the `mscclpp-torchcomms` package. The backend `.so` is automatically discovered — no environment variable needed.

### CMake build

For development or integration into an existing build:

```bash
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON ..
$ make -j$(nproc)
$ cd ..
```

When using the CMake build path (without pip install), set the environment variable so TorchComms can discover the backend:

```bash
$ export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP=$PWD/build/lib/_comms_mscclpp.cpython-*.so
```

## Usage

```bash
$ torchrun --nproc_per_node=8 your_script.py
```

```python
import torch
import torchcomms
import mscclpp_torchcomms  # auto-registers the backend .so path

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

# Create an MSCCL++ communicator
comm = torchcomms.new_comm("mscclpp", device, name="my_comm")

# AllReduce — MSCCL++ automatically selects the best algorithm
tensor = torch.randn(1024 * 1024, device=device)
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)

# AllGather
input_chunk = torch.randn(1024, device=device)
output = torch.empty(1024 * world_size, device=device)
comm.all_gather_single(output, input_chunk, False)

# Cleanup
comm.finalize()
```

Alternatively, if you prefer not to add the `mscclpp_torchcomms` import, set the environment variable directly:

```bash
$ export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP=/path/to/_comms_mscclpp.cpython-*.so
```

### Mixed-Backend Training

Use MSCCL++ for high-performance collectives and NCCL for everything else:

```python
import torch
import torchcomms
import mscclpp_torchcomms

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

# MSCCL++ for gradient sync (hot path)
mscclpp_comm = torchcomms.new_comm("mscclpp", device, name="grad_sync")

# NCCL for everything else (broadcast, barrier, etc.)
nccl_comm = torchcomms.new_comm("nccl", device, name="control")

for epoch in range(num_epochs):
    loss = model(data)
    loss.backward()

    # Fast gradient allreduce via MSCCL++
    for param in model.parameters():
        mscclpp_comm.all_reduce(param.grad, torchcomms.ReduceOp.SUM, False)

    optimizer.step()

mscclpp_comm.finalize()
nccl_comm.finalize()
```

## Architecture

### What Happens When You Create a Communicator

When `torchcomms.new_comm("mscclpp", device)` is called, TorchComms dlopen's the `_comms_mscclpp.*.so` module and invokes `init()`, which:

1. **Bootstrap** — discovers rank/world_size from the torchrun environment, exchanges a `UniqueId` through `c10d::Store` (rank 0 generates, others read), creates the MSCCL++ `Communicator` with a `TcpBootstrap`.
2. **Scratch buffer** — allocates 128MB via `GpuBuffer` (`cuMemMap`) for native algorithms that need intermediate storage.
3. **Executor** — creates the DSL plan executor (used by DSL algorithms, ignored by native ones).
4. **Algorithm collection** — calls `AlgorithmCollectionBuilder::buildDefaultAlgorithms()` which registers native algorithms + DSL plans, then wires up the topology-aware algorithm selector.
5. **Event pool** — pre-allocates a pool of 256 reusable CUDA events for async work tracking.

### What Happens When You Call a Collective

```
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    │
    ▼
TorchCommMSCCLPP::all_reduce()
    │  validates reduce op (SUM, MIN)
    │  checks tensor is contiguous
    │
    ▼
TorchCommMSCCLPP::executeCollective("allreduce", ...)
    │
    │  1. Builds a CollectiveRequest with world_size, nRanksPerNode,
    │     rank, buffer pointers, message size, stream, dtype
    │
    │  2. Calls algorithmCollection_.selectAlgorithm(request)
    │     → considers message size, NVLS support, compute capability,
    │       symmetric memory, CUDA graph capture mode
    │     → returns the best algorithm
    │
    │  3. Creates TorchWorkMSCCLPP handle, records start GPU event
    │
    │  4. Calls algo->execute(...)
    │     → native algorithms launch a CUDA kernel directly
    │     → DSL algorithms use the executor to interpret a JSON plan
    │
    │  5. Records end GPU event, returns the work handle
    │
    ▼
TorchWorkMSCCLPP (returned to caller)
    │  wait() → cudaStreamWaitEvent on caller's stream (GPU-side, no CPU block)
    │  checkStatus() → polls GPU events for completion/timeout
```

### Component Diagram

```
torchcomms.new_comm("mscclpp", device)
    │
    ▼
TorchCommMSCCLPPPy.cpp          ← pybind11 module + dynamic loader interface
    │
    ▼
TorchCommMSCCLPP.cpp/hpp        ← backend class (init, finalize, collective dispatch)
    │
    ├── TorchCommMSCCLPPBootstrap  ← rank discovery via c10d::Store
    ├── TorchWorkMSCCLPP           ← GPU event pool + async work tracking
    │
    ▼
AlgorithmCollection::selectAlgorithm()   ← MSCCL++ algorithm selection
    │
    ▼
Algorithm::execute()                      ← GPU kernel launch (native or DSL)
```

## Supported Collectives

| Collective | Status | Algorithms | Notes |
|---|---|---|---|
| AllReduce | Supported | allpair_packet, nvls_packet, packet, nvls_zero_copy, nvls_warp_pipeline, nvls_block_pipeline, fullmesh, rsag, rsag_pipeline, rsag_zero_copy | SUM, MIN. Auto-selected by message size + topology. |
| AllGather | Supported | fullmesh, fullmesh2 | Auto-selected by message size. |
| ReduceScatter | Supported (with custom algorithm) | — | No default algorithms ship. Requires registering a DSL or native algorithm via `AlgorithmCollectionBuilder`. |
| AllToAll | Supported (with custom algorithm) | — | No default algorithms ship. Requires registering a DSL or native algorithm via `AlgorithmCollectionBuilder`. |
| Broadcast | Not supported | — | Use a separate NCCL/RCCL communicator. |
| Reduce | Not supported | — | Use a separate NCCL/RCCL communicator. |
| Send/Recv | Not supported | — | Use a separate NCCL/RCCL communicator. |
| Barrier | Not supported | — | Use a separate NCCL/RCCL communicator. |
| Scatter/Gather | Not supported | — | Use a separate NCCL/RCCL communicator. |

Unsupported collectives throw a `RuntimeError` with an explicit message naming the operation and suggesting the caller use a separate NCCL/RCCL communicator.

## Algorithm Selection

The backend uses the same topology-aware algorithm selector as the NCCL compatibility extension. Selection considers:

- **Message size**: Small messages (≤1MB) use packet-based algorithms for lower latency. Large messages use non-packet algorithms for higher bandwidth.
- **NVLS support**: On NVSwitch-connected systems (H100, etc.), NVLS algorithms (warp pipeline, block pipeline) are preferred for large allreduce.
- **Compute capability**: Some algorithms require SM 9.0+ (Hopper).
- **Buffer allocation**: Zero-copy NVLS algorithms require `cuMemMap`-allocated buffers.
- **CUDA graph capture**: Some algorithms are compatible with CUDA graph capture mode.

The selector picks the best algorithm automatically. Users do not need to configure algorithm selection for default usage.

## User-Defined Algorithms

Custom algorithms (DSL or native) can be registered via the `AlgorithmCollectionBuilder` singleton **before** creating the TorchComms communicator. The backend picks them up during `init()`.

```python
import mscclpp
from mscclpp.language.collectives import AllReduce
import torchcomms
import mscclpp_torchcomms

# 1. Configure algorithms on the builder singleton
builder = mscclpp.AlgorithmCollectionBuilder()
spec = mscclpp.AlgoSpec(name="my_allreduce", collective=AllReduce(8, 1, True))
algo = mscclpp.compile(algo=my_allreduce_fn, algo_spec=spec, rank=rank)
builder.add_algorithm_builder(algo)
builder.set_algorithm_selector(my_selector)

# 2. Create comm — init() picks up everything from the builder
comm = torchcomms.new_comm("mscclpp", device, name="custom")

# 3. Collectives use the configured algorithms automatically
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
```

## Environment Variables

| Variable | Description |
|---|---|
| `TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP` | Path to the `_comms_mscclpp.*.so` module. **Automatically set** when `mscclpp-torchcomms` is pip-installed. Only needed for CMake-only builds. |

## Testing

All tests are launched via `torchrun`:

```bash
# Collective correctness (allreduce, allgather, reducescatter)
$ torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --all

# With size/dtype sweep (exercises both packet and non-packet algorithm paths)
$ torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --all --sweep

# Message size sweep (1 to 32MB)
$ torchrun --nproc_per_node=2 test/torchcomms/test_sizes.py

# Error handling (unsupported ops, invalid reduce ops)
$ torchrun --nproc_per_node=2 test/torchcomms/test_error_handling.py

# Simulated training loop
$ torchrun --nproc_per_node=2 test/torchcomms/test_training_loop.py

# User-defined algorithm registration
$ torchrun --nproc_per_node=2 test/torchcomms/test_user_algorithms.py
```

## Benchmarks

```bash
$ torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allreduce --warmup 100 --iters 200
$ torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allgather --warmup 100 --iters 200
```

Generate a report from benchmark output:

```bash
$ python test/torchcomms/bench_report.py --input bench_results/torchcomms_raw.json
```

## Limitations

- **Single-tensor variants only.** MSCCL++'s `Algorithm::execute()` operates on contiguous buffers, so the backend implements `all_gather_single` and `reduce_scatter_single` but not the tensor-list variants. The tensor-list variants throw with guidance to use the single-tensor variant.
- **Contiguous tensors required.** All input and output tensors must be contiguous. Non-contiguous tensors raise a `RuntimeError`.
- **Unsupported collectives throw at runtime.** Broadcast, reduce, send/recv, barrier, scatter, and gather throw a `RuntimeError` with guidance to use NCCL/RCCL.

## Troubleshooting

### "Backend mscclpp specified, but TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP not set"

The test or script is not importing `mscclpp_torchcomms`. Add `import mscclpp_torchcomms` before `torchcomms.new_comm()`, or set the environment variable manually if not using pip install.

### "Requested fd not found, size of fdSet_ is 0"

The scratch buffer was allocated with `cudaMalloc` instead of `GpuBuffer` (`cuMemMap`). This means POSIX file descriptors were not registered in the unix socket server for cross-rank IPC sharing. This is a build issue — ensure the backend is built against the correct MSCCL++ version.

### "No algorithm registered for collective X"

The algorithm selector found no matching algorithm for the given collective, message size, and topology. For ReduceScatter and AllToAll, you need to register a DSL algorithm via `AlgorithmCollectionBuilder` before creating the MSCCL++ communicator or use a different communicator.

### CUDA device mismatch errors

The backend uses `CudaDeviceGuard` to restore the CUDA device after `init()`. If you see device mismatch errors, ensure the `device` argument to `torchcomms.new_comm()` matches `LOCAL_RANK`.
