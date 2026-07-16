# MSCCL++ Expert-Parallel (EP) extension

The EP extension is a torch-free nanobind module for MoE dispatch and combine.
It builds two backends:

- **Low latency (LL)**: `MoERuntime`, backed by
  `low_latency/{dispatch,combine}.cu`.
- **High throughput (HT)**: `ExpertParallelRuntime`, backed by
  `ht_runtime.cc` and the CUDA sources under `ht/`.

## Status

| Feature | Status |
|---|---|
| LL dispatch/combine | Validated on Hopper and newer GPUs |
| HT dispatch/combine | Supports 2, 4, 8, or 16 ranks in one GPU IPC/NVL fabric domain |
| HT RDMA/IB fallback | Not supported |
| Python frontend | `mscclpp.ep.MoECommunicator` selects LL or HT with `MoEMode` |
| ROCm | Not supported |

## Runtime architecture

### Low latency

LL allocates CUDA physical symmetric memory and maps peer buffers through the
existing `mscclpp::Communicator`. Payloads use direct peer mappings;
`BaseMemoryChannel` handles are used only for synchronization.

The optimized LL backend is available when all participating ranks belong to
one detected GPU IPC domain. That domain may span hosts when CUDA fabric handles
and the required fabric services are available.

LL dispatch supports two user-visible layouts:

- `EXPERT_MAJOR`: one row per `(token, local expert)`.
- `TOKEN_MAJOR`: one row per `(token, destination rank)`, plus local top-k expert
  IDs, routing weights, source-token IDs, per-source-rank counts, and exclusive
  offsets. Valid rows occupy a compact prefix of the caller's worst-case capacity
  buffer. The caller must produce one pre-weighted local partial per row before
  combine.

### High throughput

HT follows the same direct-mapping resource model:

1. Python passes the existing `mscclpp::Communicator` into
   `ExpertParallelRuntime`.
2. Each rank allocates a ring/FIFO region plus a CUDA physical direct receive
   pool. Same-host ring buffers use `cudaMalloc` runtime IPC for peak latency;
   a fabric-domain job spanning hosts uses CUDA physical memory for the ring.
3. The runtime exchanges and maps those allocations with
   `Communicator::sendMemory` / `recvMemory`.
4. Dispatch and combine launch directly on the caller's CUDA stream.

The detected GPU IPC domain may span multiple hosts, such as an NVL fabric
domain with CUDA fabric handles. HT does not create a private bootstrap, proxy
service, RDMA channel, NVLS multicast object, or private communication stream,
and it has no RDMA/IB fallback outside that domain.

The HT dispatch API remains two-phase because the receive token count is data
dependent:

1. `notify_dispatch` exchanges counts and produces prefix matrices.
2. Python allocates the exact receive tensors.
3. `dispatch` moves token data and metadata.

Cached dispatch reuses the previous receive count and prefix matrices.

## HT data paths

The baseline path uses the DeepEP-style intranode ring. Optional runtime paths
use the peer-mapped receive pool:

- `MSCCLPP_EP_INTRA_DIRECT=1`: send hidden rows directly to their final receive
  slots. The physical receive pool is allocated and mapped only when this flag
  is enabled before runtime construction.
- `MSCCLPP_EP_INTRA_ALLSENDER=0|1`: controls the all-sender dispatch path when
  direct dispatch and TMA combine are enabled; default is enabled.
- `MSCCLPP_EP_COMBINE_TMA=0|1`: selects TMA direct-gather combine; default is
  enabled when its inputs are available.
- `MSCCLPP_EP_DISPATCH_NSM=<N>`: overrides the dispatch block count, rounded
  down to an even value.
- `MSCCLPP_EP_COMBINE_NSM=<N>`: overrides the TMA combine block count.

The persistent HT configuration contains only:

| Field | Meaning |
|---|---|
| `num_sms` | Maximum HT communication block budget |
| `num_max_nvl_chunked_send_tokens` | Ring send chunk size |
| `num_max_nvl_chunked_recv_tokens` | Ring receive capacity |

## Build

Python builds include the EP extension by default:

```bash
python3 -m pip install .
```

Plain CMake builds can enable it explicitly:

```bash
cmake -S . -B build -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build -j 64
```

The EP extension requires CUDA architecture 90 or newer.

Available CMake options:

| Variable | Default | Meaning |
|---|---:|---|
| `MSCCLPP_BUILD_EXT_EP` | Python builds: `ON` | Build the EP extension |
| `MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT` | `OFF` | Use a shorter kernel spin timeout |

## Source layout

```text
src/ext/ep/
├── bindings.cpp
├── moe_runtime.{cc,hpp}
├── ht_runtime.{cc,hpp}
├── ht/
│   ├── config.hpp
│   ├── buffer.cuh
│   ├── layout.cu
│   ├── intranode_kernel.cu
│   └── runtime.cu
├── include/
└── low_latency/
    ├── config.cuh
    ├── dispatch.cu
    └── combine.cu
```

## Validation

Build the extension, then run the single-node HT test:

```bash
HWLOC_COMPONENTS=-gl \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
torchrun --standalone --nproc_per_node=8 \
    test/python/ep/test_intranode_multirank.py
```

The LL validation remains:

```bash
HWLOC_COMPONENTS=-gl \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
torchrun --standalone --nproc_per_node=8 \
    test/python/ep/test_low_latency_multirank.py
```
