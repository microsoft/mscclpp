# MSCCL++ Expert-Parallel (EP) extension

The EP extension is a torch-free nanobind module for MoE dispatch and combine.
It exposes one concrete `MoERuntime` and one persistent `DeviceContext*` shared
by dispatch and combine kernels.

`MoEMode` selects a runtime context and algorithm family:

- **`LATENCY`** algorithms use broad GPU resources to minimize standalone
  dispatch/combine latency.
- **`THROUGHPUT`** algorithms use a bounded SM budget so communication can run
  concurrently with expert compute.

Mode-specific contexts are allocated conditionally; selecting one family does
not allocate the other family's buffers.

The Python call path is:

```text
MoECommunicator -> LatencyRuntime / ThroughputRuntime -> MoERuntime
                          |
                 passive mode context
```

The context is a passive holder for mode-specific configuration, tensors, and
metadata. `Runtime` owns dispatch/combine and all mode-specific execution
helpers. There is no separate backend or strategy object.

## Status

| Feature | Status |
|---|---|
| Latency dispatch/combine | Validated on Hopper and newer GPUs |
| Throughput dispatch/combine | Supports 2, 4, 8, or 16 ranks in one GPU IPC/NVL fabric domain |
| Throughput RDMA/IB fallback | Not supported |
| Python frontend | `mscclpp.ep.MoECommunicator` selects latency or throughput algorithms with `MoEMode` |
| ROCm | Not supported |

## Runtime architecture

### Latency algorithms

The latency context allocates CUDA physical symmetric memory and maps peer buffers through the
existing `mscclpp::Communicator`. Payloads use direct peer mappings;
`BaseMemoryChannel` handles are used only for synchronization.

The latency algorithms are available when all participating ranks belong to
one detected GPU IPC domain. That domain may span hosts when CUDA fabric handles
and the required fabric services are available.

Latency dispatch supports two user-visible layouts:

- `EXPERT_MAJOR`: one row per `(token, local expert)`.
- `RANK_MAJOR`: fixed-stride rows grouped by source rank. Tokens are written
  directly to registered destination buffers together with dense top-k IDs and
  weights. All three are exposed as zero-copy Torch tensors. Combine can pull
  from registered remote MoE output or push completed rank partials into
  source-local scratch and progressively reduce ready ranks.

Quantized latency dispatch supports E4M3 payloads with FP32 scales per 128 hidden
elements (`FP8_E4M3`).

### Throughput algorithms

The throughput context follows the same direct-mapping model:

1. Python passes the existing `mscclpp::Communicator` into
   `MoERuntime` with `MoEMode::THROUGHPUT`.
2. Each rank allocates a small symmetric control/FIFO region plus a CUDA physical
   internal receive pool. The pool provides stable peer mappings before the
   data-dependent receive count is known; Python later exposes its exact-size
   prefix as the dispatch output.
3. The runtime exchanges and maps those allocations with
   `Communicator::sendMemory` / `recvMemory`.
4. Dispatch and combine launch directly on the caller's CUDA stream.

The detected GPU IPC domain may span multiple hosts, such as an NVL fabric
domain with CUDA fabric handles. The throughput path does not create a private bootstrap, proxy
service, RDMA channel, NVLS multicast object, or private communication stream,
and it has no RDMA/IB fallback outside that domain.

The throughput dispatch API remains two-phase because the receive token count is data
dependent:

1. The notify phase exchanges counts and produces prefix matrices.
2. Python allocates the exact receive tensors.
3. `dispatch` moves token data and metadata.

Cached dispatch reuses the previous receive count and prefix matrices.

## Throughput data path

The throughput family has one direct path. Every dispatch block writes hidden rows and routing
metadata directly into each destination's final receive-pool slots. Combine
stages any out-of-place expert output back into that pool, synchronizes ranks,
then uses a TMA shared-memory pipeline to gather and reduce peer contributions.
There is no ring algorithm or runtime fallback. Set the communication block
budget through the `num_blocks` API configuration.

The persistent throughput configuration contains only:

| Field | Meaning |
|---|---|
| `num_blocks` | Maximum throughput communication block budget |

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

The EP extension requires CUDA architecture 90 or newer. Without an explicit
`MSCCLPP_GPU_ARCHS`, it builds for `90`, `100`, `100a`, `103`, and `103a` when
the CUDA toolkit supports those targets. An explicit `MSCCLPP_GPU_ARCHS`
overrides this list; when it contains no explicit architecture 90 or newer, the
EP extension is skipped.

Available CMake options:

| Variable | Default | Meaning |
|---|---:|---|
| `MSCCLPP_BUILD_EXT_EP` | `ON` | Build the EP extension |

## Source layout

```text
src/ext/ep/
├── bindings.cpp
├── moe_runtime.{cc,hpp}
├── moe_runtime_context.hpp
├── latency.cc
├── throughput.cc
├── common/
│   ├── latency.cuh
│   ├── recv_pool.cuh
│   └── overlap_barrier.cuh
├── dispatch/
│   ├── common.cuh
│   ├── expert_major.cu
│   ├── rank_major.cu
│   ├── token_major_prepare.cu
│   └── token_major.cu
├── combine/
│   ├── common.cuh
│   ├── rank_local_reduce.cu
│   ├── direct_send.cu
│   └── token_major_reduce.cu
├── include/
│   ├── api.cuh
│   ├── device_context.cuh
│   ├── device_helpers.cuh
│   ├── exception.cuh
│   ├── launch.cuh
│   └── quantization.cuh
└── config.hpp
```

## Validation

Build the extension, then run the single-node throughput test:

```bash
HWLOC_COMPONENTS=-gl \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
torchrun --standalone --nproc_per_node=8 \
    test/python/ep/test_intranode_multirank.py
```

The latency validation remains:

```bash
HWLOC_COMPONENTS=-gl \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
torchrun --standalone --nproc_per_node=8 \
    test/python/ep/test_latency_multirank.py
```
