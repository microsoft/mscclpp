# MSCCL++ Expert-Parallel (EP) extension

The EP extension is a torch-free nanobind module for MoE dispatch and combine.
It builds two backends:

- **Low latency (LL)**: `MoERuntime`, backed by
  `low_latency/{dispatch,combine}.cu`.
- **High throughput (HT)**: `ExpertParallelRuntime`, backed by
  `ht_runtime.cc` and the CUDA sources under `high-throughput/`.

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
- `RANK_MAJOR`: fixed-stride rows grouped by source rank. Tokens are written
  directly to registered destination buffers together with dense top-k IDs and
  weights. All three are exposed as zero-copy Torch tensors. Combine reads
  registered remote MoE output buffers and reduces the selected rank-local
  partials.

LL quantized dispatch supports E4M3 payloads with FP32 scales per 128 hidden
elements (`FP8_E4M3`) or UE8M0 scale bytes per 32 elements (`MXFP8_E4M3`).
MXFP8 scales use linear row-major layout and can be passed to FlashInfer
`cutlass_fused_moe` with `swizzled_input_sf=False`.

### High throughput

HT follows the same direct-mapping resource model:

1. Python passes the existing `mscclpp::Communicator` into
   `ExpertParallelRuntime`.
2. Each rank allocates a small symmetric control/FIFO region plus a CUDA physical
   internal receive pool. The pool provides stable peer mappings before the
   data-dependent receive count is known; Python later exposes its exact-size
   prefix as the dispatch output.
3. The runtime exchanges and maps those allocations with
   `Communicator::sendMemory` / `recvMemory`.
4. Dispatch and combine launch directly on the caller's CUDA stream.

The detected GPU IPC domain may span multiple hosts, such as an NVL fabric
domain with CUDA fabric handles. HT does not create a private bootstrap, proxy
service, RDMA channel, NVLS multicast object, or private communication stream,
and it has no RDMA/IB fallback outside that domain.

The HT dispatch API remains two-phase because the receive token count is data
dependent:

1. The notify phase exchanges counts and produces prefix matrices.
2. Python allocates the exact receive tensors.
3. `dispatch` moves token data and metadata.

Cached dispatch reuses the previous receive count and prefix matrices.

## HT data path

HT has one direct path. Every dispatch block writes hidden rows and routing
metadata directly into each destination's final receive-pool slots. Combine
stages any out-of-place expert output back into that pool, synchronizes ranks,
then uses a TMA shared-memory pipeline to gather and reduce peer contributions.
There is no ring algorithm or runtime fallback. Set the communication block
budget through the `num_sms` API configuration.

The persistent HT configuration contains only:

| Field | Meaning |
|---|---|
| `num_sms` | Maximum HT communication block budget |

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
overrides this list. Architecture-specific `100a`/`103a` builds use the native
Blackwell UE8M0 conversion instruction; generic `100`/`103` builds use the
portable conversion path.

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
├── high-throughput/
│   ├── config.cuh
│   ├── layout.cu
│   ├── dispatch.cu
│   ├── combine.cu
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
