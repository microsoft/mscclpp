# MSCCL++ Expert-Parallel (EP) extension

A port of DeepEP's MoE `dispatch`/`combine` primitives into MSCCL++, targeting:

- **High-Throughput (HT) mode** from [DeepEP](https://github.com/deepseek-ai/DeepEP),
  branch `chhwang/dev-atomic-add-cleanup` — which already swaps NVSHMEM for
  `mscclpp::PortChannel`/`MemoryChannel`.
- **Low-Latency (LL) mode** from [`nccl/contrib/nccl_ep`](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep),
  which implements pure-RDMA dispatch/combine on top of the NCCL Device API.

## Status

| Feature                            | Status                          |
|------------------------------------|---------------------------------|
| `Buffer` construction + IPC + sync | ✅ ported (NVLink + RDMA)       |
| `get_dispatch_layout`              | ✅ ported                       |
| `intranode_dispatch` (NVLink)      | ✅ ported                       |
| `intranode_combine` (NVLink)       | ✅ ported                       |
| `internode_dispatch` (NVLink+RDMA) | ✅ ported (pending H100 test)   |
| `internode_combine` (NVLink+RDMA)  | ✅ ported (pending H100 test)   |
| `low_latency_dispatch` (pure RDMA) | ⚠️ structural port, untested    |
| `low_latency_combine` (pure RDMA)  | ⚠️ structural port, untested    |
| `Connection::atomicAdd` API        | ✅ cherry-picked into mscclpp   |
| Python frontend `mscclpp.ext.ep`   | ✅ wraps HT + LL paths          |
| pybind11 module `mscclpp_ep_cpp`   | ✅ builds conditionally          |

Internode HT is code-complete but unverified on real hardware — the
`sync()` path replaces DeepEP's NVSHMEM symmetric-heap allocation with
`cudaMalloc` + `bootstrap->barrier()`, and the kernels use the new
`PortChannelDeviceHandle::atomicAdd` instead of the old raw-trigger
pattern.

The low-latency port is **structural**: the DeepEP LL kernels (pure
IBGDA) have been mechanically translated to MSCCL++ port-channel ops.
Semantic mapping:

| DeepEP / IBGDA                           | MSCCL++ replacement                                          |
|------------------------------------------|--------------------------------------------------------------|
| `nvshmemx_barrier_all_block()`           | signal+wait ring across `port_channel_handles[peer_rank]`    |
| `nvshmemi_ibgda_put_nbi_warp(...)`       | lane-0 `port_channel_handles[qp*N+dst].put(dst_off, src_off, n)` |
| `nvshmemi_ibgda_amo_nonfetch_add(...)`   | lane-0 `port_channel_handles[qp*N+dst].atomicAdd(off, int64)` |

**Known limitations**:

* LL performance will NOT match IBGDA — the MSCCL++ port channel uses a
  CPU proxy. The port is for functional parity, not latency.
* `Buffer::sync()` in `low_latency_mode=True` only connects peers sharing
  the same local GPU ID (DeepEP convention). LL kernels therefore assume
  one-GPU-per-node topology, i.e. `num_ranks == num_rdma_ranks`. Running
  with >1 GPU per node in LL mode will fail to reach cross-GPU peers.
* Multi-node H100 validation is still pending.

## Build

The extension is **off by default** and requires PyTorch's CMake package:

```bash
TORCH_CMAKE=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build \
      -DMSCCLPP_BUILD_EXT_EP=ON \
      -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}"
cmake --build build -j
```

This produces `mscclpp_ep_cpp.so` — a pybind11 PyTorch extension module.
The Python frontend picks it up automatically:

```python
from mscclpp.ext import ep
buf = ep.Buffer(group, num_nvl_bytes=..., num_rdma_bytes=0)
```

## Layout

```
src/ext/ep/
├── CMakeLists.txt              — builds mscclpp_ep_cpp (Torch + pybind11)
├── buffer.hpp / buffer.cc      — host-side Buffer, sync(), dispatch/combine
├── config.hpp / event.hpp      — Config, EventHandle
├── bindings.cpp                — PYBIND11_MODULE definition
└── kernels/
    ├── api.cuh                 — host-callable kernel prototypes
    ├── configs.cuh             — compile-time constants (GPU-only)
    ├── buffer.cuh              — Buffer/AsymBuffer/SymBuffer helpers
    ├── exception.cuh           — EP_HOST/DEVICE_ASSERT + CUDA_CHECK
    ├── launch.cuh              — SETUP_LAUNCH_CONFIG / SWITCH_* macros
    ├── utils.cuh               — device inline helpers
    ├── runtime.cu              — intranode::barrier launcher
    ├── intranode_kernel.cu     — intranode dispatch/combine kernels
    ├── internode.cu            — internode HT dispatch/combine + layout
    └── internode_ll.cu         — internode LL dispatch/combine (structural)

python/mscclpp/ext/ep/
├── __init__.py                 — reexports Buffer / Config / EventHandle
└── buffer.py                   — torch.distributed-aware frontend

test/python/ext/ep/
└── test_ep_smoke.py            — size-hint + rejection smoke test
```

## Migration plan

### Phase 1 — DONE

- [x] Copy DeepEP kernel headers (configs / buffer / utils / launch / exception).
- [x] Port intranode kernels + runtime (NVLink only).
- [x] Port `get_dispatch_layout` (host-safe subset of internode kernels).
- [x] Port host Buffer: ctor, sync, get_dispatch_layout, intranode dispatch/combine.
- [x] pybind11 `mscclpp_ep_cpp` module + Python frontend.

### Phase 2 — internode HT (NVLink + RDMA)

Port the rest of `DeepEP/csrc/kernels/internode.cu` (`notify_dispatch`,
`dispatch`, `cached_notify`, `combine`). Because we are starting from the
`chhwang/dev-atomic-add-cleanup` branch, the NVSHMEM -> MSCCL++ substitution
is already done upstream — just copy the kernel bodies and wire them through
`api.cuh`. The launchers need `PortChannelDeviceHandle*` /
`MemoryChannelDeviceHandle*` arguments that `Buffer::sync()` already builds
(see the `num_rdma_bytes > 0` branch — currently throws, but the code
populating `port_channel_handles_device_ptr` and
`memory_channel_handles_device_ptr` is ready). Finally replace the stubs in
`buffer.cc` (`internode_dispatch`, `internode_combine`) with the real bodies
from DeepEP.

### Phase 3 — Low-Latency (pure RDMA)

Port `DeepEP/csrc/kernels/internode_ll.cu` and cross-reference
`nccl/contrib/nccl_ep/device/low_latency.cu`. The nccl_ep reference is
modular (see `device_primitives.cuh`, `hybrid_ep.cuh`) and uses NCCL Device
API; the translation table is:

| nccl_ep / DeepEP primitive              | MSCCL++ replacement                             |
|-----------------------------------------|-------------------------------------------------|
| `nvshmemi_ibgda_put_nbi_warp`           | `PortChannelDeviceHandle::put` + `signal`       |
| `nvshmem_signal_wait_until`             | `PortChannelDeviceHandle::wait`                 |
| `ncclGinPutSignal`                      | same as above                                   |
| `ncclGinWaitSignal`                     | `PortChannelDeviceHandle::wait`                 |
| `ncclGetPeerPointer` / IPC              | offset into `buffer_ptrs_gpu[peer]`             |
| `ncclTeamLsa` locality check            | per-rank `rank / NUM_MAX_NVL_PEERS` comparison  |
| NVSHMEM symmetric heap                  | `cudaMalloc` + `ProxyService::addMemory`        |
| NVSHMEM barrier                         | `bootstrap->barrier()` or `intranode::barrier`  |

Finally fill in `buffer.cc::low_latency_dispatch` / `low_latency_combine`
from the DeepEP bodies (already translated on the `chhwang/...` branch).

### Phase 4 — Validation

- Port `DeepEP/tests/test_{intranode,internode,low_latency}.py` into
  `test/python/ext/ep/`.
- Run on the same H100/H800 reference rig DeepEP uses; compare throughput.
# MSCCL++ Expert-Parallel (EP) extension — migration plan

This directory is a **scaffolding-only** port of the Mixture-of-Experts (MoE)
`dispatch` / `combine` primitives from:

- **High-Throughput (HT) mode** — [DeepEP](https://github.com/deepseek-ai/DeepEP),
  branch `chhwang/dev-atomic-add-cleanup`. That branch has already replaced
  NVSHMEM / IBGDA primitives with `mscclpp::PortChannel` and
  `mscclpp::MemoryChannel`, so the port is largely mechanical.
- **Low-Latency (LL) mode** — [`nccl/contrib/nccl_ep`](https://github.com/NVIDIA/nccl),
  which implements a pure-RDMA dispatch/combine on top of the NCCL Device API
  (GIN put/signal + LSA load/store). The kernels need to be re-expressed in
  terms of MSCCL++ device handles.

## Layout

| Path | Purpose |
|------|---------|
| [`include/mscclpp/ext/ep/config.hpp`](../../../include/mscclpp/ext/ep/config.hpp) | Public host-side config + size hints (`EpConfig`). |
| [`include/mscclpp/ext/ep/event.hpp`](../../../include/mscclpp/ext/ep/event.hpp)   | RAII wrapper around `cudaEvent_t`. |
| [`include/mscclpp/ext/ep/buffer.hpp`](../../../include/mscclpp/ext/ep/buffer.hpp) | Public `Buffer` class; dispatch/combine entry points. |
| [`include/mscclpp/ext/ep/api.hpp`](../../../include/mscclpp/ext/ep/api.hpp)       | Umbrella include. |
| [`src/ext/ep/buffer.cc`](buffer.cc)                 | Host-side orchestration. Constructor + proxy service wired up; `sync()` / kernel stubs `TODO`. |
| [`src/ext/ep/config.cc`](config.cc)                 | `EpConfig` method bodies. |
| [`src/ext/ep/event.cc`](event.cc)                   | `EventHandle` implementation. |
| [`src/ext/ep/intranode.cu`](intranode.cu)           | **STUB** — HT NVLink-only dispatch/combine. |
| [`src/ext/ep/internode.cu`](internode.cu)           | **STUB** — HT NVLink+RDMA dispatch/combine. |
| [`src/ext/ep/internode_ll.cu`](internode_ll.cu)    | **STUB** — LL pure-RDMA dispatch/combine. |
| [`src/ext/ep/kernels/api.cuh`](kernels/api.cuh)    | Private kernel-facing API (prototypes only for now). |
| [`src/ext/ep/kernels/exception.cuh`](kernels/exception.cuh) | `EP_HOST_ASSERT` / `EP_DEVICE_ASSERT` / `EP_CUDA_CHECK`. |
| [`python/csrc/ext/ep/ep_py.cpp`](../../../python/csrc/ext/ep/ep_py.cpp) | nanobind bindings (submodule `mscclpp._mscclpp.ep`). |
| [`python/mscclpp/ext/ep/`](../../../python/mscclpp/ext/ep/) | Python frontend (`ep.Buffer`). |
| [`test/python/ext/ep/test_ep_skeleton.py`](../../../test/python/ext/ep/test_ep_skeleton.py) | Unit test placeholder. |

## Build

The extension is **off by default**. Enable it with:

```bash
cmake -S . -B build -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build -j
```

This produces `libmscclpp_ep.so` and, when Python bindings are built, exposes
`mscclpp._mscclpp.ep` and `mscclpp.ext.ep`.

## Migration plan (in order)

1. **HT intranode.** Port `DeepEP/csrc/kernels/intranode.cu` into
   [`intranode.cu`](intranode.cu). All communication is via peer IPC pointers,
   so only `include` paths and `torch::Tensor` -> `TensorRef` marshalling need
   to change. Flesh out `Buffer::sync()` so that
   `nvlBufferPeers_[peer] = cudaIpcOpenMemHandle(...)` is populated and the
   table is uploaded to `nvlBufferPeersDevice_`.
2. **HT internode.** Port `DeepEP/csrc/kernels/internode.cu` into
   [`internode.cu`](internode.cu). Most of the heavy lifting (NVSHMEM ->
   MSCCL++) is already done on the DeepEP `chhwang/dev-atomic-add-cleanup`
   branch; copy the kernel bodies as-is and add the launchers. Ensure the
   custom trigger type `0x0` atomicAdd path in `EpProxyService` (see
   [`buffer.cc`](buffer.cc)) is in place.
3. **LL mode.** Port from `nccl/contrib/nccl_ep/device/low_latency.cu` (or
   DeepEP `internode_ll.cu`) into [`internode_ll.cu`](internode_ll.cu). The
   translation table lives in the file header; the critical substitution is
   `ncclGinPutSignal` / `nvshmemi_ibgda_*` -> `PortChannelDeviceHandle::put`
   + `signal` + `wait`, and `ncclGetPeerPointer` -> the `nvlBufferPeersDevice_`
   offset table.
4. **TensorRef marshalling.** Extend [`ep_py.cpp`](../../../python/csrc/ext/ep/ep_py.cpp)
   to accept DLPack / `torch.Tensor` for the dispatch/combine entry points.
   The `TensorRef` type in [`buffer.hpp`](../../../include/mscclpp/ext/ep/buffer.hpp)
   is intentionally Torch-free so the C++ core can be reused from
   non-PyTorch callers.
5. **Tests.** Grow
   [`test/python/ext/ep/`](../../../test/python/ext/ep/) by porting the
   scenarios from `DeepEP/tests/test_{intranode,internode,low_latency}.py`.

## API mapping cheatsheet

| DeepEP / nccl_ep primitive                         | MSCCL++ replacement                                       |
|----------------------------------------------------|-----------------------------------------------------------|
| `nvshmemi_ibgda_put_nbi_warp`                      | `PortChannelDeviceHandle::put` + `signal`                 |
| `nvshmem_signal_wait_until`                        | `PortChannelDeviceHandle::wait`                           |
| `ncclGinPutSignal`                                 | `PortChannelDeviceHandle::put` + `signal`                 |
| `ncclGinWaitSignal`                                | `PortChannelDeviceHandle::wait`                           |
| `ncclGetPeerPointer` / `cudaIpcOpenMemHandle`      | `Buffer::nvlBufferPeersDevice_[peer]` + byte offset       |
| `ncclTeamLsa` locality check                       | `Buffer::numNvlRanks_` per-rdma-rank group                |
| NVSHMEM symmetric heap allocation                  | `cudaMalloc` + proxy-registered memory (`ProxyService`)   |
| NVSHMEM barrier                                    | `bootstrap_->barrier()` or `intranode::barrier` kernel    |

## Status

- Headers, CMake targets, Python bindings, and the frontend compile (build
  verification has not been run in this session).
- All dispatch/combine entry points throw from C++ or raise `NotImplementedError`
  from Python. Constructor, proxy-service startup, and buffer-size hints are
  real; `sync()` only flips the `available_` flag and does **not** yet open
  peer IPC handles or build MSCCL++ connections.
