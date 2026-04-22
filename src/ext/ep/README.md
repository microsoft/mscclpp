# MSCCL++ Expert-Parallel (EP) extension

A port of DeepEP's MoE `dispatch` / `combine` primitives into MSCCL++,
targeting:

- **High-Throughput (HT) mode** from [DeepEP](https://github.com/deepseek-ai/DeepEP),
  branch `chhwang/dev-atomic-add-cleanup` — which already swaps NVSHMEM for
  `mscclpp::PortChannel` / `mscclpp::MemoryChannel`.
- **Low-Latency (LL) mode** from [`nccl/contrib/nccl_ep`](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep),
  which implements pure-RDMA dispatch/combine on top of the NCCL Device API.

## Status

| Feature                            | Status                                      |
|------------------------------------|---------------------------------------------|
| `Buffer` construction + IPC + sync | ✅ ported (NVLink + RDMA)                   |
| `get_dispatch_layout`              | ✅ ported                                   |
| `intranode_dispatch` (NVLink)      | ✅ ported, validated (8 ranks, 1 node)     |
| `intranode_combine` (NVLink)       | ✅ ported, validated (8 ranks, 1 node)     |
| `internode_dispatch` (NVLink+RDMA) | ✅ ported, validated (16 ranks, 2×H100×8)  |
| `internode_combine` (NVLink+RDMA)  | ✅ ported, validated (16 ranks, 2×H100×8)  |
| `low_latency_dispatch` (pure RDMA) | ⚠️ structural port, untested                |
| `low_latency_combine` (pure RDMA)  | ⚠️ structural port, untested                |
| `Connection::atomicAdd` API        | ✅ cherry-picked into mscclpp               |
| Python frontend `mscclpp.ext.ep`   | ✅ wraps HT + LL paths                      |
| pybind11 module `mscclpp_ep_cpp`   | ✅ builds conditionally                     |

Internode HT was validated end-to-end on two H100×8 nodes connected over
Infiniband using [`test/python/ext/ep/test_internode_multirank.py`](../../../test/python/ext/ep/test_internode_multirank.py).
All 16 ranks complete dispatch followed by combine with exact (zero-diff)
recovery of the per-rank token payloads.

The low-latency port is **structural**: the DeepEP LL kernels (pure
IBGDA) have been mechanically translated to MSCCL++ port-channel ops.
Semantic mapping:

| DeepEP / IBGDA                           | MSCCL++ replacement                                              |
|------------------------------------------|------------------------------------------------------------------|
| `nvshmemx_barrier_all_block()`           | signal + wait ring across `port_channel_handles[peer_rank]`      |
| `nvshmemi_ibgda_put_nbi_warp(...)`       | lane-0 `port_channel_handles[qp*N+dst].put(dst_off, src_off, n)` |
| `nvshmemi_ibgda_amo_nonfetch_add(...)`   | lane-0 `port_channel_handles[qp*N+dst].atomicAdd(off, int64)`    |

### Known limitations

- LL performance will NOT match IBGDA — the MSCCL++ port channel uses a
  CPU proxy. The port is for functional parity, not latency.
- Unlike DeepEP, this port drives LL dispatch/combine through
  `PortChannel` rather than NVSHMEM, so `Buffer::sync()` connects every
  peer (not just same-GPU-ID peers) even in `low_latency_mode=True`.
- **LL dispatch/combine hangs for intra-node 8-GPU (single host)
  configurations** with the current `PortChannel`-over-IB setup: with
  `num_nvl_bytes=0` every peer-to-peer transfer goes through the CPU
  proxy's IB verbs path, and IB loopback between two distinct HCAs on
  the same host does not deliver atomics reliably. Using `CudaIpc` for
  same-node peers instead surfaces a 64-bit `atomicAdd` vs. 32-bit
  counter alignment mismatch in `CudaIpcConnection::atomicAdd` which
  corrupts adjacent counter slots. A proper fix requires either (a) a
  mixed-transport LL variant that uses `MemoryChannel` (IPC, no proxy)
  for same-node peers like HT does, or (b) widening `rdma_recv_count`
  slots to 64 bits. See [`test/python/ext/ep/test_low_latency_multirank.py`](../../../test/python/ext/ep/test_low_latency_multirank.py).
- H100 cross-node validation of LL mode (1 GPU per node, DeepEP's
  recommended topology) is still pending.
- The internode HT functional test inserts an explicit
  `torch.cuda.synchronize()` + `dist.barrier()` between dispatch and
  combine. Without it, fast ranks can launch combine while peers still
  have in-flight dispatch proxy traffic, deadlocking the combine NVL
  forwarder. Folding this barrier into
  `Buffer::internode_dispatch` / `Buffer::internode_combine` (or
  `cached_notify`) is tracked in the test's `XXX` comment.

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
buf = ep.Buffer(group, num_nvl_bytes=..., num_rdma_bytes=...)
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
├── test_ep_smoke.py                   — size-hint + rejection smoke test
├── test_intranode_multirank.py        — NVLink dispatch+combine, 8 ranks
├── test_internode_multirank.py        — HT dispatch+combine, 16 ranks (2×8)
└── test_low_latency_multirank.py      — LL dispatch+combine (intra-node hang; see limitations)
```

## Running the tests

Intranode (single node, 8 GPUs):

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    test/python/ext/ep/test_intranode_multirank.py
```

Internode HT (2 nodes × 8 GPUs):

```bash
# node 0 (master)
NCCL_SOCKET_IFNAME=eth0 MSCCLPP_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ext/ep/test_internode_multirank.py

# node 1 (worker)
NCCL_SOCKET_IFNAME=eth0 MSCCLPP_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ext/ep/test_internode_multirank.py
```

## Migration plan

### Phase 1 — DONE

- [x] Copy DeepEP kernel headers (configs / buffer / utils / launch / exception).
- [x] Port intranode kernels + runtime (NVLink only).
- [x] Port `get_dispatch_layout` (host-safe subset of internode kernels).
- [x] Port host Buffer: ctor, sync, get_dispatch_layout, intranode
      dispatch/combine.
- [x] pybind11 `mscclpp_ep_cpp` module + Python frontend.

### Phase 2 — internode HT (NVLink + RDMA) — DONE

- [x] Port `notify_dispatch`, `dispatch`, `cached_notify`, `combine` kernels.
- [x] Wire `Buffer::internode_dispatch` / `Buffer::internode_combine` host
      orchestration.
- [x] `Buffer::sync()` builds `port_channel_handles_device_ptr` and
      `memory_channel_handles_device_ptr`, with port channels ordered by
      peer rank (so kernel-side indexing by `peer_rank` is consistent).
- [x] Validated on 2×H100×8 with `test_internode_multirank.py`.

### Phase 3 — Low-Latency (pure RDMA) — STRUCTURAL PORT

Port `DeepEP/csrc/kernels/internode_ll.cu` and cross-reference
`nccl/contrib/nccl_ep/device/low_latency.cu`. The nccl_ep reference is
modular (see `device_primitives.cuh`, `hybrid_ep.cuh`) and uses NCCL
Device API; the translation table is:

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

Bodies of `Buffer::low_latency_dispatch` / `low_latency_combine` have been
translated from DeepEP but are **untested on real hardware**.

### Phase 4 — Validation

- [x] `test_intranode_multirank.py` — NVLink round-trip validated.
- [x] `test_internode_multirank.py` — HT round-trip validated on 2×H100×8.
- [ ] `test_low_latency_multirank.py` — LL round-trip port in place; intra-node 8-GPU hangs (see Known limitations), cross-node (1 GPU / node) pending hardware validation.
- [ ] Throughput benchmarks against DeepEP upstream.
