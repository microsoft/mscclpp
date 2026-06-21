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
| `Buffer` construction + IPC + sync | ✅ ported (NVLink + RDMA)                          |
| `get_dispatch_layout`              | ✅ ported                                          |
| `intranode_dispatch` (NVLink)      | ✅ validated (8 ranks H100; 4 ranks GB200)         |
| `intranode_combine` (NVLink)       | ✅ validated (8 ranks H100; 4 ranks GB200)         |
| `internode_dispatch` (NVLink+RDMA) | ✅ validated (16 ranks 2×H100×8; 64 ranks 16×GB200) |
| `internode_combine` (NVLink+RDMA)  | ✅ validated (16 ranks 2×H100×8; 64 ranks 16×GB200) |
| `low_latency_dispatch` (RDMA+IPC)  | ✅ validated (8 ranks H100; 16 ranks 2×H100×8; 64 ranks 16×GB200 via NVLS fabric IPC) |
| `low_latency_combine` (RDMA+IPC)   | ✅ validated (8 ranks H100; 16 ranks 2×H100×8; 64 ranks 16×GB200 via NVLS fabric IPC) |
| GB200 NVLS multimem fast path      | ✅ runtime-gated by `mscclpp::isNvlsSupported()`    |
| Multi-`ProxyService` sharding      | ✅ env-tunable, arch-aware default                 |
| `Connection::atomicAdd` API        | ✅ cherry-picked into mscclpp                      |
| Python frontend `mscclpp.ext.ep`   | ✅ wraps HT + LL paths                             |
| pybind11 module `mscclpp_ep_cpp`   | ✅ builds conditionally                            |

Internode HT was validated end-to-end on two H100×8 nodes connected over
Infiniband using [`test/python/ext/ep/test_internode_multirank.py`](../../../test/python/ext/ep/test_internode_multirank.py).
All 16 ranks complete dispatch followed by combine with exact (zero-diff)
recovery of the per-rank token payloads.

On Azure GB200 NVL72 (4 GPUs / NUMA host, CX-7 RoCE), HT and LL were
validated at 16 nodes × 4 GPUs = **64 ranks** with HIDDEN=7168,
tokens=4096, experts=256, top-k=8:

- HT internode: dispatch ~**2 006 GB/s** agg, combine ~**2 011 GB/s** agg
  (`NVL_SEND=8 NVL_RECV=256 RDMA_SEND=8 RDMA_RECV=32`).
- LL internode: dispatch ~**16 817 GB/s** agg (~262 GB/s per rank),
  combine ~**21 148 GB/s** agg (defaults).

The GB200 path bypasses Azure CX-7 RoCE's broken `IBV_ATOMIC_*` by
routing peer pointers through cuMem fabric IPC over the NVL72 fabric
(`nvidia-imex`) and emitting NVLink-SHARP `multimem.*` atomics from the
kernels. The legacy RDMA-atomic PortChannel path is retained as a
fallback when `mscclpp::isNvlsSupported()` returns `false`.

The low-latency (LL) path uses a mixed transport: `MemoryChannel` (CUDA
IPC) for same-node peers and `PortChannel` (CPU proxy + IB verbs) for
remote peers. On GB200 NVL72, cross-node peers are *also* reached via
CUDA-IPC — peer pointers are exchanged as cuMem fabric handles over
`nvidia-imex` and the kernels emit NVLink-SHARP `multimem.*` atomics
directly on the NVL72 fabric, bypassing CX-7 RoCE's broken IB atomics.
The DeepEP LL kernels were translated as follows:

| DeepEP / IBGDA                           | MSCCL++ replacement                                              |
|------------------------------------------|------------------------------------------------------------------|
| `nvshmemx_barrier_all_block()`           | signal + wait ring across per-peer channel handles               |
| `nvshmemi_ibgda_put_nbi_warp(...)` (intra-node) | `MemoryChannelDeviceHandle::put` (CUDA IPC, no proxy)     |
| `nvshmemi_ibgda_put_nbi_warp(...)` (inter-node, NVLS) | direct `st.global` on imported cuMem-fabric peer base (GB200) |
| `nvshmemi_ibgda_put_nbi_warp(...)` (inter-node, fallback) | lane-0 `PortChannelDeviceHandle::put(dst_off, src_off, n)` |
| `nvshmemi_ibgda_amo_nonfetch_add(...)` (NVLS) | `multimem.red.add.u64` on NVL72 multicast counter (GB200) |
| `nvshmemi_ibgda_amo_nonfetch_add(...)` (fallback) | lane-0 `atomicAdd` on the corresponding channel handle           |

LL was validated on:
- 8 ranks × 1 H100 node (NVLink + CUDA-IPC fast path).
- 16 ranks × 2 H100×8 nodes (mixed CUDA-IPC intra-node + IB inter-node).
- 4 ranks × 1 Azure GB200 NVL72 node (NVLink + CUDA-IPC).
- 64 ranks × 16 Azure GB200 NVL72 nodes (intra-node CUDA-IPC + cross-node
  cuMem fabric IPC via `nvidia-imex` + NVLS `multimem.*` atomics);
  dispatch ~16.8 TB/s agg, combine ~21.1 TB/s agg.

### `num_proxy_services` / proxy sharding

A single `mscclpp::ProxyService` is one CPU host thread driving one
FIFO. With 8 GPUs / node sharing one proxy, the host thread becomes the
bottleneck for cross-node LL traffic. `Buffer` therefore allocates `N`
ProxyServices and shards `PortChannel`s across them by `(qp_idx,
dst_rank)`.

- Default: `8` on Hopper (sm_90), `1` on Blackwell / sm_100+ (NVSwitch).
- Override at runtime: `MSCCLPP_EP_NUM_PROXIES=<N>` (clamped to ≥1).
  Rank 0 prints the resolved value at construction.
- Sweet spot on 2×H100×8 is `N=8`; `N=12` over-subscribes the host CPUs
  and collapses throughput.

### Known limitations

- LL performance will NOT match IBGDA for cross-node traffic — remote
  peers go through a CPU proxy. The port is for functional parity, not
  latency. (Intra-node LL traffic uses CUDA IPC and is competitive.)
- Unlike DeepEP, this port drives LL through `PortChannel` /
  `MemoryChannel` rather than NVSHMEM, so `Buffer::sync()` connects
  every peer even in `low_latency_mode=True`.
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

### Build-time CMake options

| Variable                              | Default | Meaning                                                       |
|---------------------------------------|---------|---------------------------------------------------------------|
| `MSCCLPP_BUILD_EXT_EP`                | `OFF`   | Build the EP extension at all                                 |
| `MSCCLPP_EP_NUM_MAX_NVL_PEERS`        | `8`     | Compile-time `NUM_MAX_NVL_PEERS` — set to `4` for GB200 NVL72 |
| `MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT`     | `OFF`   | Use a short ~10s kernel spin timeout (default is ~100s)       |

### Azure GB200 (NVL72, 4 GPUs / NUMA host)

GB200 NVL72 nodes expose **4 GPUs per NUMA host** (not 8 like HGX
H100), and the cross-node atomic fast-path uses NVLink-SHARP
multicast (`multimem.*` PTX) routed over the NVL72 fabric via
nvidia-imex instead of broken IB atomics on Azure CX-7 RoCE. Two
build-time settings are required:

```bash
# Option 1: plain CMake.
cmake -S . -B build \
      -DMSCCLPP_BUILD_EXT_EP=ON \
      -DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4 \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j

# Option 2: wheel-based install (pyproject.toml already sets
# MSCCLPP_BUILD_EXT_EP=ON).
CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
python3 -m pip install --no-build-isolation \
    --config-settings=cmake.define.MSCCLPP_EP_NUM_MAX_NVL_PEERS=4 \
    .
```

Add `-DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT=ON` only when triaging hangs (it
shortens the kernel-side spin timeout from ~100s to ~10s).

Runtime prerequisites on GB200:

- CUDA Toolkit ≥ 12.5 (the `cuCtxCreate` proxy-context path uses the
  4-arg signature added in 12.5; older toolkits compile against the
  3-arg fallback automatically).
- Driver ≥ 555 with nvidia-imex configured so cuMem fabric handles
  (`POSIX_FD | FABRIC`) can be exchanged across nodes.
- NVLink-SHARP / multicast support enabled (`nvidia-smi mig … --imex`
  reachable). `mscclpp::isNvlsSupported()` must return `true` at
  Buffer construction; otherwise the kernels fall back to the legacy
  PortChannel + RDMA path (and on Azure CX-7 RoCE the broken IB
  atomics will hang).
- `MSCCLPP_EP_LOCAL_WORLD_SIZE` partitions ranks into NUMA hosts; it
  defaults to the build-time `NUM_MAX_NVL_PEERS`, so a GB200 build
  (`-DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4`) auto-uses 4 and **does not
  require** setting this env var. Only set `MSCCLPP_EP_LOCAL_WORLD_SIZE=4`
  if you are running on GB200 against a stock build that still has
  `NUM_MAX_NVL_PEERS=8` (otherwise host code mis-classifies cross-node
  peers as local and `cudaIpcOpenMemHandle` fails).
- RT priority is required by NCCL/glibc. On each node:

  ```bash
  sudo tee -a /etc/security/limits.conf > /dev/null <<'EOF'
  * soft rtprio 99
  * hard rtprio 99
  EOF
  # Re-login so `ulimit -r` reports 99.
  ```
- `nvidia-imex` must be active on every node with an identical
  `/etc/nvidia-imex/nodes_config.cfg` listing all node IPs. Verify:

  ```bash
  sudo systemctl status nvidia-imex
  sudo cat /etc/nvidia-imex/nodes_config.cfg
  ls /dev/nvidia-caps-imex-channels/   # channel0 must exist
  ```

GB200 runtime env (export on every node before launching the tests):

```bash
export NCCL_IB_DISABLE=1                              # mscclpp owns IB
export NCCL_MNNVL_ENABLE=0                            # Azure GB200 is NOT one MNNVL fabric across nodes
export MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # avoid mlx5_bond_0 (PORT_DOWN)
# Bootstrap NIC (only if auto-detect picks wrong) — on Azure GB200:
export NCCL_SOCKET_IFNAME=enP22p1s0f1
export MSCCLPP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
```

Runtime knobs (env vars, exposed by `test_intranode_multirank.py` /
`test_internode_multirank.py` — defaults below are the test-script
defaults, **not** the `ep.Config(...)` constructor defaults):

| Variable                  | Maps to (`ep.Config` field)         | Default | Notes                                  |
|---------------------------|--------------------------------------|--------:|-----------------------------------------|
| `MSCCLPP_EP_NSM`          | `num_sms`                            | `152` / `20` | Single SM-count knob for **both** tests (internode default `152`, intranode `20`). `MSCCLPP_EP_NUM_SMS` is still accepted as a legacy alias. Try `64` on GB200 intranode for `dispatch` BW. |
| `MSCCLPP_EP_NVL_SEND`     | `num_max_nvl_chunked_send_tokens`    | `8`     | Must be `<` `MSCCLPP_EP_NVL_RECV`.     |
| `MSCCLPP_EP_NVL_RECV`     | `num_max_nvl_chunked_recv_tokens`    | `256`   | Scales NVL ring buffer linearly.       |
| `MSCCLPP_EP_RDMA_SEND`    | `num_max_rdma_chunked_send_tokens`   | `16`    | Internode only.                        |
| `MSCCLPP_EP_RDMA_RECV`    | `num_max_rdma_chunked_recv_tokens`   | `128`   | Scale **down** as `num_rdma_ranks` grows (4n→128, 8n→64, 16n→32) to keep the RDMA buffer under the 2 GiB `INT_MAX` limit. |
| `MSCCLPP_EP_DIRECT`       | — (runtime `getenv`)                 | unset   | **GB200 / NVL72 internode only.** `1` enables the sender direct-write dispatch **and** receiver gather-direct combine (see [GB200 direct-path optimization](#gb200-direct-path-optimization-mscclpp_ep_direct--mscclpp_ep_intra_direct)). Unset = byte-identical 2-hop baseline. |
| `MSCCLPP_EP_INTRA_DIRECT` | — (runtime `getenv`)                 | unset   | **GB200 single-node only.** `1` enables sender direct-write for the intra-node kernel. Independent of `MSCCLPP_EP_DIRECT` (see below). |
| `MSCCLPP_EP_FLAT`         | — (runtime `getenv`)                 | unset   | **GB200 / NVL72 internode only; requires `MSCCLPP_EP_DIRECT=1`.** `1` enables the flat all-sender dispatch (removes the forwarder / coordinator / receiver roles; per-token metadata goes straight to the dest recv pool) + direct-gather combine (see [flat all-sender path](#gb200-flat-all-sender-path-mscclpp_ep_flat)). Unset = the `MSCCLPP_EP_DIRECT` path. |
| `MSCCLPP_EP_DISPATCH_NSM` | — (runtime `getenv`)                 | `num_sms`/2 | **Flat path only (`MSCCLPP_EP_FLAT=1`).** Sets the all-sender dispatch block count independently of `num_sms`; clamped to `[1, num_sms]` (the flat path has no forwarder, so it can use the full SM budget — e.g. `num_sms=16` + `DISPATCH_NSM=16` → 16 blocks). The RDMA/NVL buffers are auto-sized to match. Lower it to free SMs for overlapping compute, or set it to `num_sms` to maximize dispatch throughput. |
| `MSCCLPP_EP_COMBINE_NSM`  | — (runtime `getenv`)                 | `num_sms`   | **Flat path only (`MSCCLPP_EP_FLAT=1`).** Caps the combine block count independently of `num_sms`; clamped to `[2, num_sms]`. Flat combine saturates ~76 blocks. |
| `MSCCLPP_EP_COMBINE_TMA`  | — (runtime `getenv`)                 | `1` (on) | **Flat path only (`MSCCLPP_EP_FLAT=1` + `MSCCLPP_EP_DIRECT=1`).** Selects the combine flat-gather implementation. Default (`1`) uses the TMA-staged gather (`cp.async.bulk` contributor rows → SMEM → reduce), which hides remote-NVLink read latency via the async copy engine and wins at every channel count and node scale. `0` falls back to the synchronous register-MLP gather (lean kernel ≤14 channels, else the unified flat branch). |

Validated 16-node (64-rank) configs on Azure GB200 NVL72 (HIDDEN=7168,
tokens=4096, experts=256, topk=8):

- HT internode: `NVL_SEND=8 NVL_RECV=256 RDMA_SEND=8 RDMA_RECV=32` →
  dispatch ~**2 006 GB/s** agg, combine ~**2 011 GB/s** agg.
- LL internode: defaults → dispatch ~**16 817 GB/s** agg
  (262 GB/s per rank), combine ~**21 148 GB/s** agg.

### GB200 direct-path optimization (`MSCCLPP_EP_DIRECT` / `MSCCLPP_EP_INTRA_DIRECT`)

On GB200 NVL72 the cross-node "RDMA send" is a cuMem-fabric VA write over
NVLink, so a sender can reach any peer GPU's recv pool directly. Two
env-gated flags exploit this to remove the classic DeepEP 2-hop (forwarder
transpose on dispatch / ring drain on intra-node), turning a
structural-/handshake-bound copy into a bandwidth-bound 1-hop that scales
with SM count. Both default **off**, and the binary is byte-identical to the
2-hop baseline when unset (one `.so`, runtime `getenv`-gated).

- **`MSCCLPP_EP_DIRECT=1`** — internode HT (`test_internode_multirank.py`).
  One master flag that turns on **both**:
  - *Dispatch sender direct-write:* `kRDMASender` writes each token's hidden
    straight to `recv_pool_global_ptrs[dst] + header + idx*hidden_bytes`
    (1 hop), instead of the forwarder transpose to local NVL peers (2 hops).
  - *Combine receiver gather-direct:* each token gathers its top-k expert
    contributions directly from the peer recv pools and reduces locally,
    skipping the `nvl_channel + forwarder + rdma_channel` path.
  - *Prerequisite:* cross-node fabric-IPC pool mapping — auto-detected on
    GB200; force with `MSCCLPP_EP_FABRIC_IPC=1` if needed. The combine input
    must live in the recv pool (DeepEP contract; satisfied by the round-trip,
    where combine input = dispatch output).
- **`MSCCLPP_EP_INTRA_DIRECT=1`** — single-node
  (`test_intranode_multirank.py`), a **separate** flag for the intra-node
  kernel. The sender writes hidden into the destination GPU's peer-mapped
  pool and the receiver skips the ring drain.

Measured on Azure GB200 NVL72 (2 nodes × 4 GPU, HIDDEN=7168, tokens=4096,
topk=8, experts=256):

- HT internode dispatch+combine round-trip (NSM=20): **~3 850 µs → ~1 750 µs
  (−54 %)** vs the 2-hop baseline; the win grows with node count.
- HT intra-node `INTRA_DIRECT=1` dispatch scales **980 µs (16 SM) → 285 µs
  (152 SM, ~206 GB/s per rank)**, whereas the 2-hop baseline is SM-flat at
  ~3.8 ms.

> **Single-node launch note.** Launch the intra-node test with an explicit
> `127.0.0.1` rendezvous (not `torchrun --standalone`, whose hostname
> rendezvous is not DNS-resolvable on these nodes) and set
> `NCCL_NET_PLUGIN=none` + `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3` so NCCL's
> built-in IB probe does not crash `ep.Buffer` construction.

### GB200 flat all-sender path (`MSCCLPP_EP_FLAT`)

`MSCCLPP_EP_FLAT=1` (requires `MSCCLPP_EP_DIRECT=1`) takes the direct path one
step further. The `MSCCLPP_EP_DIRECT` dispatch still runs a sender + forwarder
pair per channel and routes per-token metadata through the 2-hop RDMA ring →
forwarder → NVL-receiver pipeline; the flat path removes the forwarder,
sender-coordinator, and NVL-receiver roles entirely, so dispatch launches
**all-sender** (one block per channel) and each sender writes its tokens'
metadata straight into the destination recv pool. Combine is a flat
direct-gather (no forwarder); by default it uses a **TMA-staged** gather that
streams each contributor's hidden chunks through shared memory via `cp.async.bulk`
(`MSCCLPP_EP_COMBINE_TMA=1`, the default), letting the async copy engine hide the
remote-NVLink read latency. A small post-dispatch drain copies the pool metadata
into the `recv_*` output tensors.

Because both legs are now bandwidth-bound 1-hops, they saturate the NVLink
write/read ceiling well below the full SM grid, so the dispatch and combine
block counts can be capped **independently** of `num_sms` to free SMs for
overlapping compute:

- `MSCCLPP_EP_DISPATCH_NSM=<N>` — flat dispatch block count, clamped to
  `[1, num_sms]` (the flat path has no forwarder, so it can use the full SM
  budget; the RDMA/NVL buffers are auto-sized to the chosen count).
  Default = `num_sms/2`.
- `MSCCLPP_EP_COMBINE_NSM=<N>` — flat combine block count, clamped to
  `[2, num_sms]`. Default = `num_sms`.

Both are flat-only and leave the `MSCCLPP_EP_DIRECT` / 2-hop paths byte-
identical when unset. Measured on Azure GB200 NVL72 (2 nodes × 4 GPU,
HIDDEN=7168, tokens=4096, topk=8, experts=256, `MSCCLPP_EP_NSM=152`):

- Dispatch (combine grid fixed): `DISPATCH_NSM` 16→**698 µs**, 32→544,
  64→498, 76→**495 µs** — knee at ~64 blocks.
- Combine (dispatch grid fixed): `COMBINE_NSM` 16→**1006 µs**, 32→577,
  64→514, 152→**451 µs** — knee at ~76 blocks.

So flat dispatch reaches its floor at ~64 of 76 possible blocks and combine at
~76 of 152, leaving the remaining SMs free for the model's compute to overlap
the communication.

## Layout

```
src/ext/ep/
├── CMakeLists.txt              — builds mscclpp_ep_cpp (Torch + pybind11)
├── README.md                   — this file
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
    │                            (incl. NVLS multimem fast path for GB200)
    └── internode_ll.cu         — internode LL dispatch/combine

python/mscclpp/ext/ep/
├── __init__.py                 — reexports Buffer / Config / EventHandle
└── buffer.py                   — torch.distributed-aware frontend

test/python/ext/ep/
├── test_intranode_multirank.py        — intranode HT dispatch+combine
├── test_internode_multirank.py        — internode HT dispatch+combine
└── test_low_latency_multirank.py      — LL dispatch+combine
```

## Running the tests

### Test prerequisites

The Python tests are launched through `torchrun` / `mpirun` and require
PyTorch + a few support packages in the active environment. A minimal
install (matches the GB200 reference setup):

```bash
# Conda env (any Python >= 3.10). Use the appropriate Miniconda variant
# for the host arch (`aarch64` shown; use `x86_64` on x86 clusters).
wget -O /tmp/Miniconda3-latest-Linux-aarch64.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash /tmp/Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n torch python=3.14 -y
conda activate torch

# Runtime libs used by the tests / launcher.
conda install -c conda-forge -y cupy mpi4py pybind11 blake3 sortedcontainers

# PyTorch (pulls a matching cuda-toolkit + NCCL).
pip3 install torch

# mscclpp build deps (used by `pip install .` of this repo).
pip install scikit-build-core nanobind setuptools_scm
```

Then build the EP extension (see [Build](#build)) — `pip install .` from
the repo root installs `mscclpp_ep_cpp.so` into the active env so the
test scripts can `from mscclpp.ext import ep`.

Intranode (single node, 8 GPUs) — HT:

```bash
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=1 --nproc_per_node=8 \
    test/python/ext/ep/test_intranode_multirank.py
```

Intranode LL (single node, 8 GPUs):

```bash
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=128 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=1 --nproc_per_node=8 \
    test/python/ext/ep/test_low_latency_multirank.py
```

Internode HT (2 nodes × 8 GPUs), torchrun:

```bash
# node 0 (master)
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ext/ep/test_internode_multirank.py

# node 1 (worker)
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ext/ep/test_internode_multirank.py
```

If the bootstrap NIC is mis-detected (e.g. multi-homed hosts), pin
it explicitly:

```bash
export NCCL_SOCKET_IFNAME=<bootstrap_iface>
export MSCCLPP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
```

Internode HT via mpirun (NCCL-EP convention with NUMA binding):

```bash
mpirun -np 16 --allow-run-as-root --hostfile <hostfile> \
    --bind-to numa \
    -x MSCCLPP_EP_BENCH=1 \
    -x MSCCLPP_EP_BENCH_TOKENS=4096 -x MSCCLPP_EP_BENCH_HIDDEN=7168 \
    -x MSCCLPP_EP_BENCH_EXPERTS=256 -x MSCCLPP_EP_BENCH_TOPK=8 \
    -x MASTER_ADDR=<master_ip> -x MASTER_PORT=29600 \
    bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
             WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
             LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
             exec python3 test/python/ext/ep/test_internode_multirank.py'
```

Internode LL via mpirun — same launch wrapper, swap the test script:

```bash
mpirun -np 16 --allow-run-as-root --hostfile <hostfile> \
    --bind-to numa \
    -x MSCCLPP_EP_BENCH=1 \
    -x MSCCLPP_EP_BENCH_TOKENS=128 -x MSCCLPP_EP_BENCH_HIDDEN=7168 \
    -x MSCCLPP_EP_BENCH_EXPERTS=256 -x MSCCLPP_EP_BENCH_TOPK=8 \
    -x MASTER_ADDR=<master_ip> -x MASTER_PORT=29600 \
    bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
             WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
             LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
             exec python3 test/python/ext/ep/test_low_latency_multirank.py'
```

Add `-x NCCL_SOCKET_IFNAME=<iface> -x MSCCLPP_SOCKET_IFNAME=<iface>
-x GLOO_SOCKET_IFNAME=<iface>` to the `mpirun` lines above only if the
default bootstrap NIC is wrong. `NCCL_IB_DISABLE` / `NCCL_TOPO_FILE`
are not required — EP traffic goes through mscclpp, not NCCL.

If Open MPI's own bootstrap misbehaves (e.g. UCX is mis-configured or
the host is multi-homed), force its control channel onto plain TCP
over the local management NIC by adding
`--mca pml ob1 --mca btl tcp,vader,self --mca btl_tcp_if_include <iface>`
to `mpirun`. `<iface>` is the management NIC (e.g. `enP22p1s0f1` on
Azure GB200, `eno1`/`bond0` elsewhere) — it is **not** `eth0` on
GB200.

### Benchmark mode

All three multirank tests double as micro-benchmarks when
`MSCCLPP_EP_BENCH=1` is set. Dispatch and combine are timed separately
with CUDA events; per-rank times are reduced across ranks and reported
as `min` / `avg` / `max`, with bandwidth computed from the average time
(matching NCCL-EP's `ep_bench` convention).

Env knobs:

| Variable                      | Meaning                                | Default |
|-------------------------------|----------------------------------------|---------|
| `MSCCLPP_EP_BENCH`            | Enable benchmark pass                  | `0`     |
| `MSCCLPP_EP_BENCH_WARMUP`     | Warmup iterations                      | `10`    |
| `MSCCLPP_EP_BENCH_ITERS`      | Timed iterations                       | `50`    |
| `MSCCLPP_EP_BENCH_TOKENS`     | Tokens per rank                        | test-specific |
| `MSCCLPP_EP_BENCH_HIDDEN`     | Hidden dim                             | `7168`  |
| `MSCCLPP_EP_BENCH_EXPERTS`    | Total experts                          | test-specific |
| `MSCCLPP_EP_BENCH_TOPK`       | top-k routing                          | `8`     |
| `MSCCLPP_EP_NUM_PROXIES`      | Number of `ProxyService`s in `Buffer`  | 8 (Hopper) / 1 (Blackwell) |

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
- [x] Validated on 16×GB200 NVL72 (64 ranks, Azure CX-7 RoCE) — see
      Phase 5.

### Phase 3 — Low-Latency (RDMA + CUDA-IPC) — DONE

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

`Buffer::low_latency_dispatch` / `low_latency_combine` are validated
intra-node (8 ranks, CUDA IPC fast path) and cross-node (16 ranks on
2×H100×8, IPC + IB). Functional correctness is bit-exact against the
reference dispatch/combine.

### Phase 4 — Validation

- [x] `test_intranode_multirank.py` — NVLink HT round-trip validated.
- [x] `test_internode_multirank.py` — HT round-trip validated on 2×H100×8.
- [x] `test_low_latency_multirank.py` — LL round-trip validated intra-node (8 ranks) and cross-node (2×H100×8).
- [x] In-tree micro-benchmark harness (`MSCCLPP_EP_BENCH=1`) reporting min/avg/max + BW@avg, aligned with NCCL-EP `ep_bench`.
- [ ] Throughput benchmarks against DeepEP upstream.

### Phase 5 — Azure GB200 NVL72 port — DONE

GB200 (4 GPUs / NUMA host, CX-7 RoCE) needed three independent fixes
on top of the H100 baseline:

- [x] `NUM_MAX_NVL_PEERS` made CMake-configurable
      (`-DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4`); runtime
      `MSCCLPP_EP_LOCAL_WORLD_SIZE` defaults to the compile-time value.
- [x] Portability fixes for older toolchains/arches: multimem PTX
      guarded by `__CUDA_ARCH__ >= 900`, `cuCtxCreate` 4-arg form
      version-guarded, `NUM_TIMEOUT_CYCLES` restored, CMake install
      destination dual-mode.
- [x] **LL bypass for broken CX-7 IB atomics** (Proposal A):
      `Buffer::rdma_buffer_ptr` allocated via
      `mscclpp::detail::gpuCallocPhysical` (POSIX_FD | FABRIC handles);
      LL IPC fast-path gate lifted from `num_rdma_ranks==1` to
      `low_latency_mode`; peer bases resolved through
      `RegisteredMemory::data()` so mscclpp `CudaIpc` transport imports
      cross-node cuMem fabric handles via `nvidia-imex`.
- [x] **NVLS multimem fast path for HT atomics** (Proposal B):
      runtime-gated by `mscclpp::isNvlsSupported()`. Cross-node
      `port_channel.signal/wait` + `putWithSignal` replaced by
      `multimem.red.add.u64` on NVL72 multicast counters; legacy IB
      path retained as fallback.
- [x] Validated on 16 × Azure GB200 NVL72 (64 ranks), HIDDEN=7168,
      tokens=4096, experts=256, top-k=8:
      - HT: dispatch ~2 006 GB/s agg, combine ~2 011 GB/s agg
        (`NVL_SEND=8 NVL_RECV=256 RDMA_SEND=8 RDMA_RECV=32`).
      - LL: dispatch ~16 817 GB/s agg (~262 GB/s/rank),
        combine ~21 148 GB/s agg (defaults).
