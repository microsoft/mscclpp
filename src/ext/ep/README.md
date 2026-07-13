# MSCCL++ Expert-Parallel (EP) extension

A torch-free nanobind extension for MoE `dispatch` / `combine` primitives in
MSCCL++. The module builds two active backends:

- **Low-latency (LL)**: `MoERuntime` / `MoECommunicator(mode=LOW_LATENCY)`,
  backed by `low_latency/dispatch.cu` and `low_latency/combine.cu`.
- **High-throughput (HT)**: `ExpertParallelRuntime` /
  `MoECommunicator(mode=HIGH_THROUGHPUT)`, backed by `ht_runtime.cc` and
  `ht/kernels/*`.

## Status

| Feature                          | Status |
|----------------------------------|--------|
| `mscclpp_ep_cpp` module          | ✅ builds LL + HT backends when `MSCCLPP_BUILD_EXT_EP=ON` |
| LL dispatch/combine              | ✅ validated on 8-rank H100, 16-rank 2×H100×8, and 64-rank GB200 NVL72 |
| HT dispatch/combine              | ✅ active DeepEP-style backend with intranode and internode paths |
| HT GB200 direct/flat fast paths  | ✅ runtime-gated by `MSCCLPP_EP_DIRECT`, `MSCCLPP_EP_INTRA_DIRECT`, and `MSCCLPP_EP_FLAT` |
| GB200 LL NVLS multimem fast path | ✅ runtime-gated by `mscclpp::isNvlsSupported()` |
| Python frontend `mscclpp.ep` | ✅ `MoECommunicator` selects LL or HT by `MoEMode` |

On Azure GB200 NVL72 (4 GPUs / NUMA host, CX-7 RoCE), LL was validated at
16 nodes × 4 GPUs = **64 ranks** with HIDDEN=7168, tokens=4096,
experts=256, top-k=8: dispatch ~**16 817 GB/s** agg (~262 GB/s per rank),
combine ~**21 148 GB/s** agg (defaults).

The GB200 LL path bypasses Azure CX-7 RoCE's broken `IBV_ATOMIC_*` by
routing peer pointers through cuMem fabric IPC over the NVL72 fabric
(`nvidia-imex`) and emitting NVLink-SHARP `multimem.*` atomics from the
kernels. The legacy RDMA-atomic PortChannel path is retained as a
fallback when `mscclpp::isNvlsSupported()` returns `false`.

The LL backend is a DeepEP legacy low-latency-style port, not a port of
DeepEP elastic. It uses a mixed transport: `MemoryChannel` (CUDA IPC) for
same-node peers and `PortChannel` (CPU proxy + IB verbs) for remote peers.
On GB200 NVL72, cross-node peers are also reached through imported cuMem
fabric handles over `nvidia-imex`, and kernels emit NVLink-SHARP
`multimem.*` atomics directly on the NVL72 fabric. The LL primitive mapping is:

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
bottleneck for cross-node LL traffic. The EP runtime therefore allocates `N`
ProxyServices and shards `PortChannel`s across them by `(qp_idx,
dst_rank)`.

- Default: `8` on Hopper (sm_90), `1` on Blackwell / sm_100+ (NVSwitch).
- Override at runtime: `MSCCLPP_EP_NUM_PROXIES=<N>` (clamped to ≥1).
  Rank 0 prints the resolved value at construction.
- Sweet spot on 2×H100×8 is `N=8`; `N=12` over-subscribes the host CPUs
  and collapses throughput.

### Known limitations

- ROCm is not supported for the EP extension yet.
- LL currently supports BF16 input, optional FP8 E4M3 dispatch output, and
  `DispatchLayout.EXPERT_MAJOR`.
- HT currently supports BF16 input and `DispatchLayout.FLAT`; quantized HT
  dispatch is not implemented.
- LL fallback cross-node traffic uses `PortChannel` and a CPU proxy. GB200 NVL72
  uses the fabric-IPC/NVLS path instead when it is available.
- HT direct and flat paths are GB200/NVL72 optimizations. Leave the env vars
  unset for the baseline DeepEP-style HT path.

## Build

For Python installs, use the `ep` extra:

```bash
python -m pip install ".[cuda12,ep]"
```

The EP extension targets CUDA architectures **90 or newer**. Plain CMake builds
can enable it explicitly:

```bash
cmake -S . -B build \
      -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build -j
```

This produces `mscclpp_ep_cpp.so` — a nanobind extension module.
The Python frontend picks it up automatically:

```python
import mscclpp.ep as ep
moe_comm = ep.MoECommunicator(...)
```

### Build-time CMake options

| Variable                              | Default | Meaning                                                       |
|---------------------------------------|---------|---------------------------------------------------------------|
| `MSCCLPP_BUILD_EXT_EP`                | `ON` in Python wheels | Build the EP extension at all                |
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
      -DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4
cmake --build build -j

# Option 2: wheel-based install.
python3 -m pip install ".[cuda12,ep]" \
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
  runtime construction; otherwise the kernels fall back to the legacy
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
export NCCL_IB_DISABLE=1                              # avoid NCCL's own IB probe on Azure CX-7
export NCCL_MNNVL_ENABLE=0                            # Azure GB200 is NOT one MNNVL fabric across nodes
export MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # avoid mlx5_bond_0 (PORT_DOWN)
# Bootstrap NIC (only if auto-detect picks wrong) — on Azure GB200:
export NCCL_SOCKET_IFNAME=enP22p1s0f1
export MSCCLPP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
```

HT runtime knobs (env vars exposed by `test_intranode_multirank.py` /
`test_internode_multirank.py` — defaults below are the test-script defaults,
**not** the `ep.Config(...)` constructor defaults):

| Variable                  | Maps to (`ep.Config` field)         | Default | Notes                                  |
|---------------------------|--------------------------------------|--------:|-----------------------------------------|
| `MSCCLPP_EP_NUM_SMS`      | `num_sms`                            | `20`    | HT intranode test only. Try `64` on GB200 intranode for `dispatch` BW. |
| `MSCCLPP_EP_NSM`          | `num_sms`                            | `152`   | HT internode test only. |
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
> built-in IB probe does not crash `ep.MoECommunicator` construction.

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
├── CMakeLists.txt              — builds mscclpp_ep_cpp (nanobind)
├── README.md                   — this file
├── moe_runtime.hpp / .cc       — LL MoE runtime state and raw-pointer dispatch/combine
├── config.hpp                  — LL layout helpers and size hints
├── bindings.cpp                — nanobind module definition
├── ht_runtime.hpp / .cc        — HT runtime state and raw-pointer dispatch/combine
├── ht/                         — active HT kernel/config sources
│   ├── buffer.hpp / buffer.cc
│   ├── config.hpp
│   ├── event.hpp
│   └── kernels/
│       ├── buffer.cuh
│       ├── runtime.cu
│       ├── intranode_kernel.cu
│       ├── internode.cu
│       └── internode_ncclep.cuh
└── kernels/
    ├── api.cuh                 — host-callable kernel prototypes
    ├── configs.cuh             — compile-time constants (GPU-only)
    ├── exception.cuh           — EP_HOST/DEVICE_ASSERT + CUDA_CHECK
    ├── launch.cuh              — SETUP_LAUNCH_CONFIG / SWITCH_* macros
    ├── utils.cuh               — device inline helpers
    └── low_latency.cu          — LL dispatch/combine (RDMA + IPC paths)

python/mscclpp/ep/
├── __init__.py                 — reexports the public MoECommunicator API
└── communicator.py             — torch.Tensor frontend over raw-pointer runtime calls

test/python/ep/
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
conda install -c conda-forge -y cupy mpi4py nanobind blake3 sortedcontainers

# PyTorch (pulls a matching cuda-toolkit + NCCL).
pip3 install torch

# mscclpp build deps (used by `pip install .` of this repo).
pip install scikit-build-core nanobind setuptools_scm
```

Then build the EP extension (see [Build](#build)) — `pip install .` from
the repo root installs `mscclpp_ep_cpp.so` into the active env so the
test scripts can `import mscclpp.ep as ep`.

Intranode (single node, 8 GPUs) — HT:

```bash
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=1 --nproc_per_node=8 \
    test/python/ep/test_intranode_multirank.py
```

Intranode LL (single node, 8 GPUs):

```bash
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=128 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=1 --nproc_per_node=8 \
    test/python/ep/test_low_latency_multirank.py
```

Internode HT (2 nodes × 8 GPUs), torchrun:

```bash
# node 0 (master)
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ep/test_internode_multirank.py

# node 1 (worker)
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<master_ip> --master_port=29600 \
    test/python/ep/test_internode_multirank.py
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
             exec python3 test/python/ep/test_internode_multirank.py'
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
             exec python3 test/python/ep/test_low_latency_multirank.py'
```

Add `-x NCCL_SOCKET_IFNAME=<iface> -x MSCCLPP_SOCKET_IFNAME=<iface>
-x GLOO_SOCKET_IFNAME=<iface>` to the `mpirun` lines above only if the
default bootstrap NIC is wrong. Generic H100/HGX runs do not require
`NCCL_IB_DISABLE` or `NCCL_TOPO_FILE`; the GB200 reference env above sets
`NCCL_IB_DISABLE=1` only to avoid NCCL's own IB probing on Azure CX-7.

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
| `MSCCLPP_EP_NUM_PROXIES`      | Number of runtime `ProxyService`s      | 8 (Hopper) / 1 (Blackwell) |

## Implementation notes

### Backend map

`mscclpp_ep_cpp` exposes both backends in one Python extension:

| Backend | Python API | C++ runtime | Kernel sources | Layout |
|---------|------------|-------------|----------------|--------|
| LL | `MoECommunicator(mode=MoEMode.LOW_LATENCY)` / `MoERuntime` | `moe_runtime.cc` | `low_latency/{dispatch,combine}.cu` | `EXPERT_MAJOR` |
| HT | `MoECommunicator(mode=MoEMode.HIGH_THROUGHPUT)` / `ExpertParallelRuntime` | `ht_runtime.cc` | `ht/kernels/*` | `FLAT` |

Shared internal headers live in `include/`. The previous LL implementation is
kept in `legacy/low_latency.cu` for reference and is not compiled.

The LL runtime uses one `low_latency_num_blocks` setting. Its default is 130:
dispatch launches 128 workers plus scheduler/notify blocks, while combine
launches 128 workers. `RANK_LOCAL_REDUCE` is the default combine mode;
`DIRECT_SEND` preserves bit-exact top-k reduction order.

The LL payload layout reserves optional scale storage for future quantization,
but the active LL kernels currently accept BF16 inputs and expert outputs only.

### MSCCL++ EP LL vs NCCL EP LL vs DeepEP elastic dispatch payloads

MSCCL++ EP LL dispatch sends one message per routed expert. The destination
receive buffer is partitioned by local expert, source rank, and slot, so the
buffer position encodes the destination expert and source rank. The inline
header only carries the source token's local row index:

```text
BF16 dispatch message:
[int4 header: src_token_idx][BF16 hidden payload]

FP8 dispatch message:
[int4 header: src_token_idx][FP8 hidden payload][FP32 scale per 128 elements]
```

`topk_ids` and `topk_weights` are not sent in the LL dispatch payload. They are
kept in the dispatch handle on the source rank and are used by `combine` to
weight the returned expert outputs. Dispatch sends a small per-expert count
signal separately so the receiver knows how many token messages arrived.

NCCL EP LL sends one message per destination rank after rank-level
deduplication. Its RDMA path stores each slot as an interleaved message, while
its NVLink/P2P path stores headers and payloads in split sections inside the
same per-source-rank receive region:

```text
RDMA:
[DispatchHdr][hidden payload][optional FP8 scales]

NVLink/P2P:
[hdr0][hdr1]...[hdrN][payload0][payload1]...[payloadN]
```

The header carries `token_id` plus one routing entry per top-k choice:

```text
DispatchHdr(EXPERT_MAJOR) = [token_id][expert_id[num_topk]]
DispatchHdr(RANK_MAJOR)   = [token_id][(topk_weight, expert_id)[num_topk]]
```

The NCCL EP expert-major header is `align16(4 + num_topk * 2)` bytes because
`expert_id` is `uint16_t`. The rank-major header is
`align16(4 + num_topk * 8)` bytes because each routing entry stores an FP32
weight and a `uint16_t` expert ID plus padding.

DeepEP elastic dispatch uses a different notify/data pipeline. It deduplicates
by destination rank: if multiple top-k experts for the same token live on the
same rank, the hidden payload is sent to that rank only once. Because the
receiver must recover all local expert hits from that single token copy, the
token buffer carries the full top-k metadata:

```text
[hidden payload][optional scale-factor packs][metadata]
metadata = [topk_idx[num_topk]][optional topk_weights[num_topk]][src_token_global_idx][linked_list_idx[num_topk]]
```

`topk_idx` entries are 32-bit expert IDs, `topk_weights` entries are FP32 routing
weights, and `src_token_global_idx = src_rank * num_max_tokens_per_rank +
src_local_token_idx`. The common DeepEP token layout reserves the linked-list
slots even when the non-hybrid path does not use them. After a GPU barrier
guarantees data arrival, the copy epilogue copies data into `recv_x`, `recv_sf`,
`recv_topk_idx`, `recv_topk_weights`, and source metadata.

The optimized BF16 low-latency dispatch/combine kernels are instantiated for
`hidden = 4096`, `7168`, `8192`, and `9216`; other hidden sizes are rejected.
For BF16 dispatch with `num_tokens = 128`, `hidden = 7168`, and
`num_experts = 256`, the expected per-source-rank dispatch payload is:

| Case | Sends per token | Bytes per send | Expected dispatch bytes |
|------|-----------------|----------------|-------------------------|
| MSCCL++ LL, `topk=8` | `8` | `16 + 7168 * 2 = 14,352 B` | `128 * 8 * 14,352 = 14,696,448 B` (`14.70 MB`, `14.02 MiB`) |
| NCCL EP LL expert-major, 16 GPUs, `topk=8` | `16 * (1 - C(240, 8) / C(256, 8)) = 6.523` | `align16(4 + 8*2) + 7168 * 2 = 14,368 B` | `128 * 6.523 * 14,368 ~= 11,996,989 B` (`12.00 MB`, `11.44 MiB`) |
| NCCL EP LL rank-major, 16 GPUs, `topk=8` | `16 * (1 - C(240, 8) / C(256, 8)) = 6.523` | `align16(4 + 8*8) + 7168 * 2 = 14,416 B` | `128 * 6.523 * 14,416 ~= 12,037,069 B` (`12.04 MB`, `11.48 MiB`) |
| DeepEP elastic, 16 GPUs, `topk=8` | `16 * (1 - C(240, 8) / C(256, 8)) = 6.523` | `7168 * 2 + align32(8*4 + 8*4 + 4 + 8*4) = 14,464 B` | `128 * 6.523 * 14,464 ~= 12,077,148 B` (`12.08 MB`, `11.52 MiB`) |
| MSCCL++ LL, `topk=4` | `4` | `16 + 7168 * 2 = 14,352 B` | `128 * 4 * 14,352 = 7,348,224 B` (`7.35 MB`, `7.01 MiB`) |
| NCCL EP LL expert-major, 32 GPUs, `topk=4` | `32 * (1 - C(248, 4) / C(256, 4)) = 3.838` | `align16(4 + 4*2) + 7168 * 2 = 14,352 B` | `128 * 3.838 * 14,352 ~= 7,050,391 B` (`7.05 MB`, `6.72 MiB`) |
| NCCL EP LL rank-major, 32 GPUs, `topk=4` | `32 * (1 - C(248, 4) / C(256, 4)) = 3.838` | `align16(4 + 4*8) + 7168 * 2 = 14,384 B` | `128 * 3.838 * 14,384 ~= 7,066,111 B` (`7.07 MB`, `6.74 MiB`) |
| DeepEP elastic, 32 GPUs, `topk=4` | `32 * (1 - C(248, 4) / C(256, 4)) = 3.838` | `7168 * 2 + align32(4*4 + 4*4 + 4 + 4*4) = 14,400 B` | `128 * 3.838 * 14,400 ~= 7,073,971 B` (`7.07 MB`, `6.75 MiB`) |

The table counts token data plus inline per-token metadata. It excludes the
small count, tail, barrier, and completion signals.
