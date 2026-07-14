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
| LL dispatch/combine              | ✅ validated on 8-rank H100 with direct IPC and a forced mixed IPC/PortChannel topology |
| HT dispatch/combine              | ✅ active DeepEP-style backend with intranode and internode paths |
| HT GB200 direct/flat fast paths  | ✅ runtime-gated by `MSCCLPP_EP_DIRECT`, `MSCCLPP_EP_INTRA_DIRECT`, and `MSCCLPP_EP_FLAT` |
| LL topology discovery            | ✅ automatic node + NVML fabric-domain detection through `Bootstrap` |
| Python frontend `mscclpp.ep` | ✅ `MoECommunicator` selects LL or HT by `MoEMode` |

### LL topology and transport

The optimized LL backend selects the transport per peer:

- Same-host peers use regular CUDA IPC mappings.
- Cross-host peers use CUDA fabric handles when all ranks belong to one
  NVML-reported GPU fabric domain, such as GB200 NVL72 with `nvidia-imex`.
- Peers outside the direct IPC domain use an IB `PortChannel`. Metadata LL8
  packets are sent as soon as they are staged; the later payload uses
  `putWithSignal`, and the receiver waits only when metadata reports nonzero
  tokens. Devices without RDMA atomics use the core HostNoAtomic fallback and
  require GDRCopy.
- `BaseMemoryChannel` handles provide synchronization for direct-IPC peers.
  Each non-IPC peer has its own PortChannel QP, proxy service, and FIFO so
  independent peer traffic is not serialized by one CPU proxy thread.

Topology is automatic. `TcpBootstrap` detects:

1. ranks per host from bootstrap peer addresses;
2. ranks per GPU IPC domain from NVML fabric `clusterUuid + cliqueId`;
3. host-local IPC domains as a safe fallback when NVML fabric information is
   unavailable or incomplete.

LL therefore does **not** require topology environment variables such as
`MSCCLPP_EP_LOCAL_WORLD_SIZE`, `MSCCLPP_EP_FABRIC_IPC`, an NVML-domain ID, or
`NCCL_MNNVL_ENABLE`. Socket/HCA variables are only optional launch overrides
when the generic bootstrap or an HT/IB test selects the wrong interface.

### HT proxy sharding

A single `mscclpp::ProxyService` is one CPU host thread driving one
FIFO. With 8 GPUs / node sharing one proxy, the host thread becomes the
bottleneck for cross-node proxy traffic. The HT runtime therefore allocates `N`
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
- LL PortChannel fallback requires an available IB transport. GDRCopy is
  required only when the selected device falls back to HostNoAtomic mode. The
  registered symmetric buffer must remain below 4 GiB because PortChannel
  offsets are 32-bit.
- HT direct and flat paths are GB200/NVL72 optimizations. Leave the env vars
  unset for the baseline DeepEP-style HT path.

## Build

Python installs build the EP extension by default:

```bash
python -m pip install .
# Optional CuPy dependency:
python -m pip install ".[cuda12]"
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
| `MSCCLPP_EP_NUM_MAX_NVL_PEERS`        | `8`     | HT compile-time NVLink peer capacity; optimized LL topology is detected at runtime |
| `MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT`     | `OFF`   | Use a short ~10s kernel spin timeout (default is ~100s)       |

### Azure GB200 (NVL72, 4 GPUs / NUMA host)

GB200 NVL72 nodes expose **4 GPUs per NUMA host** (not 8 like HGX
H100). The optimized LL backend discovers this topology from bootstrap/NVML
and needs no GB200-specific build option. HT kernels still use the compile-time
NVLink peer capacity and should be built with:

```bash
cmake -S . -B build \
      -DMSCCLPP_BUILD_EXT_EP=ON \
      -DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4
cmake --build build -j 64

python3 -m pip install . \
    --config-settings=cmake.define.MSCCLPP_EP_NUM_MAX_NVL_PEERS=4
```

Add `-DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT=ON` only when triaging hangs (it
shortens the kernel-side spin timeout from ~100s to ~10s).

Runtime prerequisites on GB200:

- CUDA Toolkit ≥ 12.5 (the `cuCtxCreate` proxy-context path uses the
  4-arg signature added in 12.5; older toolkits compile against the
  3-arg fallback automatically).
- Driver ≥ 555 with nvidia-imex configured so cuMem fabric handles
  (`POSIX_FD | FABRIC`) can be exchanged across nodes.
- NVML must report a completed fabric state with a common cluster UUID and
  clique ID for all participating ranks. Bootstrap uses this information to
  form the IPC domain automatically.
- The LL runtime verifies CUDA fabric-handle allocation and NVLS-backed
  semaphore-token support. If either capability is unavailable on any rank,
  all ranks consistently use host-local IPC within each node and PortChannel
  for peers outside that IPC domain.
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

No LL-specific runtime environment variables are required. Optional overrides
are only needed when generic launch components choose the wrong interface:

```bash
export NCCL_SOCKET_IFNAME=enP22p1s0f1
export MSCCLPP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
# IB transports (LL fallback and HT), if automatic HCA selection is wrong:
export MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

`NCCL_IB_DISABLE` and `NCCL_MNNVL_ENABLE` configure NCCL, not the MSCCL++ LL
runtime. Set them only when the surrounding PyTorch/NCCL test launcher needs
those overrides.

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
├── include/                    — shared EP headers and quantization helpers
└── low_latency/
    ├── dispatch.cu             — optimized LL dispatch
    ├── combine.cu              — optimized LL combine
    └── config.cuh              — LL launch/workspace configuration

python/mscclpp/ep/
├── __init__.py                 — public exports
├── communicator.py             — backend-selecting public API
├── low_latency.py              — LL torch.Tensor frontend
└── high_throughput.py          — HT torch.Tensor frontend

test/python/ep/
├── test_intranode_multirank.py        — intranode HT dispatch+combine
├── test_internode_multirank.py        — internode HT dispatch+combine
├── test_low_latency_multirank.py      — LL correctness + CUDA Graph
├── ep_bench_ll.py                     — Python NCCL-style LL benchmark
├── mscclpp_ep_bench.cu                — pure-C++ NCCL-style LL benchmark
└── run_ep_bench.py                    — unified MSCCL++ / NCCL-EP driver
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

Intranode LL (single node, 8 GPUs), with no topology env:

```bash
torchrun --standalone --nproc_per_node=8 \
    test/python/ep/test_low_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 \
    --cuda-graph
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

Cross-node LL automatically uses direct CUDA fabric mappings when bootstrap
reports one GPU IPC domain (for example GB200 NVL72 with IMEX), otherwise it
uses PortChannel for peers outside each host-local IPC domain. The launch does
not pass a domain size:

```bash
mpirun -np 16 --allow-run-as-root --hostfile <hostfile> \
    --bind-to numa \
    -x MASTER_ADDR=<master_ip> -x MASTER_PORT=29600 \
    bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
             WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
             LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
             exec python3 test/python/ep/test_low_latency_multirank.py \
               --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256'
```

Add `-x NCCL_SOCKET_IFNAME=<iface> -x MSCCLPP_SOCKET_IFNAME=<iface>
-x GLOO_SOCKET_IFNAME=<iface>` to the `mpirun` lines above only if the
default bootstrap NIC is wrong. These variables affect the launcher/bootstrap,
not LL topology detection.

If Open MPI's own bootstrap misbehaves (e.g. UCX is mis-configured or
the host is multi-homed), force its control channel onto plain TCP
over the local management NIC by adding
`--mca pml ob1 --mca btl tcp,vader,self --mca btl_tcp_if_include <iface>`
to `mpirun`. `<iface>` is the management NIC (e.g. `enP22p1s0f1` on
Azure GB200, `eno1`/`bond0` elsewhere) — it is **not** `eth0` on
GB200.

### Benchmark mode

Use the unified driver for NCCL-EP-style LL measurements:

```bash
# Build the pure-C++ benchmark.
cmake -S test/python/ep -B test/python/ep/build \
    -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build test/python/ep/build -j 64

# BF16 rank-local.
python3 test/python/ep/run_ep_bench.py \
    --ep-lib mscclpp-cpp --nproc-per-node 8 \
    -t 128 -d 7168 -k 8 -e 256 -w 50 -i 100 \
    --dispatch-dtype bf16 --combine-mode rank_local_reduce

# FP8 direct-send.
python3 test/python/ep/run_ep_bench.py \
    --ep-lib mscclpp-cpp --nproc-per-node 8 \
    -t 128 -d 7168 -k 8 -e 256 -w 50 -i 100 \
    --dispatch-dtype fp8_e4m3 --combine-mode direct_send
```

`ep_bench_ll.py` provides the high-level Python equivalent. The existing
multirank HT tests also expose their older env-controlled benchmark pass.

HT multirank test benchmark env knobs:

| Variable                      | Meaning                                | Default |
|-------------------------------|----------------------------------------|---------|
| `MSCCLPP_EP_BENCH`            | Enable benchmark pass                  | `0`     |
| `MSCCLPP_EP_BENCH_WARMUP`     | Warmup iterations                      | `10`    |
| `MSCCLPP_EP_BENCH_ITERS`      | Timed iterations                       | `50`    |
| `MSCCLPP_EP_BENCH_TOKENS`     | Tokens per rank                        | test-specific |
| `MSCCLPP_EP_BENCH_HIDDEN`     | Hidden dim                             | `7168`  |
| `MSCCLPP_EP_BENCH_EXPERTS`    | Total experts                          | test-specific |
| `MSCCLPP_EP_BENCH_TOPK`       | top-k routing                          | `8`     |

## Implementation notes

### Backend map

`mscclpp_ep_cpp` exposes both backends in one Python extension:

| Backend | Python API | C++ runtime | Kernel sources | Layout |
|---------|------------|-------------|----------------|--------|
| LL | `MoECommunicator(mode=MoEMode.LOW_LATENCY)` / `MoERuntime` | `moe_runtime.cc` | `low_latency/{dispatch,combine}.cu` | `EXPERT_MAJOR` |
| HT | `MoECommunicator(mode=MoEMode.HIGH_THROUGHPUT)` / `ExpertParallelRuntime` | `ht_runtime.cc` | `ht/kernels/*` | `FLAT` |

Shared internal headers live in `include/`.

The LL runtime uses one `low_latency_num_blocks` setting. Its default is 130:
dispatch launches 128 workers plus scheduler/notify blocks, while combine
launches 128 workers. `RANK_LOCAL_REDUCE` is the default combine mode;
`DIRECT_SEND` preserves bit-exact top-k reduction order.

LL accepts BF16 input and can produce BF16 or FP8 E4M3 dispatch output. FP8
uses one FP32 scale per 128 hidden elements. Expert output passed to combine is
always BF16.

Each ping-pong slot is one symmetric registered buffer. Kernel APIs pass only
the current/previous slot base; `BufferView` derives payload, compact-slot-map,
and staging regions, while `TransportView` owns self/IPC/Port peer selection
and symmetric offsets.

### LL dispatch payload

Dispatch deduplicates routing by destination rank: a token routed to multiple
experts on the same rank sends its hidden data only once. The receiver expands
that payload into all matching local-expert rows. Each 128-byte-aligned payload
contains:

```text
[data: BF16[hidden] or FP8_E4M3[hidden]]
[optional scales: FP32[hidden / 128]]
[topk expert ids: int32[num_topk]]
[topk weights: FP32[num_topk]]
[source token global index: int32]
```

Small rank/expert counts are sent as LL8 packets. Direct-IPC peers use a
`BaseMemoryChannel` signal after payload stores. PortChannel peers send metadata
first, batch staged payloads into one `putWithSignal` per nonempty peer, then
wait on that signal before consuming payload data. No PortChannel flush is used.

The optimized kernels are instantiated for hidden sizes `4096`, `7168`,
`8192`, and `9216`; other hidden sizes are rejected. FP8 E4M3 currently fixes
the scale block at 128. A future scale layout must use a distinct
`DispatchDataType`.

### Current H100 performance

Measured on 8×H100 with 128 tokens/rank, hidden 7168, top-k 8, 256 experts,
50 warmup iterations, and NCCL-EP-style random routing with masked entries.
No LL topology environment variables were set.

CUDA Graph dispatch+combine E2E:

| Dispatch format | Combine mode | E2E |
|---|---|---:|
| BF16 | rank-local reduce | 81.5 µs |
| BF16 | direct send | 101.3 µs |
| FP8 E4M3 | rank-local reduce | 71.8 µs |
| FP8 E4M3 | direct send | 94.5 µs |

Forced 4+4 IPC-domain split (`MSCCLPP_EP_TEST_IPC_DOMAIN_SIZE=4`), where
cross-group traffic uses one QP and one batched payload put per peer:

| Dispatch format | Combine mode | E2E |
|---|---|---:|
| BF16 | rank-local reduce | 278.6 µs |
| BF16 | direct send | 422.2 µs |
| FP8 E4M3 | rank-local reduce | 230.9 µs |
| FP8 E4M3 | direct send | 391.6 µs |

A no-payload probe measured about 47.5 µs dispatch and 58.0 µs combine,
isolating roughly 105 µs of fixed kernel, staging, proxy, and completion
overhead before payload transfer.

Rank-local combine preserves the compact destination slot assigned during
dispatch. Each PortChannel peer therefore sends exactly its metadata count
rather than a fixed `max_tokens_per_rank` slab. With this routing, the average
cross-domain segment is about 85 rows instead of 128.

Combine uses one PortChannel semaphore token per peer per iteration. Rank-local
nonempty segments use `putWithSignal`; zero-token peers send a standalone
signal. Direct-send signals after all per-row puts have been issued. Receivers
wait exactly once before reducing.

Pure-C++ paired benchmark (`dispatch → synchronize → combine → synchronize`):

| Dispatch format | Combine mode | Dispatch | Combine | D+C |
|---|---|---:|---:|---:|
| BF16 | rank-local reduce | 45.38 µs | 45.66 µs | 99.33 µs |
| BF16 | direct send | 45.18 µs | 64.39 µs | 117.83 µs |
| FP8 E4M3 | rank-local reduce | 35.59 µs | 45.98 µs | 89.67 µs |
| FP8 E4M3 | direct send | 35.76 µs | 66.68 µs | 110.55 µs |

With the same BF16 expert-major workload, NCCL-EP measured 57.99 µs dispatch,
75.19 µs combine, and 141.26 µs D+C. MSCCL++ rank-local measured 45.49 µs,
45.30 µs, and 98.74 µs respectively in that comparison run.

Rank-local combine reduces expert rows on each destination rank before sending
one BF16 partial per source rank/token. It may differ by one BF16 ULP because
the reduction order changes. `DIRECT_SEND` preserves bit-exact top-k order.
