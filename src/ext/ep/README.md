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
├── mscclpp_ep_bench.cu                — pure-C++ NCCL-style LL benchmark
├── run_ep_bench_python.py            — in-process MSCCL++ vs NCCL-EP Python bench
└── run_ep_bench.py                    — unified MSCCL++-cpp / NCCL-EP driver
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

Cross-node LL is supported only when bootstrap reports one GPU IPC domain
(for example GB200 NVL72 with IMEX). The launch does not pass a domain size:

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

`run_ep_bench_python.py` provides the high-level Python equivalent (drives both the
MSCCL++ and NCCL-EP Python APIs in-process). The existing
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

### Unified in-process benchmark (mscclpp vs NCCL-EP)

`test/python/ep/run_ep_bench_python.py` drives **both** the mscclpp EP Python API
(`MoECommunicator.dispatch` / `.combine`) and NVIDIA NCCL-EP's `nccl.ep`
(`nccl4py`) dispatch / combine in a **single process**, through one shared paired
`dispatch -> sync -> combine -> sync -> barrier` loop. Both backends are therefore
timed with byte-for-byte the same methodology and emit the same
`=== Summary (Low Latency) ===` block as the C++ `mscclpp_ep_bench`, so the two can
be diffed directly. `--backend {mscclpp,nccl,both}` selects the backend(s); `both`
runs `nccl` first, then `mscclpp`.

Bootstrap is MPI (`mpi4py` + `mpirun`), shared by both backends (mscclpp via
`CommGroup(mpi_comm=...)`, NCCL-EP via a unique-id broadcast over MPI); torch is
used only for CUDA tensors and event timing, not for its distributed backend.

The NCCL-EP path additionally needs, in the launch environment: `nvcc` on `PATH`
(it JIT-compiles its LL kernels on first use), its build's `lib/` on
`LD_LIBRARY_PATH`, and -- when the environment's default `libnccl` is older than
the one `libnccl_ep.so` was built against -- an `LD_PRELOAD` of the in-tree
`libnccl.so`. mscclpp's LL RDMA setup needs the active HCA list
(`MSCCLPP_HCA_DEVICES`).

Single node, 4 GPUs (Azure GB200), comparing both backends at `e128`:

```bash
# Point these at your local NCCL-EP build and CUDA toolkit.
NCCL_BUILD=/opt/microsoft/mrc/ep/nccl/build
NCCL_SRC=/opt/microsoft/mrc/ep/nccl/contrib/nccl_ep
CUDA_HOME=/usr/local/cuda
NPROC=4

# libnccl_ep.so is built against the in-tree libnccl; preload it so an older
# environment libnccl (e.g. a pip nvidia-nccl wheel) does not win the dynamic
# link race and trip NCCL-EP's version check. Highest-versioned libnccl.so.* wins.
PRELOAD_NCCL="$(ls -1 "$NCCL_BUILD"/lib/libnccl.so.*.* | sort -V | tail -1)"

mpirun -np "$NPROC" --bind-to none \
    -x PATH="$CUDA_HOME/bin:$PATH" \
    -x CUDA_HOME="$CUDA_HOME" \
    -x LD_LIBRARY_PATH="$NCCL_BUILD/lib:$LD_LIBRARY_PATH" \
    -x LD_PRELOAD="$PRELOAD_NCCL" \
    -x NCCL_EP_JIT_SOURCE_DIR="$NCCL_SRC" \
    -x NCCL_EP_JIT_BUILD_INCLUDE_DIR="$NCCL_BUILD/include" \
    -x NCCL_IB_DISABLE=1 -x NCCL_MNNVL_ENABLE=0 -x NCCL_NET_PLUGIN=none \
    python test/python/ep/run_ep_bench_python.py \
        --backend both -e 128 -t 128 -d 7168 -k 8 -w 10 -i 50
```

For mscclpp only, pass `--backend mscclpp`; the NCCL-EP-specific env vars
(`NCCL_EP_JIT_*`, `LD_PRELOAD`, the NCCL build `lib/`) are then unnecessary.

Pure kernel (device) time: add `--kernel-timing`. Both backends launch their LL
dispatch/combine as *cooperative* kernels, which `torch.profiler` / Kineto
mis-handles; the in-process CUPTI Activity collector (`cupti_kernel_timer.cpp`,
built to `libcupti_kernel_timer.so` next to the script on first use) captures them
via `CUPTI_ACTIVITY_KIND_KERNEL` and buckets by the mangled-name substrings
`dispatch` / `combine` (both backends' LL kernels are named exactly that). This
adds a `--- Kernel-only performance ---` block per backend alongside the
host-observed one. The dispatch kernel ends in a cross-rank recv spin-wait, so the
cross-rank **min** (the rank that did not wait) is reported as the representative
kernel floor; combine has little recv-spin and is stable, so its avg/min/max are
all shown.

Ordering note: initializing mscclpp's LL `MoECommunicator` perturbs CUDA state
that breaks a *later* NCCL-EP cooperative-launch dispatch, so `--backend both`
runs NCCL-EP first. Each backend self-warms, so the ordering does not affect the
reported numbers.

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

The runtime owns two fixed communication receive regions inside one symmetric
allocation: dispatch always writes `dispatchRecvBuffer`, while combine reads
that region and writes `combineRecvBuffer`. Buffer sizing and allocation remain
entirely inside C++; Python passes only the static workload dimensions.

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

Small rank/expert counts are sent as LL8 packets. After every payload TMA store
has completed, the sender publishes readiness through a lightweight
`BaseMemoryChannel` signal; the receiver waits on that signal before copying
data into expert-major output.

The optimized kernels are instantiated for hidden sizes `4096`, `6656`,
`7168`, `8192`, and `9216`; other hidden sizes are rejected. FP8 E4M3
currently fixes the scale block at 128. A future scale layout must use a
distinct `DispatchDataType`.

### Current H100 performance

Measured on 8×H100 with 128 tokens/rank, hidden 7168, top-k 8, 256 experts,
50 warmup iterations, and NCCL-EP-style random routing with masked entries.
No LL topology environment variables were set.

CUDA Graph dispatch+combine E2E:

| Dispatch format | Combine mode | E2E |
|---|---|---:|
| BF16 | rank-local reduce | 81.2 µs |
| BF16 | direct send | 100.2 µs |
| FP8 E4M3 | rank-local reduce | 73.1 µs |
| FP8 E4M3 | direct send | 93.5 µs |

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
