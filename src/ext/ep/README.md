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

## Synthetic MoE reference benchmark

`test/python/ep/moe_bench/` adds real local expert computation to dispatch and
combine without changing the communication-only `run_ep_bench_python.py`.
Install the matching CUDA extra and benchmark dependencies, for example:

```bash
python -m pip install -e '.[cuda13,moe-benchmark]'
python test/python/ep/moe_bench/run.py \
  --model small --input decode --backend mscclpp --routing balanced \
  --warmup 10 --iterations 100
```

The launcher uses all visible GPUs on the current node and starts one worker per
GPU using `torchrun`; do not wrap it in another `torchrun` or `mpirun`.
It requires exactly 2 or 4 visible GPUs and respects `CUDA_VISIBLE_DEVICES`;
unsupported counts raise an error rather than selecting a subset.
There is no GPU-count option. Remove `--gpus` and `environment.num_gpus` from
older commands and YAML files; the detected count is recorded in run results.
Local worker coordination uses `127.0.0.1` to avoid hostname-resolution failures.
Gloo defaults to the loopback interface `lo`, unless `GLOO_SOCKET_IFNAME` is
already set; no IP-address option is required for this single-node benchmark.
Its default configuration is
`test/python/ep/moe_bench/configs.yaml`, independently of the working directory.
Use `--config` to select another YAML file. Named `models`, `inputs`, and `routing`
presets describe each workload. One `environment` mapping defaults to
`communication_backend: mscclpp`, `compute_backend: torch`, and `cuda_graph: false`.
Independently override these defaults with `--backend {mscclpp,nccl}`,
and `--graph {none,mlp}`. Omitted flags retain the YAML values;
`--graph none` explicitly disables graphs even when the YAML default is `mlp`.
The old `environments` mapping and `--environment` labels are no longer supported.
The shared
`benchmark` settings (`warmup`, `iterations`, `seed`, `output_dir`) can be
overridden with `--warmup`, `--iterations`, `--seed`, and `--output-dir`.

This supports MSCCL++ or NCCL-EP BF16 native communication on two
or four local GPUs, equal contiguous expert placement, and PyTorch reference
SwiGLU compute by default. `cuda_graph: false` preserves eager execution. Select
`--graph mlp` (`cuda_graph: mlp`) to capture only the local
expert MLPs and any local layout packing/weighted reduction. Dispatch and combine
remain eager, with routing weights applied exactly once. Capture and library
warmup happen before benchmark warmup and timing; replay uses current dispatched
data in fixed buffers with fixed expert counts. Timing still covers the entire
dispatch/MLP/combine interval, including eager launch gaps and peer waits.
`run.json` records `cuda_graph_scope: mlp_only` for these runs.
Other communication backends, unsupported compute format pairs,
full-pipeline graphs, overlap, and multi-node execution are rejected rather than
silently substituted. Each expert has independent seeded gate/up/down matrices.
Supported hidden dimensions are 4096, 4352, 5120, 6656, 7168, 8192, 8704, and 9216;
top-k is at most nine and cannot exceed the global expert count.
These common workload limits are retained for both backends to allow comparable
runs; they are not a list of every dimension supported by NCCL-EP.
No trained model, text dataset, router GEMM, backward pass, or optimizer is used.
The `decode` and `prefill` names describe token batch sizes, not full-model tasks.

### Independent communication mode and layout

`--mode {latency,throughput}` selects the native communication algorithm;
`--layout {expert_major,rank_major,token_major}` selects its receive layout.
Defaults remain **latency + expert_major**. YAML uses `environment.mode` and
`environment.layout`; older configurations missing these fields receive the same
defaults. Each CLI flag overrides only its own field. There is no automatic mode
switch, emulated communication layout, or fallback for an invalid combination.

| Backend | Mode | Supported native layouts |
|---|---|---|
| MSCCL++ | latency | `expert_major`, `rank_major` |
| MSCCL++ | throughput | `token_major`, `rank_major` |
| NCCL-EP | latency (`LOW_LATENCY`) | `expert_major`, `rank_major` |
| NCCL-EP | throughput (`HIGH_THROUGHPUT`) | `expert_major`, `token_major` (native `FLAT`) |

NCCL-EP `RANK_MAJOR` is LL-only; `FLAT` is HT-only. HT expert-major is
expert-grouped **2D slots**, not LL's padded 3D expert tensor. MSCCL++
throughput rank-major has fixed-capacity source-rank zones, unlike its compact
token-major receive view. This benchmark does not expose the separate latency
`rank_major_topk_expanded` layout.

```bash
# Native compact token-major throughput:
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --backend mscclpp --mode throughput --layout token_major --graph mlp

# Native rank-major latency (also supported with --backend nccl):
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --backend mscclpp --mode latency --layout rank_major --graph none

# NCCL-EP native HT flat/token-major:
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --backend nccl --mode throughput --layout token_major --graph mlp
```

All supported pairs use the shared Torch or native FlashInfer compute
mode, eagerly or with an MLP-only graph. The local adapter selects valid expert
assignments using current dispatch IDs, packs expert slots, computes each expert,
and applies routing weights in FP32 before writing BF16 native combine rows.
Same-token selections of multiple local experts are independent until this
reduction. Unused rank-zone metadata is masked using native counters; empty
experts/ranks do not enter GEMMs. Native LL expert-major retains its original
combine-applied weighting. Row selection, packing, weighting and reduction are
**inside the MLP timing interval and graph**, not hidden preprocessing.

Routing/counts stay fixed; MSCCL++ throughput reuses its public dispatch handle
cache and runtime-owned receive pool. NCCL-EP HT uses handle-time counters and
expert offsets with default alignment zero (equivalent to one: no expert padding). Stable data/metadata
addresses are checked during graph replay. Receive capacity, IDs/weights, packed
adapter workspaces and the MSCCL++ fixed native throughput pool are included in
the conservative memory estimate. MSCCL++ throughput is limited to 1024 local
experts, 65536 worst-case received rows, and a 1 GiB receive payload. This benchmark requires even
tokens/rank for NCCL-EP throughput to satisfy native S2D transfer alignment.
All modes require a supported single-node GPU IPC/NVLink topology; installed
runtime API, compiler, hardware and allocation failures are errors, not fallback
requests. `run.json` records requested configuration, resolved communication
mode/algorithm, actual layout (including NCCL's `flat` spelling), and graph scope.

### Optional MX expert compute

Communication and the external input remain **BF16**, independently
of compute precision. The default is still `--compute-backend torch` with BF16
activations and weights. Select the experimental grouped implementation using
three independent flags (no additional environment/model presets):

```bash
# Native mixed blockscaled MXFP8 activations / MXFP4 weights:
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced --backend nccl \
  --compute-backend cutlass_flashinfer --activation-dtype mxfp8 --weight-dtype mxfp4 \
  --graph mlp --warmup 3 --iterations 10
```

Both Torch BF16/BF16 and FlashInfer MXFP8/MXFP4 work with `--backend mscclpp` or `--backend nccl`; choose
`--graph none` for eager compute. YAML optionally supplies
`compute: {activation_dtype: bf16, weight_dtype: bf16}`, separate from
`input.dtype: bf16` and `environment.compute_backend`. Missing compute fields
default to BF16. CLI flags override only their corresponding fields.
`torch` supports **only** BF16/BF16; `cutlass_flashinfer` supports **only** MXFP8/MXFP4.
BF16/MXFP8 weight-only compute is not supported because it has no native MMA path.
Selecting it without changing the default weight format fails with guidance;
plain `fp8`/`fp4` are not aliases for MX formats.

| Activation / weight | Actual implementation at both GEMMs |
|---|---|
| BF16 / BF16 | PyTorch BF16 reference |
| MXFP8 / MXFP4 | FlashInfer **CuTe/CUTLASS DSL**, native E4M3 × E2M1 blockscaled MMA, FP32 accumulation; FC1 fuses SwiGLU and MXFP8 intermediate quantization |

Weights use the same independently seeded gate/up/down matrices as the Torch
baseline, transposed to `[expert,N,K]`, then quantized once outside timing.
Only one expert's BF16 staging matrices are live at a time; no expanded BF16
weight bank is retained. MXFP8 payloads are E4M3; MXFP4 packs two E2M1 values
per byte, low nibble first. Both use E8M0 scale bytes per **32 reduction elements**,
not NVFP4 block16/global scales. CuTe uses 128×4 MMA-swizzled weight/intermediate scales,
linear input scales, and alternating 64 up/64 gate rows.

Valid rows are packed from padded expert slots on **every iteration** and
unpacked afterward. Each compact row remains a separate expert assignment,
including when one token selects multiple experts on the same rank. CuTe's local
finalize uses top-k one and unit weights into compact **expert-slot** outputs,
not original token outputs. The communication adapter still applies routing
weights exactly once. Counts/schedules/indices are precomputed for fixed routing;
empty experts and entirely empty ranks are supported.

The **MLP event interval and MLP graph include packing, initial BF16→MXFP8
quantization, quantized-buffer copies, GEMMs, SwiGLU, intermediate quantization,
output initialization/scatter, and unpacking** as applicable. Initial quantization
is not cached between iterations: graph-private quantizer allocations and copies
are captured and replayed against current BF16 dispatch data. Graphs exclude
dispatch/combine. These are pipeline phase intervals including launches/movement,
**not pure GEMM kernel timings**; no blanket performance advantage is claimed.
`run.json` records actual implementation, payload/scale formats and layouts,
communication dtype, dependency versions, and timed operations.

Current support is restricted to **SM100/SM103 Blackwell**, CUDA 13 PyTorch,
and hidden/intermediate dimensions divisible by 128 (plus the common communication
limits above). Validation used GB300/SM103; SM100 is supported by the selected
FlashInfer kernels but was not exercised here. Experimental lower-level APIs are
pinned by preflight checks to `flashinfer-python==0.7.1`,
`nvidia-cutlass-dsl==4.8.0`.
Install these optional dependencies separately in a compatible environment:

```bash
python -m pip install 'flashinfer-python==0.7.1' 'nvidia-cutlass-dsl==4.8.0'
```

Inspect resolver changes before installing: do not replace a working Torch/CUDA
runtime. The validated environment retained Torch 2.14.0+cu130 and CUDA runtime 13.0.
Torch compute never imports FlashInfer; NCCL-EP still does not require MSCCL++.
There is no fallback on missing dependencies, incompatible versions, or hardware.
Preflight estimates include packed payloads/scales, single-expert staging peak,
activation workspaces and graph-private allocations; native runtime/allocator
allocations are not an exact bound. Ranks collectively reject insufficient memory
before allocating the expert bank.

Correctness tests independently decode scale bytes/nibbles, check known-value
layouts, compare dequantized-operand GEMMs separately from quantization error,
and exercise both adapters, eager/graph, changed inputs, uneven/empty experts,
multiple tiles, and same-token local expert selections. Arithmetic relative-L2
tolerance is 0.004; mixed FC1 quantization-boundary
tolerance is 0.015, and combined mixed distributed tolerance is 0.03.
Observed random-weight total error against unquantized FP32 math was approximately
20–23% for mixed MX; these are **not trained-model accuracy
claims**, nor the arithmetic tolerances.

### NCCL-EP backend

NCCL-EP is optional: MSCCL++ runs do not import or require it. Install the
official NVIDIA bindings separately, selecting the CUDA major matching PyTorch:

```bash
python -m pip install 'nccl-extensions[cu13]'
# For a CUDA 12 PyTorch environment: nccl-extensions[cu12]
python -c 'import nccl.core, nccl.ep'
python test/python/ep/moe_bench/run.py \
  --model small --input decode --backend nccl --routing balanced \
  --warmup 10 --iterations 100
```

The tested bindings are `nccl-extensions==0.1.0` and `nccl4py==0.6.0`, with
NCCL 2.30.7. The extension requires a matching CUDA toolkit (`nvcc` and headers)
for runtime kernel compilation; set `CUDA_HOME` if it is not discoverable.
See the [NVIDIA installation instructions](https://github.com/NVIDIA/nccl-extensions#installation).
The PyTorch NCCL process-group backend alone does **not** provide `nccl.ep`.
There is no fallback when the requested runtime is missing or fails.

Use `--backend nccl` with `--graph none` for eager
execution or `--graph mlp` for the same MLP-only graph mode. For example:

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model scaled_small --input decode --backend nccl --graph mlp \
  --routing balanced --warmup 200 --iterations 1000
```

Both backends use the same input generation, routing, seeded expert matrices,
preallocated buffers, compute, and phase timing. The NCCL unique ID is exchanged
over the existing CPU Gloo group; this benchmark needs no MPI. Run on a supported
single-node NVLink GPU system; multi-node execution is not implemented.
NCCL initialization and any kernel compilation happen before measurement.

Implementation files under `test/python/ep/moe_bench/`:

- `communication_backends/__init__.py`: lazy backend factory and fixed-buffer adapter contract.
- `communication_backends/mscclpp.py`: MSCCL++ native latency and throughput paths.
- `communication_backends/nccl.py`: NCCL communicator, LL/HT handles, dispatch counters,
  weighted combine, version metadata, and ordered resource teardown.
- `worker.py`: shared dispatch → expert SwiGLU → combine pipeline and timings.
- `compute_backends/__init__.py`: lazy compute factory.
- `compute_backends/torch_reference.py` / `compute_backends/flashinfer.py`: default Torch reference and optional grouped MX compute.
- `workload.py`: shared seeded synthetic inputs and expert weights.
- `memory.py`: conservative tensor memory estimates.
- `layout.py`: shared native-row selection, expert packing, and weighted reduction.
- `config.py` / `configs.yaml`: named workload presets and default environment settings.
- `tests/`: CPU unit tests and opt-in distributed reference tests.
  `tests/test_backends.py` covers dependency isolation, API wiring, and cleanup;
  `tests/test_moe_bench.py` covers per-backend distributed mathematical reference.

Combine consumes the **computed expert output**, not the dispatched input.
NCCL-EP LL expert-major applies the original FP32 routing weights when gathering BF16 expert
outputs. Unlike MSCCL++ `RANK_LOCAL_REDUCE`, it does not round intermediate
per-rank partial sums to BF16, so results need not be bitwise identical.
Rank-major/token-major adapters instead round locally reduced rank partials to
BF16; NCCL-EP HT expert-major rounds each weighted expert row to BF16.
`run.json` records the actual backend, algorithm, combine mode, and installed
backend/runtime versions; NCCL runs are not labeled `rank_local_reduce`.

### Workloads and measurements

Balanced routing gives global expert counts differing by at most one. Uniform
routing samples distinct experts per token with equal probability. Hot-expert
routing assigns a probability mass to the specified global IDs before weighted
sampling without replacement. This is not a guaranteed final route fraction:
two hot experts with top-k eight can account for at most 25% of assignments.
Routing weights are positive and normalized per token. Inputs, routing, and
expert matrices remain fixed throughout a run.

Each timed iteration includes dispatch, expert GEMMs, SwiGLU, and combine.
Expert counts are determined once outside timing, so no receive-count CPU
synchronization is required per iteration. Torch computes only valid prefixes;
grouped MX kernels round active expert tiles to 16 or 128 rows, rather than
computing the full padded communication capacity.
Reusable buffers hold GEMM outputs. The default Torch path is a reference implementation:
per-expert Python launches and GPU idle gaps are included in the CUDA-event
interval and can dominate small workloads. It is not optimized grouped GEMM.
Host barriers before iterations and completion synchronization after their end
events are outside timing. Initialization, generation, warmup, and output writing
are excluded. Native CUDA timing events are materialized before warmup; warmup
uses the same phase-event recording path as measured iterations. This prevents
lazy CUDA event creation from contaminating the measured intervals, but does
not remove Python launch gaps or host scheduling effects.

Rank 0 prints mean/p50/p95 GPU elapsed latency per rank and for the maximum rank
duration in each iteration. This maximum is a distributed duration metric, not
a globally synchronized wall-clock measurement. Global throughput is the original
input token count across all GPUs divided by the mean maximum-rank duration;
top-k copies do not count as additional input tokens. Expert/rank assignment
counts and maximum/mean rank load expose routing imbalance.

Dispatch, MLP, and combine are also timed separately on each rank, using four
CUDA events at the boundaries of the three phases on the current stream. There
is no synchronization between phases. MLP includes all expert GEMMs, SwiGLU,
launch gaps, and MX packing/quantization/unpacking where selected, rather than
isolated GEMM kernel time. Communication
times include peer waiting, and event-recording overhead affects the measured
pipeline. Phase durations are adjacent intervals; the total is measured directly
from the first to last event. Phase means/p50/p95 are reported per rank, not as
independently maximized global phases whose sum would misrepresent total latency.

Each run writes three files under
`benchmark_results/<UTC-timestamp>_<unique-id>/` (or `--output-dir`):

| File | Contents |
|---|---|
| `run.json` | Invocation and worker command, selected model/input/routing presets, fully resolved configuration (including environment and benchmark overrides), seed, versions, GPU details, actual algorithm/layout, expert loads, and completion/failure status |
| `summary.csv` | `scope,rank,phase,mean_latency_us,p50_latency_us,p95_latency_us,throughput_tokens_per_s`; one global total row and dispatch/MLP/combine/total rows per rank |
| `iterations.csv` | `iteration,rank,dispatch_us,mlp_us,combine_us,total_us` per measured rank iteration, excluding warmup |

`run.json` uses schema version 2 for these phase-aware CSVs. Throughput is only
populated for the global total summary row. Existing schema-version-1 result
files are not modified.

The launcher records its Python argv, not shell redirections or a full shell
environment. Native runtime/library allocations are not fully covered by the
preflight tensor-memory estimate. Failures are reported on stderr and marked in
`run.json` once a run directory has been created; malformed configurations fail
before launch. Only completed runs should be used for analysis. Activation tensors
and expert matrices are not saved. There are no optional profiling or validation
passes in the user-facing benchmark.

Development coverage includes CPU tests and an opt-in two-GPU mathematical
reference comparison (including repeated calls and different routing loads):

```bash
MOE_BENCH_GPU_TESTS=1 python -m pytest test/python/ep/moe_bench/tests -q
# Optional NCCL-EP coverage (separate worker processes for each backend):
MOE_BENCH_NCCL_GPU_TESTS=1 python -m pytest test/python/ep/moe_bench/tests -q
# Optional MX coverage for BOTH adapters; check GPU availability first:
MOE_BENCH_MX_GPU_TESTS=1 python -m pytest test/python/ep/moe_bench/tests/test_compute_flashinfer.py -q
```

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
There is no ring algorithm or runtime fallback. Set the dispatch/combine block
budgets through the `num_blocks` API configuration.

The persistent throughput configuration contains only:

| Field | Meaning |
|---|---|
| `num_blocks` | Throughput dispatch/combine block counts |

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
├── moe_runtime.cc
├── latency.cc
├── throughput.cc
├── common/
│   ├── device_helpers.cuh
│   ├── latency.cuh
│   ├── overlap_barrier.cuh
│   └── quantization.cuh
├── dispatch/
│   ├── common.cuh
│   ├── expert_major_dispatch.cu
│   ├── rank_major_dispatch.cu
│   ├── throughput_prepare.cu
│   └── throughput_dispatch.cu
├── combine/
│   ├── common.cuh
│   ├── rank_local_reduce_combine.cu
│   ├── direct_send_combine.cu
│   └── throughput_reduce_combine.cu
├── include/
│   ├── config.hpp
│   ├── device_context.hpp
│   ├── exception.hpp
│   ├── kernels.hpp
│   ├── launch.hpp
│   ├── moe_runtime_context.hpp
│   └── recv_pool.hpp

include/mscclpp/ext/ep/
├── moe_runtime.hpp
└── types.hpp
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
