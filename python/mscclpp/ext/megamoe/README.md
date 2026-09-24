# Native CUDA MegaMoE (experimental)

Inference-only SwiGLU experts with BF16 activations, MXFP8 weights, and native
MSCCL++ communication. The extension does not depend on FlashInfer or NVSHMEM.
Torch is required for the Python tensor adapter and benchmarks, not the native
library.

## Build

Requires Linux, SM100 GPUs (for example GB200), CUDA nvcc/ptxas >=13.3, and
runtime/toolkit libraries >=12.8. Ranks must use distinct GPUs within one active
NVLink fabric, including for multi-host runs. PCIe-only, InfiniBand-only, ROCm,
and other GPU architectures are unsupported.

From the repository root:

```bash
python -m pip install -e . -Ccmake.define.MSCCLPP_BUILD_EXT_MEGAMOE=ON
```

MegaMoE is disabled by default. CMake fetches CUTLASS at
`147295a3d4b75f3aeff247c25b8927cea9a7006a`; use
`-Ccmake.define.MSCCLPP_MEGAMOE_CUTLASS_ROOT=/path/to/cutlass` to supply it locally.
The native library is compiled for `sm_100a` independently of the core library's
architecture list. Install a compatible Torch CUDA wheel separately.

`src/ext/megamoe/megamoe.cu` constructs pipelines and dispatches warp roles.
Compile-time tuning policy is isolated in `megamoe_specialization.hpp`: routed
M/N/K tiles, pipeline depths, and the routed/local warp schedules. The schedules
assign epilogue, MMA, LoadA, LoadB, dispatch, and transform work; compile-time
checks enforce the fixed four-warp compute groups and nonoverlapping roles.
Fixed role implementations live in `megamoe_roles.cuh`, while
`megamoe_collective.cuh` defines the CUTLASS pipelines. `megamoe_launch.cu` owns
workspace layout, weight packing, plans, and launches; `megamoe_jit.cu` owns the
JIT C ABI entrypoint. Other internal headers separate device state
(`megamoe_device.cuh`), routing/dispatch (`megamoe_routing.cuh`), and
SwiGLU/epilogue/top-k combine (`megamoe_epilogue.cuh`). All three CUDA files and
their headers ship in the JIT source bundle.

## API

Set the process's CUDA device before constructing its MSCCL++ bootstrap and
communicator. Context construction is collective and packs weights once:

```python
from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig, quantize_mxfp8

config = MegaMoEConfig(
    rank=rank, world_size=world_size, max_tokens=32,
    hidden=4096, intermediate=4352, num_experts=16 * world_size,
    top_k=7, sm_margin=32,
)
# fc1_bf16: [local_experts, 2*I, H], with gate rows before up rows.
# fc2_bf16: [local_experts, H, I].
fc1, fc1_scale = quantize_mxfp8(fc1_bf16)
fc2, fc2_scale = quantize_mxfp8(fc2_bf16)
moe = MegaMoE(config, communicator, fc1, fc1_scale, fc2, fc2_scale)
output = moe(inputs_bf16, routing_int32, router_weights_float32)
```

`H` and post-SwiGLU `I` must be positive multiples of 128. Supported world sizes
are 1-72; experts divide evenly across ranks with contiguous ownership.
`1 <= top_k <= min(32, E)` and `0 <= T <= max_tokens`.

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Input / output | `[T,H]` | BF16 |
| Routing IDs | `[T,top_k]` | int32 |
| Routing weights | `[T,top_k]` | FP32 |
| FC1 / FC2 weights | `[local_experts,2*I,H]` / `[local_experts,H,I]` | FP8 E4M3FN |
| Scales for a weight matrix `[local_experts,M,K]` | `[local_experts,M,K//32]` | uint8 E8M0 |

All tensors must be contiguous on the owning GPU. Scales use canonical K32
blocks, not a backend-specific swizzle. For E5M2 weights, pass `e5m2=True` to
quantization and `weight_e5m2=True` in the config.

Routing IDs must be in `[0,E)` and weights finite. Optional
`validate_routing=True` checks values with a host synchronization and cannot be
used during capture; shape, dtype, and device checks always run. Routing weights
multiply the FP32 SwiGLU result before the BF16 FC1-to-FC2 handoff. FC2 partials
are BF16; top-k combination sums in FP32 and returns BF16.

`sm_margin` caps persistent CTA usage: margin 32 on a 152-SM GB200 permits at most
120 CTAs. `moe.cta_count` reports the actual count after occupancy and two-CTA
cluster alignment. This is not a fixed SM-ID partition or exclusive reservation.
`moe.workspace_bytes` reports workspace sizes, excluding packed weights.

For routing capacity `world_size * max_tokens * top_k <= 1024` and at most
128 local experts, routing preparation uses one CTA with cached peer headers,
IDs and weights, a shared-memory histogram, and a warp-parallel prefix. It
publishes the completed plan once, reducing preparation from five full-grid
joins to two without changing peer readiness or output-completion requirements.
The scratch storage reuses the epilogue allocation. Larger capacities/expert
counts retain the distributed planner; selection is automatic and capture-safe.

### Local shared expert

Construct a separate context with `world_size=1`, `num_experts=1`, and `top_k=1`,
using an independent one-rank bootstrap and communicator. Do not reuse the global
communicator with an overridden world size.

```python
shared_output = shared.forward_shared(
    inputs_bf16, output=preallocated_output, stream=shared_stream,
)
```

This unweighted path requires no routing metadata and skips token dispatch,
peer synchronization, and top-k combination. Ordinary `forward` also uses the
local kernel for this configuration but retains routing weights.

## Streams and CUDA Graphs

Construction, registration, peer exchange, and weight packing must occur outside
capture. Use preallocated outputs for graph replay:

```python
workspace_input = moe.input_view(T)
# Produce input into workspace_input on stream before launching.
moe(workspace_input, ids, scores, output=out, stream=stream)
```

Passing an exact `input_view` alias skips input staging; ordinary inputs are
staged on every invocation. Output must not overlap the registered workspace.

Every rank must invoke the same collective sequence, although token counts may
differ or be zero. A context is **not concurrently reusable**: order forwards
and graph replays on one stream, or use explicit event dependencies. Order tensor
producers before the launch stream and retain the context, graph, and input/output
buffers through all replays.

For routed-first overlap, call `forward(..., signal_start=True)` followed by
`wait_until_started(shared_stream)`. This graph-compatible wait establishes
**kernel entry, not output readiness**. Join the consumer stream before reusing
the context.

Before teardown, synchronize each device and perform an all-rank bootstrap
barrier so no peer access remains outstanding. Destruction itself is not
collective. Input views retain the native context through their storage.

## Benchmarks

Launch one rank per GPU. Torch distributed Gloo handles setup, untimed reference
checks, and reporting; timed communication uses MSCCL++. The native TCP bootstrap
port defaults to `MASTER_PORT+1`, configurable with `--bootstrap-port`.

### Routed experts only

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.benchmark \
  --tokens 32 --hidden 4096 --intermediate 4352 --experts 64 --top-k 7 \
  --sm-margin 32 --check --input-mode staged
```

This measures staging, dispatch, expert GEMMs, SwiGLU, return, and combination.
It excludes router, projections, shared experts, postnorm, and residual.
Effective bandwidth is local weights-plus-scales bytes divided by latency, not
hardware HBM traffic.

`--input-mode direct` initializes registered inputs before timing; it does not
include their producer. `--no-graph` selects ordinary launches instead of CUDA
Graphs. `--tile-m`, `--tile-n`, `--tile-k`, `--load-stages`, and
`--transform-stages` select a routed JIT specialization; defaults select the
precompiled kernel.

### FlashInfer MegaMoE comparison

The GB200-only companion benchmark uses FlashInfer's latest public
`Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig` backend with the same default
tokens, H/I, expert count, top-k, routing weights, graph batching, and
cross-rank latency reduction:

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.benchmark_flashinfer \
  --flashinfer-root /home/azhpcuser/mai/flashinfer \
  --flashinfer-cache-root /tmp/flashinfer-megamoe-cache \
  --tokens 32 --hidden 4096 --intermediate 4352 --experts 64 --top-k 7 \
  --check --json-output results/flashinfer-megamoe-ep4.json
```

`--flashinfer-root` defaults to `FLASHINFER_ROOT` and then a sibling
`../flashinfer` checkout. The benchmark requires the checkout's current Python
dependencies and records its exact git commit in JSON. FlashInfer's collective
`warmup()` performs compilation, symmetric-workspace setup, and optional
`--autotune` before timing. The timed public API uses
`return_workspace_view=True`, so it includes BF16 input staging, fused
dispatch/expert/return work, and top-k reduction without an output allocation.
Use `--knobs-json` for pinned FlashInfer kernel knobs and
`--in-kernel-fc2-reduce` to enable the corresponding candidate family.
For a single-node launch, the benchmark defaults NVSHMEM to local P2P transport
(`NVSHMEM_REMOTE_TRANSPORT=none`) and NCCL bootstrap to sockets; explicit
environment values override these defaults.
The GB200 validation environment follows FlashInfer's SM100 tuning notes and
uses `nvshmem4py-cu13==0.3.1`; JSON records the actual runtime package versions.
For multi-node GB200 MNNVL/IMEX runs, keep the expert data path off IB:

```bash
export NCCL_GIN_TYPE=3
export NCCL_MNNVL_ENABLE=1
export NCCL_IB_DISABLE=1
export NVSHMEM_REMOTE_TRANSPORT=none
```

Launch one `torchrun` process group across the nodes with four local processes
per node. The Ethernet interface is used only for rendezvous/control-plane
traffic. If the distributed FlashInfer source copy omits `.git`, set
`FLASHINFER_SOURCE_COMMIT` to the source checkout's full commit so the JSON
still records it.

For an end-to-end synthetic-layer comparison against `benchmark_shared`, use
the same router, squash/unsquash projections, native local shared expert,
post-RMSNorm, residual, graph batching, and four schedule modes while replacing
only the routed expert backend with FlashInfer:

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.benchmark_flashinfer_shared \
  --flashinfer-root /home/azhpcuser/mai/flashinfer \
  --flashinfer-cache-root /tmp/flashinfer-megamoe-cache \
  --check --json-output results/flashinfer-megamoe-layer-ep4.json
```

The frontend and shared branch are identical to `benchmark_shared`. FlashInfer
does not expose the native routed CTA/SM cap or kernel-entry signal, so its
`overlap` schedule is routed-enqueue-first and gates the shared stream after
input/router/squash producers, rather than after routed kernel entry. JSON
records this distinction; `routed-only`, `shared-only`, and `serial` have the
same stage boundaries as the native benchmark.

### Complete synthetic MoE layer

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --check --json-output results/megamoe-ep4.json
```

Default configuration:

| Component | Shape / policy |
| --- | --- |
| Input | BF16 `[32,8704]` per rank |
| Router | FP32 tensors, TF32 allowed; full softmax, top-7, selected-probability renormalization |
| Squash / unsquash | BF16 `8704 -> 4096` / `4096 -> 8704` |
| Routed experts | `H=4096`, post-SwiGLU `I=4352`, 16 experts/rank, MXFP8 weights |
| Shared expert | Original input, `H=8704`, `I=2048`, one local MXFP8 expert |
| Postprocess | BF16 branch sum -> FP32 RMSNorm with FP32 gamma -> BF16 handoff -> FP32 residual add |

Postnorm epsilon defaults to `1e-6`; the final output is FP32. The residual is a
separate synthetic skip buffer, not the branch sum. Controls include
`--no-post-norm`, `--no-residual`, `--rms-eps`, and `--residual-dtype {fp32,bf16}`.
Without residual addition the output is BF16; disabling postnorm leaves a BF16
pass-through copy.

Router weights are contiguous `[E,H]`, used as `inputs.float() @ weight.t()`.
TF32 is enabled only around that GEMM and the surrounding precision setting is
restored. Allowing TF32 can change near-tied selections; actual kernel choice
depends on the backend and environment. Use `--no-router-allow-tf32` for a strict
router GEMM. `--router-weight-layout {expert-major,hidden-major}` and
`--router-probability-order {softmax-first,selected-logits}` allow controlled
comparisons. Selected probabilities are renormalized with epsilon zero. Routing
uses Torch operations, not a production fused top-k kernel.

The benchmark reports four schedules on identical inputs and weights:

| Schedule | Timed scope |
| --- | --- |
| `routed-only` | Layer with router, projections, postprocess, and no shared branch |
| `shared-only` | Local shared expert and input staging only |
| `serial` | Complete layer with sequential routed/shared branches |
| `overlap` | Complete layer; shared starts after routed kernel entry, with an explicit final join |

On GB200, the default `--route-sm-margin 32 --shared-sms 32` requests 120 routed
and 32 shared CTAs. Router, projections, staging, and enabled postprocessing run
inside every timed invocation. This is **not full-model timing**: real weights,
attention, prenorm, residual gathering, and training operations are excluded.

### Native performance

These reference measurements predate the small-capacity routing planner.
Measured with `benchmark_shared` on GB200 using Torch 2.11.0+cu130, the default
geometry and precision above (FP32 router with TF32 allowed), and 120/32
routed/shared CTAs. EP4 uses one four-GPU host and 64 experts; EP32 uses eight
four-GPU hosts in one NVLink fabric and 512 experts.

| Schedule | EP4 latency | EP32 latency |
| --- | ---: | ---: |
| Layer without shared (`routed-only`) | 296.11 us | 310.84 us |
| Shared expert only (`shared-only`) | 47.18 us | 47.84 us |
| Complete layer, serial | 348.04 us | 363.27 us |
| Complete layer, routed-first overlap | **312.17 us** | **325.49 us** |

These are unprofiled CUDA-event medians with `--graph-batch 20 --warmup 10
--iterations 30`: each graph repeats the same input, and each sample takes the
maximum latency across ranks before computing the median. Layer timings include
router, projections, staging, enabled expert branches, postnorm, and residual;
`routed-only` here is not an expert-kernel-only measurement.

### Multi-host runs and measurement

For four GPUs per host, set `NNODES=2`, `4`, or `8` for EP8, EP16, or EP32.
Use the same reachable `MASTER_ADDR` and `MASTER_PORT` on every host, and a unique
`NODE_RANK` in `[0,NNODES)`. All GPUs must belong to the same NVLink fabric.
Run on every host:

```bash
torchrun --nnodes="$NNODES" --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --check --json-output results/megamoe-layer.json
```

Global experts default to `16 * WORLD_SIZE`. Use a bounded distributed launcher
that terminates all ranks if a peer fails.

`--warmup`, `--iterations`, and `--graph-batch` control measurement. CUDA-event
latency covers each graph replay, including fork/join, divided by the number of
invocations. JSON includes raw rank samples, sample-wise maximum-across-ranks
latency, shapes, precision policy, CTA counts, workspace sizes, and check results.
`--check` runs untimed numerical references and changed-input/residual graph
checks; it needs roughly 1 GB additional host memory per rank at default shapes.

Add `--trace-path results/trace.json` on every rank to profile after timing;
rank zero exports the trace. `--trace-eager` adds host stage labels instead of
profiling graph replays. Overlap attribution requires distinct routed/shared CTA
counts. Profiler samples can be distorted and must not replace unprofiled latency.

### JIT kernel specializations

JIT compiles the **same native CUDA template** with different output-feature,
token, and reduction tiles plus pipeline depths, leaving two-CTA clusters, two
accumulator stages, packed conversion, numerical semantics, and the local shared
kernel unchanged. Routed `tile_m` supports 128 and 256, and `tile_k` supports
32, 64, and 128; M128 requires K64 or K128 so every scale transaction remains
128-byte aligned. The local shared kernel remains M256/K128. The default
M256/N32/K128/load8/transform7 kernel remains precompiled and needs no compiler.

```python
from mscclpp.ext.megamoe import KernelConfig, compile_kernel, MegaMoE

kernel = compile_kernel(
    KernelConfig(tile_m=128, tile_n=64, load_stages=6, transform_stages=6, tile_k=64)
)
moe = MegaMoE(config, communicator, fc1, fc1_scale, fc2, fc2_scale, kernel=kernel)
```

Prepare modules outside collective construction and CUDA Graph capture.
`load_stages` jointly controls raw weights, scales, and activation prefetch;
`transform_stages` controls converted weights in TMEM. M, N, K, and stage counts
must fit TMEM, compiled shared memory, registers, and resident-cluster limits.
`moe.kernel_id`, `moe.kernel_config`, and `moe.shared_bytes` report the selection.

Set `MSCCLPP_MEGAMOE_CUTLASS_ROOT` to the compatible CUTLASS checkout.
`MSCCLPP_MEGAMOE_NVCC` selects nvcc >=13.3 and `CUDA_HOME` selects runtime
development libraries (default `/usr/local/cuda`). For separately installed
compiler/runtime packages, `MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS` accepts matching
runtime include directories separated by `:`. Do not override the compiler's
CCCL with an older toolkit's headers.

The cache defaults to `$XDG_CACHE_HOME/mscclpp/megamoe` (or
`~/.cache/mscclpp/megamoe`); override it with `MSCCLPP_MEGAMOE_CACHE_DIR`.
File locking avoids duplicate builds on a host. Modules and manifests use
content-addressed keys and checksums; compilation failures are explicit and
logs are retained. `load_cached_kernel(key)` reuses a compatible module without
nvcc or a CUTLASS checkout. Installed JIT sources must remain available for
fingerprint validation. Change `MSCCLPP_MEGAMOE_CACHE_TAG` when changing an
external performance policy such as GPU clocks or power limits.

Different variants require separate contexts and workspace layouts. Keep their
graphs/contexts alive while in use; neither forward nor replay compiles, tunes,
or changes the selected variant. All ranks must select the same variant.

### Offline autotuning and profile reuse

[`megamoe_tuning.json`](megamoe_tuning.json) lists kernel candidates, resource
splits, shape overrides, and inclusive token buckets with representative samples.
The default keeps M256/K128 and tunes N32/load8/transform7,
N32/load6/transform7, N64/load6/transform6, and N128/load4/transform4 at the
same 32/32 resource split. Add `tile_m` or `tile_k` variants to a custom tuning
file to search workload-specific M128, K32, or K64 variants. M128 is not in the
default search because it increases task count substantially for the default
H4096/I4352 shape. The shared kernel is not retuned.

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.autotune \
  --profile-output results/megamoe-tuned.json --topology-id nvlink-partition-a \
  --graph-batch 20 --warmup 10 --iterations 30 --trials 3
```

Use `--config /path/to/tuning.json` to change the search. Token samples come from
that file, not `--tokens`; all contexts use the bucket's upper-bound capacity.
Shape and frontend flags match `benchmark_shared`. Add workload objects such as
`{"hidden":4096,"intermediate":4352,"local_experts":16}` to tune additional shapes.
Use separate buckets around performance crossovers, for example 0-32, 33-64,
and 65-128. A bucket reuses the kernel configuration, not a variable-shape CUDA
Graph: capture separate graphs when actual input shapes or launch arguments change.
Run separately for each EP size. Multi-host runs use the same rendezvous pattern
as the benchmark, a shared topology identifier, and identical source/toolchains.

The tuner first prepares modules and agrees on their IDs across ranks, then
checks numerical correctness, changed-input graph replay, empty and unequal-rank
token counts. It rotates candidate order across trials and measures complete
routed-first layers. The objective minimizes the worst representative-sample
latency ratio against one common builtin reference. Raw timings, failures,
actual resource usage, and the selected configuration are saved atomically on
each node. This selects the best measured candidate, not a global optimum for
every routing distribution or token count within the bucket.

Reuse a profile without compiling or measuring:

```bash
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --tokens 32 --graph-batch 20 --check \
  --tuned-profile results/megamoe-tuned.json --topology-id nvlink-partition-a
```

Selection matches hardware/topology, EP size, dimensions, experts/top-k, frontend
precision, execution policy, and build/environment fingerprints exactly.
Missing/stale profiles, missing local modules, and out-of-range buckets are
errors, not triggers for online retuning or silent fallback. Every node needs
its saved profile and local JIT cache. `autotune.collect_selection_key()` and
`autotune.resolve_profile()` expose the same selection for application setup;
reuse the returned kernel and capacity before constructing contexts/graphs.

## Tests

From the repository root with the extension installed:

```bash
MSCCLPP_TEST_MEGAMOE_SHARED=1 python -m pytest --noconftest \
  python/test/test_megamoe.py python/test/test_megamoe_shared.py -q
```

GPU cases require SM100. Omitting `MSCCLPP_TEST_MEGAMOE_SHARED` skips the opt-in
shared/router GPU cases. Single-GPU tests do not replace multi-rank `--check`
and CUDA Graph validation on the target fabric.

JIT cache/profile host tests run without compilation. Set
`MSCCLPP_TEST_MEGAMOE_JIT=1` with the JIT toolchain configured to also exercise
non-default kernels, module lifetime, and graph replay:

```bash
python -m pytest --noconftest \
  python/test/test_megamoe_jit.py python/test/test_megamoe_autotune.py -q
```

Routing-planner boundary, masking, and changing-routing graph tests:

```bash
MSCCLPP_TEST_MEGAMOE_ROUTING=1 python -m pytest --noconftest \
  python/test/test_megamoe_routing.py -q
```

The same file supports two/four-rank `torchrun` execution. Add
`MSCCLPP_TEST_MEGAMOE_JIT=1` to cover each routed JIT specialization.
