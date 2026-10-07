# Native CUDA MegaMoE (experimental)

Inference-only SwiGLU experts with BF16 API activations, MXFP8 or MXFP4
weights, and native MSCCL++ communication. The extension does not depend on
FlashInfer or NVSHMEM. Torch is required for the Python tensor adapter and
benchmarks, not the native library.

## Build

Requires Linux, SM100-family GPUs (compute capability 10.0, 10.3, or 10.7), CUDA
nvcc/ptxas and runtime/toolkit libraries >=13.0. Ranks must use distinct GPUs within one active
NVLink fabric, including for multi-host runs. PCIe-only, InfiniBand-only, ROCm,
and other GPU architectures are unsupported.

From the repository root:

```bash
python -m pip install -e . -Ccmake.define.MSCCLPP_BUILD_EXT_MEGAMOE=ON
```

MegaMoE is disabled by default. CMake fetches CUTLASS at
`147295a3d4b75f3aeff247c25b8927cea9a7006a`; use
`-Ccmake.define.MSCCLPP_MEGAMOE_CUTLASS_ROOT=/path/to/cutlass` to supply it locally.
The native library contains an `sm_100f` family cubin independently of the core
library's architecture list. It is compatible with compute capabilities 10.0,
10.3, and 10.7. Install a compatible Torch CUDA wheel separately.
CUDA 13.3 and newer use packed FP8/E8M0-to-BF16 conversion; CUDA 13.0-13.2
automatically use the compatible CUTLASS conversion path.

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

W8A16 and W4A8 share the `KernelParameters` aggregate, `TilePolicy`, epilogue
scratch layout, task indexing, and epilogue warp loop. Their common epilogue
handles TMEM reads, SwiGLU staging, BF16 output packing and peer stores; compile-time
branches preserve each precision's activation handoff and router-weight placement.
Per-store waits require only shared-source consumption for both precisions.
Routed W8A16 still drains destination completion at each chunk boundary, before
publishing FC1 `hiddenReady` or reaching the peer-completion join. Local W8A16
and W4A8 drain shared-source reads per chunk and destination writes at role exit.
Host parameter construction, weight-value packing, resource checks and cluster
launch configuration are also shared. W8A16's weight transform and W4A8's
block-scaled MMA/chunk-ready dispatch remain separate algorithms in
`megamoe_roles.cuh` and `megamoe_w4a8_roles.cuh`, respectively.

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

For native W4A8, set `weight_mxfp4=True` and use `quantize_mxfp4`. Packed
weights are uint8 `[local_experts,2*I,H//2]` / `[local_experts,H,I//2]`;
the even K element occupies the lower nibble. BF16 input rows are quantized to
MXFP8 E4M3 plus E8M0 K32 scales before dispatch. Both values and scales use bulk
peer pulls; scale rows are internally padded to 16 bytes, staged in shared
memory, and packed locally into the MMA scale layout. FC1 uses FP32 accumulation and
SwiGLU, then quantizes the unweighted hidden row to MXFP8. FC2 accumulates in
FP32, applies the FP32 router weight, and emits BF16 partials for the existing
FP32 top-k combination.

Dispatch uses one 3 KiB input buffer with the corresponding K32 scales.
Each chunk is published only after its values have reached the local input pool
and its scales are visible in the MMA layout. The activation loader waits for all
valid rows of the current N tile's chunk, then issues its K128 TMA loads; MMA
consumes completed pipeline stages without waiting for the rest of the H dimension.
The last chunk also marks the full row ready. Chunk counters are reset on every
forward, including CUDA Graph replay; partial chunks retain the H/I multiple-of-128
contract.

All routed kernels publish each rank's live token count and every
`{expert ID, router weight}` pair as epoch-tagged LL16 packets. A
live-token-sized set of planner CTAs maps each token once and loops over its
top-k packets, coordinating count, offset, and fill phases through device epoch
flags rather than a cross-rank or grid barrier. Input staging or quantization
precedes the main kernel on each rank's launch stream. The main kernel then
publishes route packets, and every destination consumes the current-epoch
packets before dispatch, so separate rank- or token-level input-ready flags are
unnecessary. W4A8 input quantization handles activation values and scales only.
Aligned inputs use one 16-byte `int4` load for eight BF16 values per thread, four
threads per K32 scale group, and one packed 64-bit FP8 store. Direct BF16 input
pointers must be 16-byte aligned; cross-input aliases retain the ordered-copy
path through aligned internal storage. Routing IDs and weights are passed to
the main kernel, which stages them and publishes epoch-tagged packets at routing
startup. Exact input-buffer aliases remain supported. Only live routing rows
are consumed, and masked slots are excluded from combination rather than
requiring a full partial-buffer clear.
The final completing CTA publishes GPU completion after acquiring all preceding
CTA arrivals; every CTA waits for peer completion before reducing peer-written partials.

The W4A8 path is routed-only and uses the builtin SM100-family
M256/N64/K128/load9 block-scaled specialization with 16 warps. Separate weight
and activation loader warps share a nine-stage pipeline. Both must publish their
TMA transaction counts before a stage can become ready. Mainloop warps retain
128 registers, independently of the 32-register dispatch warps. FC1 quantization
uses warp reductions over live token groups without staging the activated values
back through shared memory or iterating through inactive groups.
MMA rounds the live token count up to 16 rows, and the second CTA's activation
load shifts to the corresponding runtime half-tile. Accumulator and scale storage
retain their allocated pitch. This reduces padded MMA work without reloading
weights for a second 32-token tile when an expert receives slightly more than
32 tokens. Task-index division is prepared on the host rather than recomputed
inside every device role. Experimental N32 configurations retain a 64-row storage
pitch for even TMEM scale-column alignment. K tails are zero-filled by TMA,
preserving support for H/I divisible by 128.
FC2 peer stores wait only for shared-source consumption between chunks; every
issuing epilogue warp drains destination completion before the CTA publishes its
arrival. This overlaps return traffic without weakening peer visibility.
E8M0 scale selection compares FP32 exponent/mantissa bits directly, preserving
the ceiling-power-of-two rule without a floating-point division.
Unclamped EP4/E64 and EP32/E512 contexts with capacity 32, 64, or 128, top-8,
H9216, and I4096 or I4608 select configuration-specialized kernels that assume
every rank has the same live token count. These kernels calculate route offsets
directly; other W4A8 configurations read token-count packets and build dynamic
prefixes.
Fixed-token routing retains each thread's route through the count/fill phases
when planner capacity permits, and falls back to rereading routes with smaller
CTA budgets. Warp-aggregated expert updates share count and cursor atomics, and
warps distribute block construction for experts spanning multiple token blocks.
Count and arrival counters are cleared after their last use for the next ordered
forward, eliminating the fixed-token initialization epoch wait while retaining
the offset and final routing-ready publication.
Launch resources are borrowed from grid-constant parameters rather than copied
into thread-local memory.
Other capacities, intermediate widths, expert/top-k counts, and clamped
activations use runtime configuration within a compiled hidden specialization.
Current W4A8 hidden specializations are 128, 384, 2176, 4096, 8704, and 9216.
Adding another hidden size requires adding an explicit `megaMoeW4A8<Hidden>`
case in `w4a8KernelEntry`; there is no dynamic `Hidden=0` W4A8 kernel.
Local shared experts and routed
JIT specializations remain W8A16; passing a custom `KernelConfig` with
`weight_mxfp4=True` is rejected.

For an experimental native warp timeline, build with
`-DMSCCLPP_MEGAMOE_W4_TRACE=1`. This enables bounded `%globaltimer` records on
the first two CTAs, with separate routing, dispatch, loader acquire/issue, MMA
wait/issue, epilogue, output-join, and local top-k reduction ranges. The debug
`mscclpp_megamoe_w4_trace_reset/copy` exports must be called only after the
context's CUDA work is synchronized. This is native C++ instrumentation, not
CuTe DSL's `run-iket`; normal builds compile it out. Trace durations include
instrumentation overhead and must not be used as uninstrumented latency.
`-DMSCCLPP_MEGAMOE_W4_TRACE=2` retains coarse ranges but removes per-K timestamp
reads and readiness probes, for lower-perturbation graph-phase measurements.
`-DMSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES=1` separates weight/SFA and activation/SFB
barriers for diagnosis with two loaders. The MMA waits for both inputs and
releases each buffer only after its asynchronous use; the default shared-barrier
schedule remains unchanged. Per-input waits are sequential observations, not
independent TMA transfer-duration measurements.

For controlled compile-time experiments, the `MSCCLPP_MEGAMOE_W4_` macros in
`megamoe_specialization.hpp` select the token/K tiles, load stages, warp count,
mainloop register budget, epilogue token chunk, and dispatch buffer geometry.
Keep definitions consistent across a library's translation units and across ranks.

Routing IDs must be in `[0,E)` and weights finite. Optional
`validate_routing=True` checks values with a host synchronization and cannot be
used during capture; shape, dtype, and device checks always run. For W8A16,
routing weights multiply the FP32 SwiGLU result before the BF16 FC1-to-FC2 handoff; W4A8 applies
them to the FP32 FC2 result after intermediate MXFP8 quantization. FC2 partials
are BF16; top-k combination sums in FP32 and returns BF16.

`sm_margin` caps persistent CTA usage: margin 32 on a 152-SM GB200 permits at most
120 CTAs. `moe.cta_count` reports the actual count after occupancy and two-CTA
cluster alignment. This is not a fixed SM-ID partition or exclusive reservation.
`moe.workspace_bytes` reports workspace sizes, excluding packed weights.
Use `sm_margin=0` for routed-only measurements; reserve CTAs only when another
concurrent branch needs them.

For routing capacity `world_size * max_tokens * top_k <= 1024` (W8A16) or
`<= 4096` (W4A8), and at most 128 local experts, routing preparation uses one
CTA with cached peer headers,
IDs and weights, a shared-memory histogram, and a warp-parallel prefix. It
publishes the completed plan once, avoiding the distributed planner's histogram,
prefix, and route-construction grid joins without changing peer readiness or
output-completion requirements.
The histogram/prefix scratch reuses the epilogue allocation. W4A8 stages
peer IDs and weights in the otherwise idle GEMM tensor buffers; the final routing
grid join completes all metadata reads before GEMM reuses that storage. This
fits 4096 route slots without increasing the kernel's shared-memory allocation.
Larger capacities/expert
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
  --sm-margin 0 --check --input-mode staged
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

Graph timing defaults to `--graph-timing isolated`: the first collective can
include rank-to-rank CPU submission skew after the Gloo rendezvous.
Use `--graph-timing steady-state` to queue one untimed replay before each timed
replay on the same stream. The timed replay still includes the full forward,
including W4A8 input quantization, but excludes the initial host submission skew.
These are different measurement scopes, not a kernel optimization; retain the
isolated result when reporting independent-call latency.

Add `--mxfp4` to benchmark the routed W4A8 path. This keeps BF16 public inputs,
includes the input MXFP8 quantization kernel in timing, and reports the fixed
effective M256/N64/K128/load9 specialization. It cannot be combined with
`--e5m2` or nondefault JIT tile/stage flags.

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
`MSCCLPP_MEGAMOE_NVCC` selects nvcc >=13.0 and `CUDA_HOME` selects runtime
development libraries (default `/usr/local/cuda`). For separately installed
compiler/runtime packages, `MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS` accepts matching
runtime include directories separated by `:`. Do not override the compiler's
CCCL with an older toolkit's headers.

The cache defaults to `$XDG_CACHE_HOME/mscclpp/megamoe` (or
`~/.cache/mscclpp/megamoe`); override it with `MSCCLPP_MEGAMOE_CACHE_DIR`.
File locking avoids duplicate builds on a host. Modules and manifests use
content-addressed keys and checksums; compilation failures are explicit and
logs are retained. Kernel cache keys cover native libraries, JIT sources,
headers, and toolchains, but not benchmark-only frontend code. Offline profile
selection separately fingerprints the frontend, so a frontend change rejects a
stale profile without recompiling an otherwise identical kernel.
`load_cached_kernel(key)` reuses a compatible module without nvcc or a CUTLASS
checkout. Installed JIT sources must remain available for fingerprint
validation. Change `MSCCLPP_MEGAMOE_CACHE_TAG` when changing an external
performance policy such as GPU clocks or power limits.

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

The tuner first prepares modules and agrees on their IDs across ranks. Before
timing, every candidate checks changed-input eager/graph and serial/overlap
equivalence. Each kernel specialization runs one independent CPU oracle for the
representative samples, even when the same kernel is paired with multiple
resource splits. Empty and unequal-rank cases check finite outputs and schedule
equivalence without repeating the expensive CPU oracle; add
`--full-edge-references` to restore independent CPU oracles for those edge
cases. It rotates candidate order across trials and measures complete
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

### Packaged resident tuning results

[`megamoe_resident_tuning.json`](megamoe_resident_tuning.json) stores measured
resident-kernel winners for the exact 32-rank GB200 W4A8 H9216/E512/top-8
configuration. It records the complete builtin kernel policy, TMA cache hints,
token/intermediate-specific CTA margins, measurements, and provenance.
Resolution is opt-in and never compiles or benchmarks:

```python
from mscclpp.ext.megamoe import MegaMoEConfig, resolve_resident_tuning

config = MegaMoEConfig(
    rank=rank, world_size=32, max_tokens=tokens,
    hidden=9216, intermediate=intermediate,
    num_experts=512, top_k=8, weight_mxfp4=True,
)
tuning = resolve_resident_tuning(config, graph_batch=10)
config = tuning.apply(config)
```

The packaged winners are:

| Intermediate | T32 | T64 | T128 |
| ---: | ---: | ---: | ---: |
| 4096 | `sm_margin=4` | `sm_margin=4` | `sm_margin=2` |
| 4608 | `sm_margin=4` | `sm_margin=6` | `sm_margin=2` |

The resolver requires an exact match for GPU name/capability/SM count, EP size,
dimensions, experts/top-k, W4A8 precision, capacity, and graph timing. Missing
coverage raises `ResidentTuningMismatchError`; it never silently falls back or
changes the default `sm_margin=0`. This resident profile is distinct from
`megamoe_tuning.json`, which defines the generic offline autotune search space.

## Tests

From the repository root with the extension installed:

```bash
MSCCLPP_TEST_MEGAMOE_SHARED=1 python -m pytest --noconftest \
  python/test/test_megamoe.py python/test/test_megamoe_shared.py -q
```

GPU cases require the SM100 family. Omitting `MSCCLPP_TEST_MEGAMOE_SHARED` skips the opt-in
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
