# Native CUDA MegaMoE (experimental)

This extension implements routed SwiGLU experts in native CUDA, with MSCCL++
registered buffers and direct peer mappings. It does **not** load FlashInfer,
NVSHMEM, a Python DSL kernel, or a Torch C++ extension. Torch is only the optional
Python tensor, reference, and benchmark frontend.

## Build

Requires Linux, CUDA nvcc/ptxas 13.3 or newer (runtime/toolkit libraries >=12.8),
CUTLASS with the SM100 mixed-input collective,
and SM100 GPUs (for example GB200). All ranks must use distinct GPUs within one
active NVLink fabric. PCIe-only, InfiniBand-only, ROCm, and other GPU architectures
are rejected; there is no silent fallback.

CMake fetches standalone CUTLASS pinned at
`147295a3d4b75f3aeff247c25b8927cea9a7006a` unless the root below is supplied.

```bash
python -m pip install -e . \
  -Ccmake.define.MSCCLPP_BUILD_EXT_MEGAMOE=ON \
  -Ccmake.define.MSCCLPP_MEGAMOE_CUTLASS_ROOT=/path/to/cutlass
```

The default build leaves MegaMoE disabled. A direct CMake build may instead set
`-DMSCCLPP_BUILD_EXT_MEGAMOE=ON`; the separate `mscclpp_megamoe` shared-library target
is compiled for `sm_100a` regardless of the core library's architecture list.
There is no Torch ABI dependency in the library or nanobind binding.

## API

```python
from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig, quantize_mxfp8

# Set this process's CUDA device BEFORE constructing its MSCCL++ bootstrap.
config = MegaMoEConfig(
    rank=rank, world_size=world_size, max_tokens=32,
    hidden=4096, intermediate=4352, num_experts=64, top_k=7, sm_margin=32,
)
# fc1_bf16: [local_experts, 2*I, H]; gate rows precede up rows.
# fc2_bf16: [local_experts, H, I].
fc1, fc1_scale = quantize_mxfp8(fc1_bf16)
fc2, fc2_scale = quantize_mxfp8(fc2_bf16)
moe = MegaMoE(config, communicator, fc1, fc1_scale, fc2, fc2_scale)
output = moe(inputs_bf16, routing_int32, router_weights_float32)
```

`H` and post-SwiGLU `I` must be positive multiples of 128. Experts divide evenly
across the world and use contiguous rank ownership. `1 <= top_k <= min(32, E)`.
Canonical FP8 E4M3FN (or `weight_e5m2=True` E5M2) weights are row-major, with uint8
E8M0 scales for every consecutive 32 K elements. Scales have shape
`[local_experts, M, K//32]`, **not** a backend-specific scale swizzle. Weights are
packed once during collective construction into context-owned buffers.

Inputs are BF16 `[T,H]`, routing IDs int32 `[T,top_k]`, router weights float32
`[T,top_k]`, output BF16 `[T,H]`. All tensors must be contiguous on the owning GPU;
autograd is unsupported. Routing IDs must be in `[0,E)` and weights finite.
`validate_routing=True` checks values with a host synchronization outside capture;
shape/dtype/device validation always occurs. Router weights multiply the SwiGLU
result before its BF16 conversion and FC2. FC2 partials are BF16 and the final
combination sums in FP32 before BF16 output, so it need not be bit-identical to a
reference that applies routing weights after FC2.

`sm_margin` reserves SMs: on a 152-SM GB200, margin 32 permits **at most 120 CTAs**.
The actual `moe.cta_count` also reflects two-CTA cluster alignment and occupancy.

### Local shared expert

For a context with `world_size=1`, `num_experts=1`, and `top_k=1`, use:

```python
shared_output = shared.forward_shared(inputs_bf16, output=preallocated_output, stream=shared_stream)
```

This unweighted SwiGLU path does not accept routing IDs or scores. It reads the
registered input directly and writes the final BF16 output directly, avoiding
token dispatch, peer synchronization, partial-output writes, and top-k
combination. It retains FP32 FC1 accumulators through activation and the BF16
FC1-to-FC2 handoff. The ordinary `forward` also uses the local kernel for this
configuration, while retaining its routing weights and masking behavior.

The local kernel uses 384 threads per CTA, a 128-token tile, and four stages for
weight, activation, and transformed-weight pipelines. Registered input capacity
is padded to 64 tokens; logical input/output sizes remain unchanged. FC1
publishes the handoff with a global async-proxy fence and device-scoped
release/acquire readiness. Local FC2 stores wait for source-buffer reuse and
are fully drained before kernel exit. There are no full-grid metadata barriers.

The distributed routed kernel remains separate: 512 threads, a 32-token tile,
eight weight/activation stages, and seven transformed-weight stages.

## Direct writes, streams, and CUDA Graphs

```python
workspace_input = moe.input_view(T)       # zero-copy; owns native storage
# squash_kernel(..., out=workspace_input, stream=stream)
moe(workspace_input, ids, scores, output=out, stream=stream)
```

`input_view` supports a squash or projection writing directly to registered
storage. An exact workspace-input alias skips staging; ordinary inputs are staged
by default. Output must not overlap the registered workspace. This is a hook for
layering shared experts, squash/unsquash, and residual computation; those are not
part of the routed kernel or its timings.

Construction, memory registration, peer exchange, and weight packing occur outside
capture. `forward(..., output=preallocated, stream=stream)` is capture-safe and
uses GPU-resident epochs, including on graph replay. Every rank must invoke the
same collective sequence, though token counts may differ or be zero.

A context is **not concurrently reusable**. Order forwards/replays on one stream,
or establish explicit event dependencies when switching streams. Tensor producers
must be ordered before the provided launch stream. Keep context, graph, and graph
inputs/outputs alive across all replays. Views retain the native context through
DLPack storage, including sliced views. The native destructor synchronizes local
CUDA execution before releasing imports; applications must additionally ensure
all ranks have finished peer accesses before releasing contexts (synchronize each
device, then use a bootstrap barrier). Destruction itself is not collective.

## Benchmark

Install a compatible Torch CUDA wheel separately, then launch one rank per GPU:

```bash
torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29500 \
  --nproc-per-node=4 -m mscclpp.ext.megamoe.benchmark \
  --tokens 32 --hidden 4096 --intermediate 4352 --experts 64 --top-k 7 \
  --sm-margin 32 --check --input-mode staged
```

For EP32 use a 32-GPU NVLink fabric, a multi-node torchrun rendezvous, and
`--experts 512`. Torch distributed **Gloo** is used only for setup, untimed
reference validation, and result reporting; timed communication is MSCCL++.
The MSCCL++ TCP bootstrap defaults to `MASTER_PORT+1` and can be changed with
`--bootstrap-port`.

`--input-mode direct` initializes the registered view before timing (it does not
benchmark a squash kernel). `--no-graph` times ordinary launches; graph mode
captures `--graph-batch` collectives per replay. JSON reports per-rank measurements,
maximum-across-ranks latency, workspace sizes, and actual CTA count. Results are
explicitly **routed-only**, excluding shared experts, squash/unsquash, router, and
residuals; they must not be presented as identical to full-model timing.

## Synthetic end-to-end layer with a native shared expert

`mscclpp.ext.megamoe.benchmark_shared` measures the routed branch plus one
**native MSCCL++ shared expert**, including a synthetic router and dense
squash/unsquash projections, followed by **post-RMSNorm and residual addition
by default**. It uses no FlashInfer or NVSHMEM imports or
dependencies. This is a synthetic MoE-layer benchmark, **not full SGLang model
parity**: randomly initialized matrices stand in for learned parameters.
Attention, prenorm, residual gathering, dropout, stochastic rounding, other model
layers, and real model weights are excluded.

Default geometry:

| Component | Shape / arithmetic |
| --- | --- |
| Original input | BF16 `[32, 8704]` per rank |
| Synthetic router | Original input converted to FP32; FP32 linear `8704 -> E`, FP32 logits, top-7, selected-logit softmax; int32 expert IDs |
| Squash | BF16 dense linear `8704 -> 4096` |
| Routed experts | BF16 activations, MXFP8 E4M3FN/E8M0 weights; `H=4096`, post-SwiGLU `I=4352`, top-7 |
| Routed ownership | 16 experts/rank; global `E=64` at EP4 or `E=512` at EP32 |
| Shared expert | **Original unsquashed input**, BF16 activations, MXFP8 E4M3FN/E8M0 weights; `H=8704`, post-SwiGLU `I=2048`, `E=1`, top-1, weight 1 |
| Unsquash | BF16 dense linear `4096 -> 8704` |
| Branch sum | BF16 addition of unsquashed routed output and shared output |
| Post-RMSNorm | FP32 arithmetic and FP32 gamma, epsilon `1e-6`, then BF16 output |
| Residual / final output | Separate FP32 residual added **after** postnorm; FP32 output by default |

The routed context uses the actual global MSCCL++ communicator, registered
workspace, and peer mappings. Each process constructs the shared context using
an **independent** `TcpBootstrap.create(0, 1)`, initialized with
`TcpBootstrap.create_unique_id()`, and a separate communicator/workspace. Merely
setting `world_size=1` on the global communicator would not implement this.

### Schedules and stream ordering

All four schedules run on identical inputs and weights, with persistent output
and router metadata buffers:

* **`routed-only`**: router, squash, native routed experts, unsquash, postnorm,
  residual add: the complete synthetic MoE path with shared experts disabled.
* **`shared-only`**: native shared expert on the original input; this component
  timer does **not** include global layer postnorm or residual addition.
* **`serial`**: router, squash, routed experts, shared expert, unsquash, BF16 add,
  postnorm, residual add, all ordered on the main stream.
* **`overlap`**: router and squash on the main stream; launch routed experts
  **first**; release the shared stream only after the routed kernel enters;
  unsquash follows routed completion; the main stream joins shared completion
  before the BF16 branch add, postnorm, and residual add.

The overlap schedule calls `routed.forward(..., signal_start=True)`, then
`routed.wait_until_started(shared_stream)`. The latter waits on the routed
reset event and a stream-ordered flag-value wait. The routed kernel publishes
its flag at entry; this is **not a routed-completion event**, a host launch-order
assumption, or a GPU polling kernel. Input-producer dependencies and the final
fork/join are explicit and included in CUDA Graph capture/replay. The start
event/flag and shared completion event have stable lifetimes. Neither context is
reused concurrently across layer invocations.
Graph objects are explicitly released and garbage-collected before context
teardown. After device synchronization and an all-rank bootstrap barrier, native
contexts are destroyed while their bootstraps remain alive; communicator and
bootstrap destruction also finishes before interpreter shutdown.

By default the routed margin is 32 and the shared capacity is 32:
on a 152-SM GB200 these request **120 routed CTAs and 32 shared CTAs**.
The shared context's margin is `physical_sms - shared_sms`.
These are **soft persistent CTA/SM capacity caps**, not fixed SM-ID partitions,
exclusive SM ownership, or a promise that every shared invocation overlaps.
The benchmark raises an error if native occupancy selects a different CTA count.
Requested counts must be even and leave at least two CTAs.

Both native inputs are staged **inside every invocation**, including graph
replays: squash writes an ordinary BF16 tensor, not a prefilled registered view;
the shared branch stages the original input after the routed-entry gate and
uses `forward_shared`, without staging top-1 routing metadata.
Static inputs and fixed top-1 shared metadata are allocated before measurement,
but no per-invocation staging, router, or projection work is moved outside timing.
`--graph-batch` defaults to five repeated invocations on the same input, with
router logits, IDs, scores, projections, and native staging recomputed each time.

### Postnorm and residual semantics

The order is **branch sum -> post-RMSNorm -> residual add**, not normalization
of the branch-plus-residual sum. Postnorm uses a deterministic, nonconstant
FP32 gamma (`linspace(0.75, 1.25, original_hidden)`) as a synthetic learned-weight
stand-in. Its arithmetic is `x * rsqrt(mean(x.float()**2) + eps) * gamma` in
FP32, explicitly rounded to BF16 **before** residual addition.

This reference implementation uses **Torch**. The timed path copies the BF16
branch result into a preallocated FP32 input buffer, calls
`torch.nn.functional.rms_norm` with **both input and weight in FP32**, then
copies the FP32 result into the persistent BF16 normalized buffer. It does not
silently round gamma to BF16 or use the mixed BF16-input/FP32-weight call that
can select Torch's multi-kernel fallback. The FP32 RMSNorm return tensor is an
internal per-invocation allocation, retained in the CUDA Graph pool on capture;
input, gamma, normalized, residual, and final-output addresses remain stable.

The default residual is a **separate preallocated FP32 copy of the initial BF16
input**, not an alias of the input, branch sum, or output. This is explicitly a
synthetic shortcut: a real model's skip may originate before prenorm and must be
provided separately for model parity. Input and residual updates when switching
samples are untimed producer work, just like initial input setup. Each invocation
reads the incoming stored skip; it does not copy new input into the skip or feed
the previous result back into either input buffer. Residual gather, prenorm,
dropout, and stochastic rounding are not performed.

Controls:

* `--post-norm` / `--no-post-norm`, default on.
* `--residual` / `--no-residual`, default on.
* `--rms-eps`, finite and positive, default `1e-6`.
* `--residual-dtype {fp32,bf16}`, default FP32. The BF16 option explicitly rounds
  the stored skip and final result to BF16. Without residual addition, the final
  output is BF16 regardless of this setting.

Use **`--no-post-norm --no-residual`** to reproduce the earlier mathematical
branch-only scope. A BF16 pass-through copy remains when postnorm is disabled;
JSON calls it a bypass copy rather than claiming RMSNorm was measured. Historical
timings below predate that copy and the new default postprocess.

The reusable helper supports both whole-layer and separate-stage timing:

```python
from mscclpp.ext.megamoe.benchmark_shared import _Postprocess

post = _Postprocess(
    incoming_residual, epsilon=1e-6, post_norm=True,
    add_residual=True, residual_dtype="fp32",
)
output = post.forward(branch_sum_bf16)
# Equivalent staged form:
normalized_bf16 = post.normalize(branch_sum_bf16)
output_fp32 = post.add_residual()
reference_cpu = post.reference(branch_sum_bf16, residual=incoming_residual)

# Before replaying with another sample, on the input producer stream:
post.residual.copy_(next_incoming_residual)
```

`.residual` owns the converted copy; `.weight`, `.epsilon`, `.normalized`, and
`.output` expose stable helper state. The caller must order input/residual
producers before the execution stream, keep the helper alive across captures,
and use separate helpers or explicit ordering for concurrent invocations.

### EP4 first, then EP32

Set explicit Gloo rendezvous variables. Gloo handles setup, untimed routed
reference collection, barriers outside timing, and result reporting only.
Timed cross-rank communication remains native MSCCL++.

```bash
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --tokens 32 --original-hidden 8704 --hidden 4096 --intermediate 4352 \
  --experts 64 --top-k 7 --shared-intermediate 2048 \
  --route-sm-margin 32 --shared-sms 32 --graph-batch 5 \
  --warmup 5 --iterations 30 --check --bootstrap-port 29501 \
  --json-output results/megamoe-shared-ep4.json
```

EP32 must use distinct GPUs in a single active NVLink fabric. For example,
on **each of eight four-GPU hosts**, set the same reachable rank-0 address and
set `NODE_RANK` to that host's index in `[0,7]`, then run:

```bash
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
# Set NODE_RANK=0, 1, ..., 7 separately on the corresponding host.
torchrun --nnodes=8 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --tokens 32 --original-hidden 8704 --hidden 4096 --intermediate 4352 \
  --experts 512 --top-k 7 --shared-intermediate 2048 \
  --route-sm-margin 32 --shared-sms 32 --graph-batch 5 \
  --warmup 5 --iterations 30 --check --bootstrap-port 29501 \
  --json-output results/megamoe-shared-ep32.json
```

Adapt host/GPU counts to the fabric while retaining 32 total ranks.
Omitting `--experts` selects `16 * WORLD_SIZE`.
All ranks emit UTC phase logs for initialization, correctness checks, graph
capture, timing, and tracing. Use an external bounded distributed process runner
to terminate **all ranks** if a peer fails.

### Correctness and interpreting results

`--check` checks two inputs (the second is negated to change routing) and two
separately updated residuals, before
timing. The independent untimed CPU Torch routed oracle uses the actual global
Gloo group. The shared CPU oracle is strictly local: dequantized weights, FP32 FC1 and
SwiGLU, BF16 activation handoff, FP32 FC2, then BF16 output. It never performs a
global gather using the shared context's world size. The combined oracle also
includes unsquash and BF16 branch addition. Independent CPU postprocess oracles
then compute explicit FP32 squared-mean/rsqrt/gamma, the BF16 handoff, and residual
addition. They do not call the Torch RMSNorm implementation under test.
Canonical weights and the unsquash matrix are copied to host memory only for
checking; budget approximately another 1 GB of host memory per rank at the
default geometry. CPU checking is intentionally outside graph capture/timing.

Checks report relative L2/max/mean absolute errors for both branches and the
combined, normalized, and final outputs (`--reference-relative-l2`, default 0.05), require finite,
nonzero shared contributions for nonempty inputs, and require **bitwise
equivalence** of serial/overlap outputs in eager execution and batched graph
replay for both inputs and residuals. `--tokens 0` exercises empty invocations using native
capacity one; the default and representative workload remains 32 tokens/rank.

Latency is CUDA-event time around the complete graph replay, divided by
`graph_batch`; the overlap interval includes the start gate and final join.
JSON contains raw per-rank samples and medians, a maximum-across-ranks sample for
each iteration, summary statistics, dimensions/dtypes, actual CTA counts,
workspace sizes, initialization times, and correctness results. The primary
speedup is **same-cap native serial / routed-first overlap**, including enabled
postnorm/residual work. JSON explicitly identifies enabled operations, the
synthetic skip source, FP32 gamma/arithmetic, BF16 normalization handoff, and
selected residual/final dtype. There is no
full-resource serial baseline in this benchmark. This does not establish
performance against a previous production shared kernel: earlier shared-first
or router-free experiments measure different work. No routed-weight bandwidth
number is presented as whole-model HBM bandwidth.

### Trace actual routed/shared overlap

Add `--trace-path` on **every rank**; all ranks enable the Torch CUDA profiler
and only rank zero exports JSON. For EP4:

```bash
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m mscclpp.ext.megamoe.benchmark_shared \
  --experts 64 --check --graph-batch 5 --warmup 5 --iterations 30 \
  --bootstrap-port 29501 --trace-path results/megamoe-shared-ep4-trace.json \
  --json-output results/megamoe-shared-ep4-traced.json
```

For EP32, add the same trace option to every host's EP32 command.
Tracing happens **after unprofiled performance measurement**, and records three
single-layer overlap graph replays. Native kernels share the `megaMoe` name;
their CUDA stream IDs and grids (120 versus 32 CTAs on default GB200 geometry)
identify the branches. CPU postprocessing checks actual GPU start order and
distinct streams, and reports each pair's durations, overlap microseconds, total
paired span, and observed kernel/pair counts. Zero actual overlap is reported as
zero, not inferred from separate streams. Trace attribution requires distinct
routed/shared CTA counts.

Trace samples, particularly the first, can have profiler-induced outliers and
are **not steady-state performance measurements**. For host stage labels
(`router`, `squash`, `routed`, `shared`, `unsquash`, `combine`, `postnorm`,
`residual`), also pass
`--trace-eager`; this profiles three eager overlap invocations instead of graph
replays and can further perturb overlap. There is no fallback from failed
overlap or graph capture to an unreported serial schedule.

### Tests

Run the CPU argument/configuration, projection/router, local and postprocess
oracles, FP32 precision/ordering, buffer lifetime, trace
parser, and result-reduction tests without the MPI-dependent project conftest:

```bash
python -m pytest --noconftest python/test/test_megamoe_shared.py -q
```

The small `H=128`, `I=128`, single-GPU native schedule test is disabled unless
explicitly enabled. It uses two independent one-rank bootstraps, validates the
shared oracle and eager/graph schedule equivalence, changes graph inputs and
residuals (including a residual-only update), and tests zero-token graph replay:

```bash
MSCCLPP_TEST_MEGAMOE_SHARED=1 \
  python -m pytest --noconftest python/test/test_megamoe_shared.py \
  -q -k native_shared_schedule
```

This local test does not replace EP4/EP32 live-fabric correctness and tracing.

### Experimental routed-first branch-only results

The 2026-09-12 experimental build (best routed candidate plus the start-gate API
and corrected named-barrier IDs) measured the following CUDA Graph latencies
**before postnorm and residual addition were included**. These are not timings
of the new default complete-layer scope.
Both schedules use 120 routed CTAs and 32 shared CTAs per GB200. EP32 values
are medians of three independent run medians; each run uses 30 samples of
10 branch-only layer invocations.

| Schedule | EP4 | EP32 |
| --- | ---: | ---: |
| Routed branch, including router and projections | 309.40 us | 328.82 us |
| Shared branch alone, 32 CTAs | 78.01 us | 79.60 us |
| Same-cap serial, without postnorm/residual | 388.97 us | 410.10 us |
| Routed-first overlap, without postnorm/residual | 326.01 us | 345.34 us |

EP32 overlap reduced latency by 64.76 us (15.79%) versus same-cap serial.
Rank-0 CUDA activity traces confirmed routed-first execution on separate streams:
after excluding the profiler-startup outlier, the shared kernel started
11.71-13.34 us after the routed kernel, and its approximately 98.3 us execution
was entirely within the routed kernel's interval. This describes the shared
**kernel**, not every shared-branch staging operation or scheduling cost.

All 32 ranks completed changed-input CPU-oracle checks and bitwise
serial/overlap equivalence in eager and graph execution. The maximum combined
relative L2 error against the CPU oracle was 0.002216.

These results include the synthetic **FP32 router**, unlike the earlier
189.71/194.68 us routed-expert-only measurements. They were collected with a
separate experimental build before source integration. The optimized native
path is now integrated in the working tree; these tables remain historical
measurements and exclude the new default postnorm/residual work.

### Controlled routed-backend comparison

A separate 2026-09-12 comparison held the shared expert and all surrounding work
fixed, replacing only the routed backend. Both backends used the **same compiled
CuTe DSL lean FC12 shared kernel**, the same BF16 `torch.mm` squash/unsquash
matrices and buffers, and the same 20 inputs, routing metadata, and canonical
expert weights. Router/top-k generation, residual,
and postnorm were excluded, matching the earlier FlashInfer overlap benchmark.

| Squash-then-fork schedule | EP4 | EP32 |
| --- | ---: | ---: |
| FlashInfer routed + common shared | 253.15 us | 271.93 us |
| Native routed + common shared | 225.18 us | 231.67 us |

Each value is a median of three trials, with 20 layer invocations per graph and
30 replays per trial. Routed/shared grids remained 120/32 CTAs. The EP32 native
backend saved 40.26 us (14.81%). All 640 checked EP32 outputs were bitwise equal
between routed backends, and traces verified the identical shared kernel.

This comparison intentionally reproduces the previous schedule: after squash,
submit shared work on the second stream, then routed work on the main stream.
It is **not strict routed-first execution**. Keeping the same shared kernel but
using the native entry gate instead measured 233.77 us on EP4 and 239.32 us on
EP32; these are separate scheduling results.

FlashInfer is a dependency of that external comparison harness, **not** the
native MSCCL++ library or `benchmark_shared.py`. These measurements do not claim
that the native E1 shared implementation has the lean kernel's performance.

### Source checkpoint and performance

The current working tree **integrates the optimized native mainloop**, compact
epilogue, role-local register budgets, MemoryChannel signaling, corrected named
barriers, and graph-compatible start gate. It uses a bundled MMA helper and
unmodified pinned CUTLASS, without external header overrides or a FlashInfer
dependency.

Source-build validation passed 94 core and benchmark tests, including E4M3/E5M2
weights, ragged pipeline cases, graph replay, post-RMSNorm, residual precision,
and changed inputs. The wheel includes the adapted MMA helper's license notice
under `mscclpp/licenses/mscclpp_megamoe/`. Three EP4 routed-only runs measured
**189.93, 190.06, and 191.03 us**, consistent with the earlier experimental
189.71 us result. The earlier EP32 194.68 us artifact measurement remains a
historical reference, not a newly measured source-build result. None of these
routed-only timings includes shared experts, router, projections, postnorm, or
residual addition.

The integrated source was also exercised on EP32 with all default postprocessing
enabled (`epsilon=1e-6`, FP32 norm weights, BF16 normalized output, FP32 residual
and final output):

| Complete synthetic layer configuration | EP32 latency |
| --- | ---: |
| `benchmark_shared.py`: native routed + native E1 shared, same-cap serial | 431.34 us |
| `benchmark_shared.py`: native routed + native E1 shared, strict routed-first overlap | 365.55 us |
| External controlled comparison: FlashInfer routed + common CuTe shared, squash-then-fork | 384.21 us |
| External controlled comparison: native routed + common CuTe shared, squash-then-fork | 340.83 us |

These full-path measurements include the FP32 router, projections, staging,
expert branches, combination, post-RMSNorm, and residual addition. They start
from an already supplied BF16 activation and separate local FP32 skip tensor;
prenorm, attention, residual-gather communication, training operations, and
checkpoint loading are outside the boundary. The native-E1 and common-CuTe
shared configurations are deliberately listed separately, not treated as
interchangeable implementations.

All EP32 ranks completed the source-build numerical and graph checks. In the
controlled comparison, all 640 sampled final outputs were bitwise identical
between routed backends. These integration results are distinct from the
historical router/postprocessing-free latency tables above.

The routed-only `benchmark` reports routed-layer latency, including staging,
dispatch, both expert GEMMs, SwiGLU, return, and combination. Effective bandwidth
is the local FP8 weight and E8M0 scale byte count divided by that latency, not a
hardware HBM-counter measurement.

Single-rank kernel-replay profiling cannot validate the collective protocol:
correctness must also be checked with live multi-rank execution using actual
MSCCL++ buffer registration and peer imports, including CUDA Graph replay.
