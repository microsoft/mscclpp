<!-- Copyright (c) Microsoft Corporation. -->
<!-- Licensed under the MIT License. -->

# MegaMoE

MegaMoE is an SM100-family routed SwiGLU expert implementation supporting W4A8
and W8A16 execution with CUDA IPC communication. A world-size-one FP8-weight
context also supports communication-free shared-expert execution through
`forward_shared()`, with BF16 activations (W8A16) or static-scaled FP8
activations (W8A8, MXFP8 or `fp8_nv` weights).

## Shared expert

A shared-expert context uses `world_size=1`, `num_experts=1`, `top_k=1`,
`combine_route_weights=False`, and `weight_dtype="mxfp8"` (`float8_e4m3fn`
weights with `uint8` E8M0 K32 scales, same shapes as the routed MXFP8 layout).
`forward_shared(x, output=..., stream=...)` runs every token through the one
expert with route weight 1.0 and writes BF16 rows; there is no routing,
dispatch, or combine.

- `activation_dtype="bf16"` (W8A16): the W8A16 kernel dequantizes weights to
  BF16 in software before each BF16 MMA.
- `activation_dtype="fp8_static"` with `activation_scale=s` (W8A8): the input
  and the FC1 hidden are quantized to FP8 E4M3 with the per-tensor scale `s`,
  and both GEMMs run on the block-scaled (`kind::mxf8f6f4`) W4A8 pipeline with
  MXFP8 weights. The FC2 epilogue applies `s` before storing BF16. A 16-byte
  aligned input is quantized directly; other inputs are staged first. `forward()`
  is not supported for this configuration.
- `weight_dtype="fp8_nv"` (W8A8 only): the same `float8_e4m3fn` value tensors
  with one `float32` scale per weight matrix, `fc1_scale` and `fc2_scale` of
  shape `[1]` (dequantized weight = value * scale, e.g. scale = amax / 448).
  Packing writes unit E8M0 block scales for the same pipeline, and the epilogue
  multiplies FC1 by `s * fc1_scale` and FC2 by `s * fc2_scale`. The scales are
  read once at construction.

```python
shared = MegaMoE(
    MegaMoEConfig(rank=0, world_size=1, max_tokens=16, hidden=8704,
                  intermediate=2048, num_experts=1, top_k=1,
                  combine_route_weights=False, weight_dtype="mxfp8",
                  activation_dtype="fp8_static", activation_scale=1.0),
    local_communicator, fc1, fc1_scale, fc2, fc2_scale,
)
output = shared.forward_shared(x, output=out, stream=side_stream)
```

## Requirements

- NVIDIA SM100-family GPU
- CUDA toolkit and `nvcc` 13.0 or newer
- A compatible CUTLASS checkout
- One rank per GPU in the same active NVLink fabric

Build with `MSCCLPP_BUILD_EXT_MEGAMOE=ON`. The installed JIT source bundle
contains `megamoe_w4a8.cu`, `megamoe_launch.cu`, `megamoe_jit.cu`, and their
internal headers.

## Weights and configuration

`MegaMoEConfig` describes a routed W4A8 context. `hidden` must be at least 4096;
`hidden` and `intermediate` must be divisible by 128. Experts are partitioned
contiguously across ranks.

FC1 and FC2 values are packed `uint8` MXFP4 tensors:

- FC1: `[local_experts, 2 * intermediate, hidden // 2]`
- FC2: `[local_experts, hidden, intermediate // 2]`

The low nibble stores the even K element. E8M0 scales are canonical row-major
`uint8` tensors with shape `[local_experts, M, K // 32]`.

```python
from mscclpp.ext.megamoe import MegaMoEConfig, quantize_mxfp4

config = MegaMoEConfig(
    rank=rank,
    world_size=world_size,
    max_tokens=128,
    hidden=4096,
    intermediate=6656,
    num_experts=128,
    top_k=8,
)

fc1, fc1_scale = quantize_mxfp4(fc1_fp32)
fc2, fc2_scale = quantize_mxfp4(fc2_fp32)
```

Construction is collective and must happen outside CUDA Graph capture:

```python
from mscclpp.ext.megamoe import MegaMoE

moe = MegaMoE(config, communicator, fc1, fc1_scale, fc2, fc2_scale)
output = moe(inputs, topk_ids, topk_weights)
```

`inputs` and outputs are BF16. Routing IDs are int32; `-1` masks an unused slot.
Routing weights are float32.

## Kernel policy and JIT

The default `W4A8KernelConfig()` selects the builtin kernel and does not require
a compiler at runtime. A non-default policy is compiled once and cached by
content:

```python
from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

policy = W4A8KernelConfig(
    tile_n=128,
    load_stages=7,
    epilogue_warps=8,
    epilogue_registers=208,
    dispatch_chunk=4096,
    expert_major_tasks=True,
)
kernel = compile_kernel(policy)
moe = MegaMoE(config, communicator, fc1, fc1_scale, fc2, fc2_scale, kernel=kernel)
```

The compile-time token tile is fixed at N128. `tile_n` remains configuration
metadata but accepts only `128`; N32/N64 policies are no longer selectable.
Dynamic N is internal to the N128 kernel: activation TMA descriptors select
N32/N64 loads for smaller blocks, and MMA rows round up to a multiple of 16.
Other policy knobs, including load stages, remain configurable.

Set `MSCCLPP_MEGAMOE_CUTLASS_ROOT` to the compatible CUTLASS checkout.
`MSCCLPP_MEGAMOE_CACHE_DIR` optionally changes the content-addressed cache.
`fixed_token_count=True` is intended for captured decode where every rank uses
the same token-row count; route masks may still differ.

Weight values and their E8M0 scales support three L2 cache policies:

| Policy | Configuration | L2 priority |
| --- | --- | --- |
| Static streaming (default) | Both cache flags false | Evict-first for every expert |
| Static retention | `cache_weights=True` | Normal for every expert |
| Adaptive retention | `adaptive_cache_weights=True` | Normal when total expert rows exceed `tile_n`; otherwise evict-first |

`cache_weights` and `adaptive_cache_weights` are mutually exclusive. Adaptive
selection uses the expert's total routed rows from all ranks and slots for the
current launch, not the current block's tail length. It reuses the final routing
cursors, adds no workspace, and selects the hint once per task outside the K
loop. The last block of a multi-block expert therefore retains normal priority.
Activation cache hints, tile sizes, task order, and synchronization are unchanged.
This option is independent of `expert_major_tasks` and disabled by default;
compare policies on the same workload before enabling it.

```python
kernel = compile_kernel(W4A8KernelConfig(adaptive_cache_weights=True))
```

`fuse_static_quantize=True` (default) lets captured fixed-token static-FP8 launches
quantize their BF16 input inside the kernel instead of a separate launch when the
batch is small (H4096, 1-64 tokens per rank, `tokens * top_k <= ` the CTA thread
count, and enough CTAs). Routing CTAs `1..tokens` each quantize one token. With a
fused shared expert of a different hidden size, the MMA/load warps of every CTA
also quantize the shared input during the shared FC1 start delay, and the shared
FC1 loads wait for a grid-wide arrival counter. Set it to false to keep the
separate quantize launch.

Packaged exact-match policies can be loaded with
`resolve_w4a8_kernel_config`. Resident-grid results can be loaded with
`resolve_resident_tuning`.

JIT profile JSON is grouped by GPU model, for example `"GB300": {"entries": [...],
"provenance": {...}}`. Resolution automatically uses the current CUDA device name
and never falls back to another GPU's profiles. Offline callers can select a
section explicitly with `resolve_w4a8_kernel_config(config, device_name="GB300")`.
Profiles keep `"version": 1`; GPU-model sections are required instead of the
old flat structure.

The packaged GB300 EP32 profiles cover H4096/I6656, 32 local experts per GPU,
1024 global experts, top-8, and 32/64/128/256/512/1024 input tokens per rank.
These are the only six packaged JIT profiles; legacy entries have been removed.
All use N128 with `sm_margin=0`. Other knobs are tuned on balanced routing
separately for each token bucket, then frozen across balanced, Zipf,
rank-skewed Zipf, and hot-expert routing, without per-distribution retuning.
Load stages and other knobs may differ between token buckets.
Select the profiles with `--w4-profile --sm-margin 0` in the benchmark. They
do not change the builtin policy. Historical resident-grid tuning metadata is
separate from the selectable JIT policies.

## Benchmark

Launch the routed benchmark with `torchrun`:

```bash
torchrun --nproc-per-node=4 -m mscclpp.ext.megamoe.benchmark \
  --tokens 128 --hidden 4096 --intermediate 6656 --experts 128 --top-k 8 \
  --check
```

Use `--w4-profile` to select an exact packaged or external W4A8 policy, or
override configurable `W4A8KernelConfig` fields (such as
`--load-stages 7 --epilogue-warps 8 --epilogue-registers 208
--transfer-registers 64 --dispatch-chunk 4096 --adaptive-cache-weights`). Use
`--routing-pattern` (`balanced`, `zipf`, `zipf_skewed`, `hot_experts`, or
`lognormal` with `--lognormal-sigma`) to benchmark unbalanced expert routing
distributions. Use `--activation-dtype fp8_static` with optional
`--activation-scale` to run with per-tensor static-scaled FP8 E4M3 activations
and FP8 combine, `--weight-dtype mxfp8` (with `activation_dtype='bf16'`) to
run W8A16 (`float8_e4m3fn` weights + `uint8` E8M0 K32 scales and BF16
activations), or `--combine-route-weights` (`MegaMoEConfig(combine_route_weights=True)`)
to apply routing weights during the combine step rather than in FC1/FC2.
Use `--kernel-key KEY --cache-dir DIR` to load a prepared content-addressed
module without requiring CUTLASS or `nvcc` on the benchmark hosts.

Add `--overlap-shared-expert` to construct a separate world-size-one shared
expert context on every rank and overlap `forward_shared()` with routed MegaMoE
on a side stream. The default shared shape is the Run6.5 shape (H8704/I2048).
The shared expert is W8A16 by default; `--shared-activation-dtype fp8_static`
with `--shared-activation-scale S` (default 1.0) selects W8A8. W8A8 uses MXFP8
weights by default; `--shared-weight-dtype fp8_nv` uses `fp8_nv` weights with
one FP32 scale per matrix (amax / 448) instead.
`--shared-sm-count` sets its exact persistent CTA count. Both kernels are
cooperative persistent launches, so `--sm-margin` must leave at least
`--shared-sm-count` SMs free for them to be co-resident; the benchmark rejects
splits that would serialize unless `--allow-sm-oversubscription` is given
(diagnostic: each CTA fills an SM, so shared CTAs then only start on SMs the
routed kernel releases near its end). By default the shared branch waits for the routed
kernel to signal entry; with a co-resident split this wait is not needed, and
`--no-shared-wait-for-routed-start` avoids its extra graph nodes.
`--shared-weight-banks N` rotates N equal-valued shared weight copies so they
are not L2-resident across calls, as in a multi-layer model.
`--shared-join batch` joins the side stream once per graph batch instead of
before every routed call (diagnostic only).

The JSON reports the routed-only control at the same routed CTA count
(`routed_control_latency_us_max_across_ranks`) and the combined per-call
latency of an overlap graph without timing events
(`latency_us_max_across_ranks`). A second overlap graph with branch events
reports `overlap_timed_latency_us_max_across_ranks`, the routed and shared
branch latencies, the shared start delay and tail, and a shared-only control.
Event nodes add graph latency, so compare the untimed combined latency with the
routed-only control. `--trace-dir DIR` also profiles a few replays of each
graph, writes per-rank Chrome traces to DIR, and reports per-graph median
kernel durations (`kernel_trace_us_max_across_ranks`).

```bash
torchrun --nproc-per-node=4 -m mscclpp.ext.megamoe.benchmark \
  --tokens 16 --hidden 4096 --intermediate 5120 --experts 64 --top-k 7 \
  --sm-margin 16 --overlap-shared-expert --shared-sm-count 16 \
  --no-shared-wait-for-routed-start --shared-weight-banks 4 \
  --shared-hidden 8704 --shared-intermediate 2048 \
  --shared-activation-dtype fp8_static --shared-activation-scale 1.0
```
