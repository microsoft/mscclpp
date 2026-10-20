<!-- Copyright (c) Microsoft Corporation. -->
<!-- Licensed under the MIT License. -->

# MegaMoE W4A8

MegaMoE is an SM100-family routed SwiGLU expert implementation with packed
MXFP4 weights, dynamically quantized E4M3 activations, BF16 inputs/outputs, and
CUDA IPC communication. Only routed W4A8 execution is supported.

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
distributions.
