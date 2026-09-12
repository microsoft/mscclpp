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

### Source checkpoint and performance

This source checkpoint contains the functional baseline, not the subsequent
performance experiments. A separately built native candidate measured 189.71 us
on EP4 and 194.68 us on EP32 with 32 SMs reserved per GPU, but those optimizations
have **not yet been integrated into this source**. These numbers must not be
reported as results of building this checkpoint.

The benchmark reports the complete routed-layer latency, including staging,
dispatch, both expert GEMMs, SwiGLU, return, and combination. Effective bandwidth
is the local FP8 weight and E8M0 scale byte count divided by that latency, not a
hardware HBM-counter measurement.

The current source baseline has **not reached FlashInfer performance parity**.
Single-rank kernel-replay profiling cannot validate the collective protocol:
correctness must also be checked with live multi-rank execution using actual
MSCCL++ buffer registration and peer imports, including CUDA Graph replay.
