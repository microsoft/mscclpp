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
Graphs.

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

## Tests

From the repository root with the extension installed:

```bash
MSCCLPP_TEST_MEGAMOE_SHARED=1 python -m pytest --noconftest \
  python/test/test_megamoe.py python/test/test_megamoe_shared.py -q
```

GPU cases require SM100. Omitting `MSCCLPP_TEST_MEGAMOE_SHARED` skips the opt-in
shared/router GPU cases. Single-GPU tests do not replace multi-rank `--check`
and CUDA Graph validation on the target fabric.
