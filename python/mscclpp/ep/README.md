# Expert-parallel PyTorch interface

`mscclpp.ep` is an optional PyTorch tensor interface backed by the native
`mscclpp.mscclpp_ep_cpp` runtime. Install `mscclpp[ep,cuda12]` or
`mscclpp[ep,cuda13]` for the appropriate CUDA version. Source builds must
use CUDA and enable the Python and EP targets
(`MSCCLPP_USE_CUDA=ON`, `MSCCLPP_BUILD_PYTHON_BINDINGS=ON`, and
`MSCCLPP_BUILD_EXT_EP=ON`). Importing the root `mscclpp` package does not
eagerly import this interface.

## Minimal latency example

Save this as `ep_identity.py` and run `mpirun -np 8 python ep_identity.py` on one
host with eight CUDA GPUs visible to each process. `mpi4py` is needed for this
example; `CommGroup(torch_group=...)` also accepts an initialized PyTorch
distributed process group.

```python
from mpi4py import MPI
import torch

from mscclpp import CommGroup
from mscclpp.ep import MoECommunicator, MoECommunicatorConfig, MoEMode


def main():
    world = MPI.COMM_WORLD
    local = world.Split_type(MPI.COMM_TYPE_SHARED)
    torch.cuda.set_device(local.Get_rank())
    device = torch.device("cuda", local.Get_rank())
    group = CommGroup(mpi_comm=world)

    config = MoECommunicatorConfig(
        comm=group,
        device=device,
        mode=MoEMode.LATENCY,  # EXPERT_MAJOR by default
        num_experts=world.Get_size(),  # one expert per rank
        hidden_size=4096,
        topk=1,
        max_tokens_per_rank=64,
    )
    moe = MoECommunicator(config)
    if not moe.is_available():
        raise RuntimeError("EP is unavailable for this configuration")
    moe.initialize()  # collective; all ranks must call in the same order
    stream = torch.cuda.current_stream(device)

    with torch.cuda.stream(stream):
        x = torch.randn((32, 4096), dtype=torch.bfloat16, device=device)
        ids = torch.full(
            (32, 1),
            (world.Get_rank() + 1) % world.Get_size(),
            dtype=torch.int64,
            device=device,
        )
        weights = torch.ones((32, 1), dtype=torch.float32, device=device)
        received, handle = moe.dispatch(x, ids, weights, stream=stream)

        # Identity expert, topk=1, weight=1. The tensor is capacity-sized;
        # native combine ignores rows outside its private routing metadata.
        result = moe.combine(received.tokens, handle, stream=stream)

    # Caller-owned completion, only for checking the example and safe teardown.
    stream.synchronize()
    group.barrier()  # peers also finish before registered buffers are released
    torch.testing.assert_close(result, x, rtol=0, atol=0)


if __name__ == "__main__":
    main()
```

Keep communicators, runtimes, handles, and results function-local so they are
released after GPU completion and the peer barrier, **before Python interpreter
shutdown**. Do not rely on module-global destruction order for communication
resources. The wrapper itself does not synchronize on destruction.

All ranks must configure matching dimensions, layout, capacity, and algorithms,
and invoke collectives in the same order. Initialization is idempotent and is
otherwise performed lazily by `prepare`, `dispatch`, or
`get_dispatch_output_buffer`.
`is_available()` reports basic native availability for the resolved topology
and capacity; check it consistently across ranks before explicit initialization.
Per-launch validation still applies.
For supported CUDA graph capture, construction and initialization are caller
preconditions; the Python wrapper does not query capture state. An explicit
`device` is honored during native construction and initialization, even when
another CUDA device is current; the previous device is restored afterward.
Operations use the caller stream's device scope.

`MoECommunicator` takes a frozen `MoECommunicatorConfig`. To change settings,
create a new config with `dataclasses.replace(config, ...)`.

## Shapes, counts, and computation

Let `R` be the rank count, `A` the active per-rank capacity, `H` the hidden size,
`K` the top-k count, and `L = num_experts / R`. `A` defaults to the configured
`max_tokens_per_rank`. A call's `runtime_max_tokens_per_rank=A` may reduce it,
provided `num_input_tokens <= A <= configured_capacity` and all ranks agree.
Runtime-owned payload and communication storage is allocated at configured
capacity; returned runtime views and strides use **active** capacity. Calls do
not resize those buffers or read counts back to the host.

| Mode / layout | `tokens` shape | Count metadata | Physical row grouping | Expert-output contract |
| --- | --- | --- | --- | --- |
| LATENCY / EXPERT_MAJOR (default) | `[L, R*A, H]` | `layout.num_tokens_per_expert` | One fixed slice per local expert | BF16 per-expert results; native combine applies the original routing weights |
| LATENCY / RANK_MAJOR | `[R, A, H]` | `layout.num_tokens_per_rank` | One fixed slice per source rank | BF16 weighted local sums, or unweighted per-topk rows for DIRECT_SEND |
| THROUGHPUT / TOKEN_MAJOR (default) | `[R*A, H]` | Optional `layout.num_tokens_per_expert` (prepare `output_count[L]`) | Compact source-rank segments, preserving token order | BF16 already-weighted local expert sums |
| THROUGHPUT / RANK_MAJOR | `[R, A, H]` | Optional `layout.num_tokens_per_rank` (prepare `output_count[R]`) | One fixed slice per source rank | BF16 already-weighted local expert sums |

Counts, when present, are CUDA int32 tensors. Latency dispatch always returns
them through `DispatchLayoutInfo`. Throughput counts are written only to the
caller-provided `output_count` tensor passed to explicit preparation. Both
the preparation handle and dispatch metadata retain that tensor: dispatches
using the handle expose it as `layout.num_tokens_per_expert` for TOKEN_MAJOR or
`layout.num_tokens_per_rank` for RANK_MAJOR, without copying. The other field is
`None`. Count-free preparation and automatic dispatch leave both fields as
`None`. Reusing a preparation handle reuses the counts without rewriting them.

In TOKEN_MAJOR, preparation's per-expert `output_count` is an expert workload
statistic, not a description of physical row ranges. One token row can route
to multiple local experts, so it increments multiple expert counts while
occupying only one physical row. Consequently, expert counts cannot determine
the number of valid rows or their offsets.

TOKEN_MAJOR rows are grouped by source rank and then by source token order.
Describing those segments would require `num_tokens_per_rank`; their offsets
would be `exclusive_cumsum(num_tokens_per_rank)`. That metadata is not currently
exposed. The output remains capacity-sized, and native combine uses private
routing metadata to ignore unused tail rows. When RANK_MAJOR `output_count` is
requested, a GPU mask can be formed with
`torch.arange(A, device=device)[None, :] < output_count[:, None]`. Capacity
tails and padded metadata are unspecified.

Input `topk_ids` is contiguous CUDA int64 `[num_input_tokens, K]`, using global
expert IDs in `[0, num_experts)` or negative values for dropped routes.
`weights`, when provided, is contiguous CUDA FP32 of the same shape; omission
means one per valid route. Routing values are not read or validated on the host.

RANK_MAJOR and throughput outputs include CUDA int32 `topk_ids` and FP32
`weights` with shape `tokens.shape[:-1] + (K,)`. Latency IDs are global, with
`invalid_token_expert_id` (default `num_experts`) marking dropped or non-local
entries. Throughput IDs are local to the destination rank, with `-1` marking
invalid routes in valid rows.

Latency RANK_MAJOR must dispatch into the runtime-owned buffer. It also exposes
the required BF16 `combine_input_buffer`. With `CombineMode.DIRECT_SEND`, that
buffer and expert outputs have shape `[R, A, K, H]`: write unweighted BF16
per-topk results for entries whose returned `topk_ids` identify a local expert.
Combine applies the original routing weights during FP32 reduction. Otherwise
write already-weighted local expert sums. Before `combine`, place these results
in the runtime-owned buffer, either by computing there or by explicitly copying
them there. Latency rank-major combine rejects an external expert-output tensor
rather than performing a hidden staging copy. Both throughput layouts have the
same exact-buffer requirement: pass `combine_input_buffer`, after writing or
copying expert results into it. Throughput dispatch payload, top-k IDs, weights,
and FP8 scales are also runtime-owned views.

`combine` returns token outputs only and has no `output_topk_weights` argument.

THROUGHPUT supports only `CombineMode.RANK_LOCAL_REDUCE`.
For preweighted expert outputs, use `combine(..., apply_router_weights=False)`
with latency `DIRECT_SEND` (either layout). The default is `True` for
`DIRECT_SEND`, `False` otherwise. Other modes ignore this option.

## Formats and buffers

* BF16 dispatch uses BF16 inputs/outputs and no block scales.
* FP8 E4M3 uses one FP32 scale per 128 hidden elements. Latency EXPERT_MAJOR
  quantizes **BF16 input** on the fly with
  `QuantConfig(format=DispatchDataType.FP8_E4M3)` and does not accept
  precomputed input scales. Returned FP32 scales have logical shape
  `[L, R*A, H//128]`, transposed from physical `[L, H//128, R*A]`; they are not
  contiguous in logical order.
* Throughput FP8 requires `torch.float8_e4m3fn` input and
  `QuantConfig(format=DispatchDataType.FP8_E4M3, block_scales=scales)`, where
  `scales` is contiguous CUDA FP32 `[num_input_tokens, H//128]`. Output scales
  have shape `tokens.shape[:-1] + (H//128,)` and are contiguous.
* Latency RANK_MAJOR is BF16-only. **Combine always consumes BF16**, including
  after FP8 dispatch.

Configure a default `quant` on the communicator or override it per dispatch.
Explicit `QuantConfig(format=DispatchDataType.BF16)` overrides an FP8 default.
There are no implicit casts or contiguous copies to accept incompatible input.
Payload pointers must be 16-byte aligned. `output_buffer` must match the active
dispatch shape, while `combine(out=...)` must be `[num_input_tokens, H]`; both
require the exact dtype, device, and contiguous layout. Only latency
EXPERT_MAJOR permits an external dispatch `output_buffer`. Other layouts require
the exact runtime-owned dispatch buffer.
`get_dispatch_output_buffer(quant=..., runtime_max_tokens_per_rank=...)` returns
a correctly shaped runtime view without moving data.

**Buffer aliasing is the caller's responsibility and is not checked.**
Dispatch payload, routing, and scale inputs must not overlap its output or
runtime receive storage written by the operation. Combine outputs must be
disjoint from expert inputs and runtime combine storage. Latency RANK_MAJOR and
both throughput layouts require the exact runtime `combine_input_buffer`.
These restrictions apply on a single stream too: arbitrary in-place operations
and partially overlapping copies are unsupported.

Throughput's runtime dispatch and BF16 combine views use **separate storage**.
Expert computation can read BF16 or FP8 dispatch data while writing BF16 results
into `combine_input_buffer` without overwriting the dispatched payload.
Runtime views are reused by later operations, not independent results.

## Streams, handles, and limitations

* The first successful preparation or dispatch binds the runtime to the supplied
  stream (`None` means the device's current PyTorch stream). Dispatch, expert
  computation, and combine must then use that same stream. The wrapper creates
  no internal CUDA stream or event, inserts no cross-stream dependencies, and
  performs no host count readback. Host calls sharing a communicator must be
  serialized.
* `prepare` is throughput-only. Pass an `output_count` CUDA int32 tensor only
  when per-expert or per-rank counts are needed. Omit `prepare_handle` for
  count-free automatic GPU preparation, or reuse a `PrepareHandle` for unchanged
  IDs, pointer, token count, and active capacity. All ranks must choose the same
  path. A new preparation, including automatic preparation, invalidates earlier
  preparation and dispatch handles. Complete the matching combine before the
  next dispatch or preparation because runtime storage is reused. A dispatch
  that explicitly reuses a preparation leaves that `PrepareHandle` reusable.
  The Python wrapper checks handle type and runtime ownership, while the native
  runtime validates generations and exact preparation pointer, token count,
  capacity, and block count. Preparation mismatches raise an MSCCL++ error
  rather than a Python `ValueError`.
* Keep routing and borrowed metadata unchanged through matching combine work.
  Python handles retain the native runtime and borrowed tensors; tensor storage
  retains native owners even across slices and views, without owner cycles.
  PyTorch allocations are recorded on the caller stream to prevent premature
  allocator reuse. This is not cross-stream execution or peer synchronization.
  The caller must complete **all local and peer GPU use** before releasing the
  last runtime, handle, or view.
* CUDA graph replay supports throughput dispatch/combine with either preparation
  captured in the graph (automatically by dispatch or explicitly before dispatch)
  or a reusable `PrepareHandle` created before capture. Construct, initialize,
  and warm up outside capture; for the reusable-handle path, also prepare outside
  capture and keep routing unchanged.
  Captured preparation recomputes routing on each replay, so routing values may
  change while their buffer pointer, shape, and active capacity remain fixed.
  Preserve graph ordering, buffers, and owners through the last replay. Python
  handle checks run during capture, not replay. Latency dispatch/combine graph
  replay is unsupported because its host-side epoch is not advanced by replay.
* CUDA SM90+ and one IPC domain are required; HIP and inter-domain transports
  are unsupported. The implementation accepts 1-64 ranks, not a claim of
  hardware qualification of every 64-rank topology. Expert placement is even
  and contiguous; top-k is 1-8; throughput allows at most 128 experts per rank.
* Latency hidden sizes are `1024, 2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216`.
  Throughput combine always consumes BF16 and requires hidden size to be a
  multiple of 8 for 16-byte rows, regardless of the dispatch format. FP8 dispatch
  additionally requires a multiple of 128.
  Default dispatch/combine blocks are `(130, 128)` for latency or `(24, 32)` for
  throughput, clipped to SM count. Custom values must satisfy `D <= 130` and
  `1 <= C <= 128`; latency also requires `D >= R+2`, and latency RANK_MAJOR
  combine requires `C > 1`. Native shared-memory, residency, and occupancy
  checks still apply. `num_blocks=(D, C)` sets the two grid sizes explicitly. A
  scalar `N` sets the dispatch grid to `N`; latency dispatch reserves two control
  blocks, so combine uses the remaining `N-2` worker blocks, while throughput
  uses `N` for both operations.
* The API does not automatically overlap communication with expert computation
  or provide PyTorch autograd backward support.
