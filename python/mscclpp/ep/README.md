# Expert-parallel PyTorch interface

`mscclpp.ep` is an optional, pure-Python tensor interface to the native
`mscclpp.mscclpp_ep_cpp` runtime. Install `mscclpp[ep,cuda12]` or
`mscclpp[ep,cuda13]` for the appropriate CUDA version and build
MSCCL++ with its EP Python extension. Importing the root `mscclpp` package does
not eagerly import this interface.

## Minimal throughput example

Save this as `ep_identity.py` and run `mpirun -np 8 python ep_identity.py` on one
host with eight CUDA GPUs visible to each process. `mpi4py` is needed for this
example; `CommGroup(torch_group=...)` also accepts an initialized PyTorch
distributed process group.

```python
from mpi4py import MPI
import torch

from mscclpp import CommGroup
from mscclpp.ep import MoECommunicator, MoEMode

def main():
    world = MPI.COMM_WORLD
    local = world.Split_type(MPI.COMM_TYPE_SHARED)
    torch.cuda.set_device(local.Get_rank())
    device = torch.device("cuda", local.Get_rank())
    group = CommGroup(mpi_comm=world)

    moe = MoECommunicator(
        comm=group,
        device=device,
        mode=MoEMode.THROUGHPUT,  # TOKEN_MAJOR by default
        num_experts=world.Get_size(),  # one expert per rank
        hidden_size=128,
        topk=1,
        max_tokens_per_rank=64,
    )
    moe.initialize()  # collective; call before CUDA graph capture
    stream = torch.cuda.current_stream(device)

    with torch.cuda.stream(stream):
        x = torch.randn((32, 128), dtype=torch.bfloat16, device=device)
        ids = torch.full(
            (32, 1), (world.Get_rank() + 1) % world.Get_size(),
            dtype=torch.int64, device=device,
        )
        weights = torch.ones((32, 1), dtype=torch.float32, device=device)
        routing = moe.prepare(ids, stream=stream)
        received, handle = moe.dispatch(
            x, ids, weights, prepare_handle=routing, stream=stream
        )

        # The receive count stays on the GPU; capacity tails are unspecified.
        valid = torch.arange(received.tokens.shape[0], device=device) < received.layout.num_recv_tokens
        # Identity expert, topk=1, weight=1: already weighted and locally aggregated.
        expert_output = torch.where(valid[:, None], received.tokens, 0)
        result = moe.combine(expert_output, handle, stream=stream)

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
`get_dispatch_output_buffer` through the shared initialization decorator.
Construction and initialization before graph capture are caller preconditions;
the Python wrapper does not query capture state. An explicit `device` is honored
during native construction and initialization, even when another CUDA device is
current; the previous device is restored afterward. Operations use the caller
stream's device scope. Passing a `MoECommunicatorConfig` instead of
constructor keywords is equivalent.

## Shapes, counts, and computation

Let `R` be the rank count, `A` the active per-rank capacity, `H` the hidden size,
`K` the top-k count, and `L = num_experts / R`. `A` defaults to the configured
`max_tokens_per_rank`. A call's `runtime_max_tokens_per_rank=A` may reduce it,
provided `num_input_tokens <= A <= configured_capacity` and all ranks agree.
Storage is allocated once at configured capacity; returned views and strides
use **active** capacity. No host sizing, receive-pool resizing, or compaction
kernel is introduced.

| Mode / layout | `tokens` shape | Valid rows | Expert-output contract |
| --- | --- | --- | --- |
| LATENCY / EXPERT_MAJOR (default) | `[L, R*A, H]` | `layout.num_tokens_per_expert` | BF16 per-expert results; native combine applies the original routing weights |
| LATENCY / RANK_MAJOR | `[R, A, H]` | `layout.num_tokens_per_rank` | BF16 already-weighted local expert sums |
| THROUGHPUT / TOKEN_MAJOR (default) | `[R*A, H]` | scalar `layout.num_recv_tokens` | BF16 already-weighted local expert sums |
| THROUGHPUT / RANK_MAJOR | `[R, A, H]` | `layout.num_tokens_per_rank` | BF16 already-weighted local expert sums |

Counts are CUDA int32 tensors. Throughput's `layout.num_recv_tokens` (also exposed
as `layout.num_tokens`) is a borrowed, read-only scalar containing the number
of distinct received token rows, excluding padding. It is overwritten by the
next preparation. **Do not modify this tensor:** PyTorch cannot enforce its
read-only contract. TOKEN_MAJOR per-expert counts count routes, not disjoint row
ranges; summing them is **not** a replacement for this scalar. `offsets` is not
populated. For RANK_MAJOR, a GPU mask can be formed with
`torch.arange(A, device=device)[None, :] < layout.num_tokens_per_rank[:, None]`.
Consumers must honor these counts; capacity tails and padded metadata are
unspecified.

Input `topk_ids` is contiguous CUDA int64 `[num_input_tokens, K]`, using global
expert IDs in `[0, num_experts)` or negative values for dropped routes.
`weights`, when provided, is contiguous CUDA FP32 of the same shape; omission
means one per valid route. Routing values are not read or validated on the host.

RANK_MAJOR and throughput outputs include CUDA int32 `topk_ids` and FP32
`weights` with shape `tokens.shape[:-1] + (K,)`. Latency IDs are global, with
`invalid_token_expert_id` (default `num_experts`) marking non-local entries.
Throughput IDs are local to the destination rank, with `-1` marking invalid
routes in valid rows.

Latency RANK_MAJOR must dispatch into the runtime-owned buffer. It also exposes
the required BF16 `combine_input_buffer`. With `CombineMode.DIRECT_SEND`, that
buffer and expert outputs have shape `[R, A, K, H]`: write already-weighted
per-topk results, with zero for absent routes. Otherwise write local expert
sums. Native combine stages external latency rank-major expert outputs into the
required buffer on the caller stream after validating the dispatch handle.
Throughput also accepts external expert outputs and lets native combine stage
them; neither path adds a second Python staging copy.
Its optional `combine(..., output_topk_weights=buffer)` writes combined weights
to a contiguous CUDA FP32 `[num_input_tokens, K]` buffer.

## Formats and buffers

* BF16 dispatch uses BF16 inputs/outputs and no block scales.
* Latency EXPERT_MAJOR can quantize **BF16 input** on the fly with
  `QuantConfig(format=DispatchDataType.FP8_E4M3)`. Returned FP32 scales have
  logical shape `[L, R*A, H//128]`, transposed from physical
  `[L, H//128, R*A]`; they are not contiguous in logical order.
* Throughput FP8 requires `torch.float8_e4m3fn` input and
  `QuantConfig(format=DispatchDataType.FP8_E4M3, block_scales=scales)`, where
  `scales` is contiguous CUDA FP32 `[num_input_tokens, H//128]`. Output scales
  have shape `tokens.shape[:-1] + (H//128,)` and are contiguous.
* Latency RANK_MAJOR is BF16-only. **Combine always consumes BF16**, including
  after FP8 dispatch.

Configure a default `quant` on the communicator or override it per dispatch.
Explicit `QuantConfig(format=DispatchDataType.BF16)` overrides an FP8 default.
There are no implicit casts or contiguous copies to accept incompatible input.
Payload pointers must be 16-byte aligned. `output_buffer` and `combine(out=...)`
must have the exact active shape, dtype, device, and contiguous layout.
`get_dispatch_output_buffer(quant=..., runtime_max_tokens_per_rank=...)` returns
a correctly shaped runtime view without moving data.

**Buffer aliasing is the caller's responsibility and is not checked.**
Dispatch payload, routing, and scale inputs must not overlap its output or
runtime receive storage written by the operation. Combine outputs must be
disjoint from each other, expert inputs, and runtime combine storage. Expert
results may use the exact runtime `combine_input_buffer`; otherwise they must
not overlap it. These restrictions apply on a single stream too: arbitrary
in-place operations and partially overlapping copies are unsupported.

Throughput's runtime dispatch and BF16 combine views **share physical storage**.
After FP8 dispatch, consume the FP8 data into separate expert-output storage
before writing BF16 into `combine_input_buffer`; never expand FP8 to BF16
in-place while reading it. Runtime views are reused, not independent results.

## Streams, handles, and limitations

* The first successful preparation, dispatch, or combine binds the runtime to
  the supplied stream (`None` means the device's current PyTorch stream).
  Subsequent operations on another stream are rejected. Use that same caller
  stream for expert computation; arrange any external producer dependencies
  yourself. The wrapper creates no internal stream, event, wait, or host count
  readback. Host calls sharing a communicator must be serialized.
* `prepare` is throughput-only. Omit `prepare_handle` for automatic GPU
  preparation, or reuse a `PrepareHandle` for unchanged IDs, pointer, token
  count, and active capacity. All ranks must choose the same path. A new
  preparation, including automatic preparation, invalidates earlier preparation
  and dispatch handles. Every successful dispatch invalidates earlier dispatch
  handles but can retain its explicitly reused preparation. Native code checks
  ownership and generations; invalid Python tensor metadata is rejected first.
* Keep routing and borrowed metadata unchanged through matching combine work.
  Python handles retain the native runtime and borrowed tensors; tensor storage
  retains native owners even across slices and views, without owner cycles.
  PyTorch allocations are recorded on the caller stream to prevent premature
  allocator reuse. This is not cross-stream execution or peer synchronization.
  The caller must complete **all local and peer GPU use** before releasing the
  last runtime, handle, or view.
* Construct and initialize outside CUDA graph capture. Preparation/dispatch/combine
  may be captured on the same stream. Handle checks occur while capturing, not on
  replay: preserve graph ordering, routing, buffers, and owners through the last
  replay. A graph reusing preparation without recomputing it needs unchanged
  routing.
* CUDA SM90+ and one IPC domain are required; HIP and inter-domain transports
  are unsupported. The implementation accepts 1-64 ranks, not a claim of
  hardware qualification of every 64-rank topology. Expert placement is even
  and contiguous; top-k is 1-8; throughput allows at most 128 experts per rank.
* Latency hidden sizes are `4096, 4352, 6656, 7168, 8192, 8704, 9216`.
  Throughput BF16 hidden size is a multiple of 8; FP8 requires a multiple of 128.
  Default dispatch/combine blocks are `(130, 128)` for latency or `(24, 32)` for
  throughput, clipped to SM count. Latency dispatch needs at least `R+2` blocks;
  native cooperative occupancy checks still apply. A scalar latency block count
  `N` means `(N, N-2)`; throughput means `(N, N)`.
* This port does not include the donor's notify/count caches, receive pools,
  overlap/event stubs, `previous_handle`, `enable_overlap`, or
  `expert_alignment`. There is no automatic autograd integration.
