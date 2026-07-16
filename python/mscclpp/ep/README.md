# Expert Parallel Python API Design

This document proposes a simplified Python API for MoE expert-parallel
dispatch and combine. It is a design draft for review, not a committed API
contract.

## Goals

The API should expose the tensors that the model naturally owns:

- token activations,
- top-k expert ids,
- routing weights,
- quantization scales.

It should adapt to different MoE runner backends, such as Triton ragged/grouped
GEMM, CUTLASS-style grouped GEMM, DeepGEMM, FlashInfer/CuteDSL, and custom MLP
kernels, without forcing one physical layout on every backend.

The dispatch output should make the local MLP contract explicit:

- the token layout returned by dispatch,
- per-local-expert valid token counts,
- optional expert offsets for flat layouts,
- optional quantization scale layout,
- optional overlap capability when the selected runner can notify combine.

## Public class name

Use `MoECommunicator` as the public class name:

```python
from mscclpp.ep import MoECommunicator

moe_comm = MoECommunicator(...)
```

The class owns MoE dispatch/combine communication, but it does not own the MLP
compute backend.

## MoECommunicator configuration

`MoECommunicator` owns communication setup, scratch buffers, expert placement,
layout choice, and optional overlap resources. These fields should be configured
once instead of being passed to every dispatch/combine call.

```python
@dataclass
class MoECommunicatorConfig:
    # Communication
    comm: Optional[mscclpp.CommGroup] = None
    device: Optional[torch.device | int] = None

    # Expert topology
    num_experts: int = 0
    num_local_experts: Optional[int] = None  # inferred for even contiguous placement
    local_expert_start: Optional[int] = None # inferred as rank * num_local_experts

    # Model shape and capacity
    hidden_size: int = 0
    topk: int = 0
    max_tokens_per_rank: int = 0
    max_recv_tokens_per_rank: Optional[int] = None

    # Runtime mode and output layout
    mode: MoEMode = MoEMode.LOW_LATENCY
    output_layout: Optional[DispatchLayout] = None  # default is derived from mode

    # Quantization defaults
    quant: Optional[QuantConfig] = None

    # Launch resources
    num_sms: int = 20

    # Overlap
    enable_overlap: bool = False
```

The constructor can accept either a config object or keyword arguments:

```python
moe_comm = MoECommunicator(
    comm=comm,
    num_experts=num_experts,
    num_local_experts=num_local_experts,
    hidden_size=hidden_size,
    topk=topk,
    max_tokens_per_rank=max_tokens,
    mode=MoEMode.HIGH_THROUGHPUT,
)
```

### Communication fields

`comm` is the MSCCL++ communication object used for rank information,
out-of-band metadata exchange, and connection setup.

The class should cache:

| Field | Purpose |
|---|---|
| `comm` | MSCCL++ communicator or `CommGroup` |
| `rank`, `world_size` | global EP rank information |
| `local_rank`, `device` | CUDA device binding |
| internal runtime | nanobind/C++ EP runtime implementation |

### Expert placement fields

The class needs enough information to map global expert ids to local expert
ids:

```python
local_expert_id = global_expert_id - local_expert_start
```

For the common contiguous placement:

```python
if num_local_experts is None:
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

if local_expert_start is None:
    local_expert_start = rank * num_local_experts
```

So both fields may be `None` for the common even contiguous placement. If the
placement is uneven or non-contiguous, the caller must provide enough placement
information explicitly. The first version can require contiguous local experts;
a later version can add an explicit `expert_map` for arbitrary placement.

### Runtime fields

`MoECommunicator` should keep these runtime decisions internally:

| Field | Purpose |
|---|---|
| `mode` | Backend selection (`MoEMode.LOW_LATENCY` or `MoEMode.HIGH_THROUGHPUT`) |
| `output_layout` | MLP input layout returned by dispatch |
| `max_tokens_per_rank` | dispatch capacity |
| `max_recv_tokens_per_rank` | recv buffer capacity |
| scratch buffers | internally sized from mode, capacity, topology, and shape |
| `num_sms` | backend launch/resource tuning |
| `dispatch_config`, `combine_config` | backend-specific tuning configs |
| `overlap_capability` | whether selected MLP/backend supports notify |

The user should not pass these to `dispatch` unless explicitly overriding a
specialized advanced path.

### Mode selection

The active implementation supports `mode=MoEMode.LOW_LATENCY` and
`mode=MoEMode.HIGH_THROUGHPUT`. `mode` must be a `MoEMode` enum value, not a
string. LL uses an expert-major output layout. HT uses a flat output layout and
supports 2, 4, 8, or 16 ranks within one detected GPU IPC/NVL fabric domain;
that domain may span multiple hosts.

```python
moe_comm = MoECommunicator(..., mode=MoEMode.LOW_LATENCY)
```

This keeps `MoECommunicator` policy-free. Serving frameworks such as SGLang can
choose a mode based on their own scheduling policy, batch shape, runner backend,
and benchmarking data once multiple active backends are available.

The selected mode determines the default dispatch output layout:

| Mode | Default layout |
|---|---|
| `ht` | `DispatchLayout.TOKEN_MAJOR` |
| `ll` | `DispatchLayout.EXPERT_MAJOR` |

`output_layout` may still be kept as an advanced override if a backend supports
multiple layouts within the same mode.

Use `DispatchLayout` instead of string literals for this field:

| Layout enum | Tensor shape |
|---|---|
| `DispatchLayout.TOKEN_MAJOR` | HT: `[total_recv_tokens, hidden]`; LL: `[world_size * max_tokens_per_rank, hidden]` |
| `DispatchLayout.EXPERT_MAJOR` | `[num_local_experts, max_slots_per_expert, hidden]` |

## MoECommunicator methods

```python
class MoECommunicator:
    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        quant: Optional[QuantConfig] = None,
        *,
        output_buffer: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[DispatchOutput, DispatchHandle]:
        ...

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        ...

    def dispatch_async(..., overlap_config: Optional[CommOverlapConfig] = None) -> DispatchRequest:
        ...

    def combine_async(..., overlap_config: Optional[CommOverlapConfig] = None) -> CombineRequest:
        ...

    def create_overlap_config(
        self,
        op: str,  # "dispatch" or "combine"
        *,
        handle: Optional[DispatchHandle] = None,
        level: str = "op",  # "op" or "block"
    ) -> CommOverlapConfig:
        ...
```

The blocking `dispatch` and `combine` methods should be the default path. The
`*_async` methods and `create_overlap_config` are optional advanced APIs for
communication/computation overlap.
If `stream` is not provided, both methods launch on `torch.cuda.current_stream()`.

## High-level API

```python
dispatch_out, handle = moe_comm.dispatch(
    input,
    topk_ids,
    weights=None,
    quant=None,
    output_buffer=output_buffer,
)

expert_output = mlp(dispatch_out)

output = moe_comm.combine(expert_output, handle)
```

`dispatch_out` is for the local MLP. `handle` is for `combine`. The MLP should
not need to inspect the opaque handle.

`DispatchOutput.layout` carries both the layout kind (`TOKEN_MAJOR` or `EXPERT_MAJOR`)
and layout-specific metadata.
Expert-grouped layouts populate
`num_tokens_per_expert`; future layouts that do not expose per-expert grouping
can leave those fields as `None`.

## Proposed types

```python
@dataclass
class QuantConfig:
    format: Optional[DispatchDataType] = None
    block_scales: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None


class DispatchLayout(str, Enum):
    EXPERT_MAJOR = "expert_major"
    TOKEN_MAJOR = "token_major"


@dataclass
class DispatchLayoutInfo:
    kind: DispatchLayout
    num_tokens_per_expert: Optional[torch.Tensor | list[int]] = None
    offsets: Optional[torch.Tensor] = None
    num_tokens_per_rank: Optional[torch.Tensor | list[int]] = None


@dataclass
class DispatchOutputInfo:
    layout: DispatchLayoutInfo
    quant: Optional[QuantConfig] = None


@dataclass
class DispatchOutput:
    tokens: torch.Tensor
    quant: Optional[QuantConfig]
    layout: DispatchLayoutInfo
    topk_ids: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None


@dataclass
class ExpertMajorCombineContext:
    topk_ids: torch.Tensor
    weights: torch.Tensor
    num_experts: int
    num_tokens: int
    hidden_size: int
    src_info: torch.Tensor
    layout_range: torch.Tensor
    num_max_dispatch_tokens_per_rank: int


@dataclass
class TokenMajorCombineContext:
    topk_ids: torch.Tensor
    num_experts: int
    num_tokens: int
    hidden_size: int
    source_token_ids: torch.Tensor
    num_tokens_per_rank: torch.Tensor
    rank_offsets: torch.Tensor
    num_max_dispatch_tokens_per_rank: int


@dataclass
class HighThroughputCombineContext:
    ...


CombineContext = ExpertMajorCombineContext | TokenMajorCombineContext | HighThroughputCombineContext


class DispatchHandle:
    """Base opaque handle returned by dispatch and consumed by combine."""

    output_info: DispatchOutputInfo


class ExpertMajorDispatchHandle(DispatchHandle):
    combine_context: ExpertMajorCombineContext


class TokenMajorDispatchHandle(DispatchHandle):
    combine_context: TokenMajorCombineContext


class HighThroughputDispatchHandle(DispatchHandle):
    combine_context: HighThroughputCombineContext


@dataclass
class OperationOverlapConfig:
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    num_comm_sms: Optional[int] = None


@dataclass
class BlockOverlapConfig:
    block_size_m: int
    ready_signal: torch.Tensor
    ready_value: int = 1
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    num_comm_sms: Optional[int] = None


@dataclass
class CommOverlapConfig:
    operation: Optional[OperationOverlapConfig] = None
    block: Optional[BlockOverlapConfig] = None

    @property
    def level(self) -> str: ...

```

`create_overlap_config` creates optional overlap configuration for async
dispatch/combine calls. The `op` argument is used only to validate construction;
the returned config describes how to overlap, not which operation will consume it.

```python
dispatch_overlap_config = moe_comm.create_overlap_config(op="dispatch")
combine_overlap_config = moe_comm.create_overlap_config(op="combine", handle=handle)
```

Operation-level overlap does not require `create_overlap_config`; `dispatch_async`
and `combine_async` can use their default stream/event behavior. Use
`create_overlap_config` when the caller wants explicit stream/event/SM settings
or block-level combine overlap.

For block-level MLP/combine overlap, the config includes the combine-side wait
protocol and the device signal that an overlap-capable MLP backend must publish:

```python
combine_overlap_config = moe_comm.create_overlap_config(
    op="combine",
    handle=handle,
    level="block",
)
```

`op="dispatch", level="block"` is not part of the first version. Dispatch
overlap is operation-level only.

`CommOverlapConfig` contains exactly one overlap mode:

| Field | Purpose |
|---|---|
| `operation` | Operation-level stream/event/SM config |
| `block` | Block-level ready-signal config |

`OperationOverlapConfig` fields:

| Field | Purpose |
|---|---|
| `stream` | Optional communication stream |
| `wait_event` | Optional event the communication op waits on before starting |
| `num_comm_sms` | Optional SM budget for communication |

`BlockOverlapConfig` fields:

| Field | Purpose |
|---|---|
| `block_size_m` | Rows/tokens per ready block |
| `ready_signal` | Device tensor written by MLP and waited on by combine |
| `ready_value` | Signal value that marks one block as ready for combine |
| `stream` | Optional communication stream |
| `wait_event` | Optional event the communication op waits on before starting |
| `num_comm_sms` | Optional SM budget for communication |

Each concrete `DispatchHandle` stores a layout-specific `combine_context` used
to reverse dispatch and finish combine. `ExpertMajorDispatchHandle` uses
`ExpertMajorCombineContext` (`topk_ids`, `weights`, source info, layout ranges,
shape, and capacity). `TokenMajorDispatchHandle` records source-token IDs,
per-source-rank counts, and the original routing needed for cross-rank combine.
High-throughput handles use the intranode combine context with
receive-side weights, source indices, prefix matrices, and send-head tensors.
The MLP should treat the handle as opaque and pass it back to `combine`.

## Dispatch inputs

### `input`

`input` is the original local token matrix.

```python
input: torch.Tensor  # [num_tokens, hidden]
```

Requirements:

| Field | Requirement |
|---|---|
| Shape | `[T, H]`, token-major |
| Layout | contiguous row-major |
| Device | CUDA tensor |
| dtype | BF16, FP16, FP8, NVFP4, MXFP8, or another supported activation dtype |
| Ordering | original local token order; not expert sorted |

The user should not expand `input` by top-k and should not convert it to
expert-major before calling `dispatch`.

`dispatch` includes any metadata exchange needed before moving token payloads.
For normal/high-throughput modes this typically means computing send counts from
`topk_ids`, exchanging counts or layout information across ranks, choosing recv
slots, and then dispatching the activation payload. Users should not call a
separate metadata-exchange API in the simple path.

### `topk_ids`

```python
topk_ids: torch.Tensor  # [T, K], int64
```

`topk_ids[t, k]` is the global expert id selected for token `t` at top-k slot
`k`. Invalid slots may use `-1` if supported by the backend.

### `weights`

```python
weights: Optional[torch.Tensor]  # [T, K], usually float32
```

These are MoE routing weights, not quantization scales. They are used by
combine to reduce the `K` expert results for each token back to `[T, H]`.

### `quant`

`quant` contains activation quantization metadata for `input`. It should be
`None` for BF16/FP16 input. `quant.format` defines the tensor representation
and scale layout.

Examples:

| Format | `input` | `quant.block_scales` | `quant.global_scale` |
|---|---|---|---|
| BF16/FP16 | `[T, H]` | `None` | `None` |
| FP8 E4M3 | `[T, H]` FP8 | `[T, H / 128]` | usually `None` |
| NVFP4 | backend-defined packed/logical `[T, H]` | block scale tensor | optional global scale |
| MXFP8 | backend-defined `[T, H]` | micro-scale tensor, e.g. E8M0 blocks | optional/global if required |

The API should not assume quantization scale is a scalar. For FP8 paths in
DeepEP/SGLang, scales are usually per token and per hidden block.

### `output_buffer`

Low-latency dispatch requires the caller to provide the receive token buffer:

```python
output_buffer: torch.Tensor
```

For padded expert-major LL layout:

```text
output_buffer: [num_local_experts, world_size * max_tokens_per_rank, hidden]
```

For token-major LL layout:

```text
output_buffer: [world_size * max_tokens_per_rank, hidden]
```

The token-major tensor keeps worst-case capacity to avoid a CPU synchronization,
but all valid rows are compacted into one contiguous prefix. For source rank
`r`, its rows are:

```python
begin = dispatch_out.layout.offsets[r]
end = dispatch_out.layout.offsets[r + 1]
```

`offsets[-1]` is the total number of valid rows.

The dtype must match the dispatch output dtype. For BF16 dispatch it is BF16.
For FP8 dispatch it is FP8 and the returned `DispatchOutput.quant` carries the
matching format and scale tensor.

`output_buffer` is required for LL because the MLP runner often owns or reuses
workspace memory. `MoECommunicator` writes dispatch output into the provided
buffer instead of allocating it internally.

## Dispatch output layout for MLP

`dispatch` should return MLP-ready tokens. The MLP should not run another
token-major to expert-major permutation unless it uses a custom adapter.

### Normal / high-throughput token-major layout

HT uses `DispatchLayout.TOKEN_MAJOR`:

```python
dispatch_out.tokens  # [total_recv_tokens, H]
```

Each row represents one `(source token, destination rank)` and is accompanied by
`dispatch_out.topk_ids`, `dispatch_out.weights`, and source-token metadata. A
token routed to multiple experts on the same destination rank is transferred
only once.

### Low-latency output layouts

LL defaults to `DispatchLayout.EXPERT_MAJOR`, a padded expert-major tensor:

```python
dispatch_out.tokens  # [num_local_experts, max_slots_per_expert, H]
```

LL can also return `DispatchLayout.TOKEN_MAJOR`:

```python
dispatch_out.tokens            # [world_size * max_tokens_per_rank, H]
dispatch_out.topk_ids          # [world_size * max_tokens_per_rank, K], int32 local expert IDs
dispatch_out.weights           # [world_size * max_tokens_per_rank, K], float32
```

Only the prefix ending at `dispatch_out.layout.offsets[-1]` is valid. Non-local
top-k entries use expert ID `-1` and weight `0`. Per-source-rank counts are
returned in `dispatch_out.layout.num_tokens_per_rank`.
For expert-major output, only the first
`dispatch_out.layout.num_tokens_per_expert[i]` slots are valid:

```python
expert_major_tokens = dispatch_out.tokens.view(num_local_experts, max_slots_per_expert, H)
expert_major_tokens[i, : dispatch_out.layout.num_tokens_per_expert[i], :]
```

The remaining slots are padding or scratch space. The MLP output must keep the
same layout and slot order.

### Scale output layout

If `dispatch_out.quant` is not `None`, its block scale tensor should follow
the same packed/expert-major layout as `dispatch_out.tokens`, with the hidden
dimension replaced by the scale dimension.

Examples:

```text
token-major tokens:   HT [total_recv_tokens, H]; LL [world_size * max_tokens_per_rank, H]
token-major scales:   LL [world_size * max_tokens_per_rank, H / 128]

expert-major tokens:  [num_local_experts, max_slots, H]
expert-major scales:  [num_local_experts, max_slots, H / 128]
```

## MLP contract

The MLP consumes `dispatch_out`, not the original token-major input.

For token-major output, the local MLP consumes each token once, runs the local
experts selected by `topk_ids`, applies `weights`, and returns one pre-reduced
rank partial in the same row:

```python
rank_partial = token_major_mlp(
    dispatch_out.tokens,
    dispatch_out.topk_ids,
    dispatch_out.weights,
    dispatch_out.quant,
)
```

For padded expert-major output:

```python
expert_output = expert_major_mlp(
    dispatch_out.tokens,
    dispatch_out.layout,
    dispatch_out.quant,
)
```

The MLP must preserve the dispatch output layout and row/slot order. For
token-major output, combine assumes each row is already weighted and reduced
across all local experts. `CombineMode.DIRECT_SEND` is therefore available only
for expert-major output.

## Combine API

```python
output = moe_comm.combine(
    expert_output,
    handle,
    out=None,
)
```

`expert_output` must have the same physical layout and order as
`dispatch_out.tokens`.

`combine` uses `handle` to:

- map each expert output row/slot back to the original source rank and token,
- find the corresponding top-k slot,
- apply the routing weight,
- reduce all selected expert outputs for each token,
- return local output in original token-major order.

The output shape is:

```python
output  # [T, H]
```

`combine` should not require users to pass `topk_ids`, `weights`, prefix
matrices, source indices, or layout ranges again. Those belong in `handle`.

An optional `weights` override could be added later, but the default API should
use the weights captured by `dispatch`.

## Communication/computation overlap

The default API should be blocking and simple:

```python
dispatch_out, handle = moe_comm.dispatch(
    input,
    topk_ids,
    weights,
    quant,
    output_buffer=output_buffer,
)
expert_output = mlp(dispatch_out.tokens, dispatch_out.layout)
output = moe_comm.combine(expert_output, handle)
```

For overlap, expose two optional APIs rather than adding many flags to the
default path:

| API | Purpose |
|---|---|
| `dispatch_async` / `combine_async` | Coarse-grained async launch and wait |
| `create_overlap_config(..., level="block")` | Fine-grained block notify between MLP down-GEMM and combine |

### Coarse-grained overlap

Coarse-grained overlap lets the caller launch communication on a communication
stream and wait later.

```python
dispatch_overlap_config = moe_comm.create_overlap_config(op="dispatch")
dispatch_req = moe_comm.dispatch_async(
    input,
    topk_ids,
    weights,
    quant,
    output_buffer=output_buffer,
    overlap_config=dispatch_overlap_config,
)

# Run unrelated work while dispatch metadata/payload communication is in flight.

dispatch_out, handle = dispatch_req.wait()
expert_output = mlp(dispatch_out.tokens, dispatch_out.layout)

combine_overlap_config = moe_comm.create_overlap_config(op="combine", handle=handle)
combine_req = moe_comm.combine_async(
    expert_output,
    handle,
    overlap_config=combine_overlap_config,
)

# Run unrelated work while combine is in flight.

output = combine_req.wait()
```

This is similar to the event/hook style used by DeepEP and SGLang. The request
object should own any stream event or receive hook required by the backend.

### Fine-grained MLP/combine overlap

Fine-grained overlap sends combine data as soon as the MLP produces a block.
This requires a device-side notify/signal from the MLP backend to the combine
kernel.

```python
combine_overlap_config = moe_comm.create_overlap_config(
    op="combine",
    handle=handle,
    level="block",
)

# User must adapt the MLP backend/adapter to consume this config and notify
# combine as blocks become ready.
config = combine_overlap_config
expert_output = mlp(
    dispatch_out.tokens,
    dispatch_out.layout,
    config=config,
)

combine_req = moe_comm.combine_async(
    expert_output,
    handle,
    overlap_config=combine_overlap_config,
)

output = combine_req.wait()
```

The overlap config is not routing metadata. It only tells combine when a
region of `expert_output` is ready to read. The routing/source mapping still
comes from `handle`.

The MLP backend must follow these rules when using notify:

- write `expert_output` in the same row/slot order as `dispatch_out.tokens`,
- publish data before signaling readiness,
- signal at the block granularity defined by `overlap_config.block.block_size_m`,
- use the ready value/protocol provided by `overlap_config.block`.

If the MLP backend does not support notify, it can still use the blocking API or
coarse-grained `combine_async` after the full `expert_output` tensor is ready.

This must be a joint contract between the dispatcher and the MLP runner. The
dispatcher can provide the signal buffer and combine-side wait protocol, but it
cannot infer readiness by itself. The MLP runner must write the signal after it
finishes the corresponding output region.

SGLang follows this model for its DeepEP low-latency path. It computes overlap
arguments after dispatch, passes combine-side arguments to the DeepEP dispatcher,
and passes down-GEMM arguments to the MoE runner. Backend support is selective:

- DeepGEMM FP8 masked down-GEMM can return block metadata such as `block_size_m`
  and `ready_value` and signal combine readiness.
- FlashInfer CuteDSL can receive down-GEMM signal/start-event arguments.
- Some paths, such as BF16 masked DeepGEMM and generic Triton runners, do not
  support this block overlap protocol.

Therefore, the API should expose overlap as an optional capability advertised by
the MLP backend, not as a guaranteed feature of every `combine_async` call.

## Internal metadata exchange

Normal/high-throughput dispatch usually needs a metadata phase before payload
movement:

```text
topk_ids
  -> compute send counts per rank/expert
  -> exchange counts or layout metadata
  -> compute recv slots and local expert counts
  -> dispatch token payload
```

Low-latency modes may use fixed-capacity buffers and device-side counters, but
they still generate metadata such as source info, layout ranges, and valid
counts.

These details should remain internal. The user-facing API should only expose
MLP-relevant layout information through `DispatchOutput` and combine-relevant
metadata through `DispatchHandle`.

## Example

```python
recv, handle = moe_comm.dispatch(
    input=hidden_states,          # [T, H]
    topk_ids=topk_ids,            # [T, K]
    weights=topk_weights,         # [T, K]
    quant=None,                   # BF16 path
    output_buffer=recv_buffer,
)

expert_output = triton_grouped_mlp(
    recv.tokens,
    recv.layout,
)

output = moe_comm.combine(expert_output, handle)
```

Quantized path:

```python
moe_comm = MoECommunicator(
    ...,
    quant=QuantConfig(format=DispatchDataType.FP8_E4M3),
)

recv, handle = moe_comm.dispatch(
    input=hidden_states,          # BF16 input, quantized during dispatch
    topk_ids=topk_ids,
    weights=topk_weights,
    quant=None,
    output_buffer=recv_buffer,
)

expert_output = fp8_grouped_mlp(
    recv.tokens,
    recv.quant,
    recv.layout,
)

output = moe_comm.combine(expert_output, handle)
```
