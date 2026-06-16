# Expert Parallel Python API Design Draft

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
from mscclpp.ext.ep import MoECommunicator

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
    mode: str = "ht"            # "ht" or "ll"
    output_layout: Optional[str] = None  # default is derived from mode

    # Quantization defaults
    input_dtype: Optional[torch.dtype] = None
    quant_format: Optional[str] = None

    # Scratch / transport resources
    num_nvl_bytes: Optional[int] = None
    num_rdma_bytes: Optional[int] = None
    num_rdma_qps_per_rank: int = 12  # RDMA QPs per peer rank; advanced tuning
    num_sms: int = 20

    # Streams and overlap
    comm_stream: Optional[torch.cuda.Stream] = None
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
    mode="ht",
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
| `buffer` / `runtime` | pybind/C++ EP buffer implementation |
| `comm_stream` | optional stream for async dispatch/combine |

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
| `mode` | HT/normal or LL backend selection |
| `output_layout` | MLP input layout returned by dispatch |
| `max_tokens_per_rank` | dispatch capacity |
| `max_recv_tokens_per_rank` | recv buffer capacity |
| `num_nvl_bytes`, `num_rdma_bytes` | scratch buffer sizing |
| `num_rdma_qps_per_rank`, `num_sms` | backend launch/resource tuning |
| `dispatch_config`, `combine_config` | backend-specific tuning configs |
| `overlap_capability` | whether selected MLP/backend supports notify |

The user should not pass these to `dispatch` unless explicitly overriding a
specialized advanced path.

### Mode selection

The first version should require the upper layer to select the communication
mode explicitly:

```python
moe_comm = MoECommunicator(..., mode="ht")  # normal/high-throughput
moe_comm = MoECommunicator(..., mode="ll")  # low-latency
```

This keeps `MoECommunicator` policy-free. Serving frameworks such as SGLang can
choose HT for prefill and LL for decode based on their own scheduling policy,
batch shape, runner backend, and benchmarking data.

The selected mode determines the default dispatch output layout:

| Mode | Default layout |
|---|---|
| `ht` | `flat_expert_major` |
| `ll` | `padded_expert_major` |

`output_layout` may still be kept as an advanced override if a backend supports
multiple layouts within the same mode.

## MoECommunicator methods

```python
class MoECommunicator:
    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        scales: Optional[QuantScales] = None,
    ) -> tuple[DispatchOutput, DispatchHandle]:
        ...

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
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

## High-level API

```python
dispatch_out, handle = moe_comm.dispatch(
    input,
    topk_ids,
    weights=None,
    scales=None,
)

expert_output = mlp(
    dispatch_out.tokens,
    dispatch_out.num_tokens_per_expert,
    dispatch_out.scales,
)

output = moe_comm.combine(expert_output, handle)
```

`dispatch_out` is for the local MLP. `handle` is for `combine`. The MLP should
not need to inspect the opaque handle.

## Proposed types

```python
@dataclass
class QuantScales:
    local: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    format: Optional[str] = None
    block_size: Optional[int] = None


@dataclass
class DispatchOutput:
    tokens: torch.Tensor
    scales: Optional[QuantScales]
    num_tokens_per_expert: torch.Tensor | list[int]
    expert_offsets: Optional[torch.Tensor] = None
    layout: str = "flat_expert_major"


class DispatchHandle:
    """Opaque handle returned by dispatch and consumed by combine."""


@dataclass
class CommOverlapConfig:
    op: str  # "dispatch" or "combine"
    level: str = "op"  # "op" or "block"
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    signal: Optional[torch.Tensor] = None
    num_comm_sms: Optional[int] = None
    block_m: Optional[int] = None
    block_ready_value: Optional[int] = None

```

`create_overlap_config` creates optional overlap configuration for async
dispatch/combine calls.

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

`CommOverlapConfig` fields:

| Field | Purpose |
|---|---|
| `op` | `"dispatch"` or `"combine"` |
| `level` | `"op"` or `"block"` |
| `stream` | Optional communication stream |
| `wait_event` | Optional event the communication op waits on before starting |
| `signal` | Device tensor written by MLP and waited on by combine for block overlap |
| `num_comm_sms` | Optional SM budget for communication |
| `block_m` | Rows per block for block overlap |
| `block_ready_value` | Signal value that marks one block as ready for combine |

`DispatchHandle` should store the metadata needed to reverse dispatch:

- source rank and source token index,
- top-k slot or equivalent routing metadata,
- top-k ids and routing weights, or stable references/copies,
- dispatch layout/range/count metadata,
- capacity, local expert placement, and launch parameters needed by kernels,
- optional cached metadata for repeated routing.

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

### `scales`

`scales` contains activation quantization metadata for `input`. It should be
`None` for BF16/FP16 input.

Examples:

| Format | `input` | `scales.local` | `scales.global_scale` |
|---|---|---|---|
| BF16/FP16 | `[T, H]` | `None` | `None` |
| FP8 E4M3 | `[T, H]` FP8 | `[T, H / block_size]`, often block size 128 | usually `None` |
| NVFP4 | backend-defined packed/logical `[T, H]` | block scale tensor | optional global scale |
| MXFP8 | backend-defined `[T, H]` | micro-scale tensor, e.g. E8M0 blocks | optional/global if required |

The API should not assume quantization scale is a scalar. For FP8 paths in
DeepEP/SGLang, scales are usually per token and per hidden block.

## Dispatch output layout for MLP

`dispatch` should return MLP-ready tokens. The MLP should not run another
token-major to expert-major permutation unless it uses a custom adapter.

### Normal / high-throughput mode

Prefer a flat expert-major layout:

```python
dispatch_out.tokens  # [total_recv_tokens, H]
```

Rows are grouped by local expert id:

```text
expert0 tokens
expert1 tokens
expert2 tokens
...
```

`dispatch_out.num_tokens_per_expert` is ordered by local expert id:

```python
num_tokens_per_expert[i] = valid token count for local expert i
```

For flat layout, `expert_offsets` may be provided or derived by cumulative sum:

```python
expert_offsets = cumsum([0] + num_tokens_per_expert)
tokens[expert_offsets[i] : expert_offsets[i + 1]]
```

This layout is efficient for Triton or grouped GEMM kernels because it avoids
padding.

### Low-latency expert-major mode

Some backends may return a padded expert-major tensor:

```python
dispatch_out.tokens  # [num_local_experts, max_slots_per_expert, H]
```

For expert `i`, only the first `num_tokens_per_expert[i]` slots are valid:

```python
tokens[i, :num_tokens_per_expert[i], :]
```

The remaining slots are padding or scratch space. The MLP output must keep the
same layout and slot order.

### Scale output layout

If `dispatch_out.scales` is not `None`, its local scale tensor should follow
the same packed/expert-major layout as `dispatch_out.tokens`, with the hidden
dimension replaced by the scale dimension.

Examples:

```text
flat tokens:          [total_recv_tokens, H]
flat FP8 scales:      [total_recv_tokens, H / 128]

expert-major tokens:  [num_local_experts, max_slots, H]
expert-major scales:  [num_local_experts, max_slots, H / 128]
```

## MLP contract

The MLP consumes `dispatch_out`, not the original token-major input.

For flat expert-major output:

```python
expert_output = triton_mlp(
    dispatch_out.tokens,
    dispatch_out.num_tokens_per_expert,
    dispatch_out.scales,
)
```

For padded expert-major output:

```python
expert_output = expert_major_mlp(
    dispatch_out.tokens,
    dispatch_out.num_tokens_per_expert,
    dispatch_out.scales,
)
```

The MLP must preserve the dispatch output layout and row/slot order. It may
apply expert-specific GEMMs, but it must not compact or reorder tokens unless it
also produces compatible metadata for combine.

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
dispatch_out, handle = moe_comm.dispatch(input, topk_ids, weights, scales)
expert_output = mlp(dispatch_out.tokens, dispatch_out.num_tokens_per_expert)
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
    scales,
    overlap_config=dispatch_overlap_config,
)

# Run unrelated work while dispatch metadata/payload communication is in flight.

dispatch_out, handle = dispatch_req.wait()
expert_output = mlp(dispatch_out.tokens, dispatch_out.num_tokens_per_expert)

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
    dispatch_out.num_tokens_per_expert,
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
- signal at the block granularity defined by `overlap_config`,
- use the signal value/protocol provided by `overlap_config`.

If the MLP backend does not support notify, it can still use the blocking API or
coarse-grained `combine_async` after the full `expert_output` tensor is ready.

This must be a joint contract between the dispatcher and the MLP runner. The
dispatcher can provide the signal buffer and combine-side wait protocol, but it
cannot infer readiness by itself. The MLP runner must write the signal after it
finishes the corresponding output region.

SGLang follows this model for its DeepEP low-latency path. It computes overlap
arguments after dispatch, passes combine-side arguments to the DeepEP dispatcher,
and passes down-GEMM arguments to the MoE runner. Backend support is selective:

- DeepGEMM FP8 masked down-GEMM can return block metadata such as `block_m` and
  `block_ready_value` and signal combine readiness.
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
    scales=None,                  # BF16 path
)

expert_output = triton_grouped_mlp(
    recv.tokens,
    recv.num_tokens_per_expert,
)

output = moe_comm.combine(expert_output, handle)
```

Quantized path:

```python
recv, handle = moe_comm.dispatch(
    input=x_fp8,
    topk_ids=topk_ids,
    weights=topk_weights,
    scales=QuantScales(
        local=x_scales,
        format="fp8_e4m3",
        block_size=128,
    ),
)

expert_output = fp8_grouped_mlp(
    recv.tokens,
    recv.scales,
    recv.num_tokens_per_expert,
)

output = moe_comm.combine(expert_output, handle)
```

## Open questions

- Whether `weights` should be required for dispatch or allowed to be `None`
  for unweighted combine.
- Whether `DispatchHandle` should copy `topk_ids` and `weights` for safety or
  hold references for lower overhead.
- How to represent backend-specific quantization formats beyond FP8, NVFP4,
  and MXFP8 without growing the dispatch argument list.
