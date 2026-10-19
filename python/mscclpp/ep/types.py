# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Public configuration, GPU metadata, and opaque operation handles."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Union

import torch

from mscclpp import CommGroup

from . import _cpp
from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode


@dataclass
class QuantConfig:
    """Payload format and optional quantization scales.

    FP8 E4M3 uses one FP32 scale per 128 hidden elements.
    For FP8 dispatch, latency quantizes BF16 input
    and returns scales with logical shape ``[local_expert, row, hidden // 128]``
    and transposed physical storage. Throughput FP8 dispatch requires contiguous
    input scales with shape ``[num_tokens, hidden // 128]``.
    BF16 dispatch does not use scales.
    """

    format: Optional[DispatchDataType] = None
    block_scales: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MoECommunicatorConfig:
    """Fixed runtime configuration, which must agree across participating ranks.

    Experts are partitioned evenly into contiguous rank-local ranges.
    ``num_blocks`` is ``(dispatch_grid_blocks, combine_grid_blocks)``. A scalar
    ``N`` sets the dispatch grid to ``N`` blocks. Latency dispatch reserves two
    of those blocks for control, so combine uses the remaining ``N - 2`` worker
    blocks; throughput uses ``N`` blocks for both operations. Either entry of a
    pair may be ``None`` for its mode default. ``quant`` supplies a default
    which a dispatch's explicit quant config overrides.
    """

    comm: CommGroup
    num_experts: int
    hidden_size: int
    topk: int
    max_tokens_per_rank: int
    num_local_experts: int = field(init=False)
    local_expert_start: int = field(init=False)
    device: Optional[torch.device] = None
    mode: MoEMode = MoEMode.LATENCY
    output_layout: Optional[DispatchLayout] = None
    invalid_token_expert_id: Optional[int] = None
    quant: Optional[QuantConfig] = None
    num_blocks: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    combine_mode: CombineMode = CombineMode.RANK_LOCAL_REDUCE

    def __post_init__(self) -> None:
        """Validate and resolve configuration for the communicator and CUDA device."""
        from .utils import resolve_num_blocks, validate_quant

        for name in ("num_experts", "hidden_size", "topk", "max_tokens_per_rank"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < (1 << 31):
                raise ValueError(f"{name} must be a positive int32")
        if self.topk > 8:
            raise ValueError("topk must be in [1, 8]")
        if not isinstance(self.mode, MoEMode):
            raise TypeError("mode must be a MoEMode")
        if not isinstance(self.combine_mode, CombineMode):
            raise TypeError("combine_mode must be a CombineMode")
        latency = self.mode == MoEMode.LATENCY
        layout = self.output_layout
        if layout is None:
            layout = DispatchLayout.EXPERT_MAJOR if latency else DispatchLayout.TOKEN_MAJOR
        if not isinstance(layout, DispatchLayout):
            raise TypeError("output_layout must be a DispatchLayout")
        supported = (
            (DispatchLayout.EXPERT_MAJOR, DispatchLayout.RANK_MAJOR)
            if latency
            else (DispatchLayout.TOKEN_MAJOR, DispatchLayout.RANK_MAJOR)
        )
        if layout not in supported:
            raise ValueError("output_layout is unsupported for the selected mode")
        if not latency and self.combine_mode != CombineMode.RANK_LOCAL_REDUCE:
            raise ValueError("THROUGHPUT supports only RANK_LOCAL_REDUCE combine")
        if latency:
            if self.hidden_size not in (1024, 2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216):
                raise ValueError(
                    "latency hidden_size must be one of 1024, 2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216"
                )
        elif self.hidden_size % 8:
            raise ValueError("throughput combine requires 16-byte-aligned BF16 rows, even after FP8 dispatch")
        if self.comm is None:
            raise ValueError("MoECommunicatorConfig requires an mscclpp.CommGroup via comm")
        if not isinstance(self.comm, CommGroup):
            raise TypeError("comm must be an mscclpp.CommGroup")
        world_size = self.comm.nranks
        if not 1 <= world_size <= 64 or self.comm.nranks_per_ipc_domain != world_size:
            raise ValueError("EP requires 1-64 ranks in one CUDA IPC domain")
        if self.num_experts % world_size:
            raise ValueError("num_experts must be divisible by world_size")
        num_local = self.num_experts // world_size
        if not latency and num_local > 128:
            raise ValueError("throughput requires at most 128 experts per rank")
        invalid_id = self.invalid_token_expert_id
        if not latency and invalid_id is not None and invalid_id != -1:
            raise ValueError("invalid_token_expert_id is supported only in LATENCY mode; throughput uses -1")
        if invalid_id is None:
            invalid_id = self.num_experts if latency else -1
        if type(invalid_id) is not int or not -(1 << 31) <= invalid_id < (1 << 31):
            raise ValueError("invalid_token_expert_id must fit in int32")
        if 0 <= invalid_id < self.num_experts:
            raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
        validate_quant(self.quant, self.mode, layout, self.hidden_size)
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("MSCCL++ EP requires CUDA and an SM90 or newer GPU")
        device = self.device
        if device is not None and not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device or None")
        if device is not None and device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        if device is None or device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        if properties.major < 9:
            raise RuntimeError("MSCCL++ EP requires an SM90 or newer GPU")
        num_sms = properties.multi_processor_count
        blocks = resolve_num_blocks(
            self.num_blocks,
            default=(min(130, num_sms), min(128, num_sms)) if latency else (min(24, num_sms), min(32, num_sms)),
            scalar_combine_offset=-2 if latency else 0,
        )
        if not (world_size + 2 if latency else 1) <= blocks[0] <= 130 or not 1 <= blocks[1] <= 128:
            raise ValueError(
                "num_blocks must satisfy dispatch <= 130 and combine <= 128; "
                "latency dispatch requires at least world_size + 2 blocks, and all counts must be positive"
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "output_layout", layout)
        object.__setattr__(self, "num_local_experts", num_local)
        object.__setattr__(self, "local_expert_start", self.comm.my_rank * num_local)
        object.__setattr__(self, "invalid_token_expert_id", invalid_id)
        object.__setattr__(self, "num_blocks", blocks)
        object.__setattr__(self, "quant", None if self.quant is None else replace(self.quant))


@dataclass
class DispatchLayoutInfo:
    """GPU-resident per-expert or per-rank counts for dispatched tensors.

    TOKEN_MAJOR outputs are capacity-sized; the native combine operation uses
    private routing metadata and ignores unused tail rows. Its per-expert counts
    count routes and do not define disjoint token ranges.
    RANK_MAJOR uses ``num_tokens_per_rank`` to bound each source rank.
    EXPERT_MAJOR uses ``num_tokens_per_expert`` to bound each local expert.
    Throughput dispatch exposes the caller-provided ``output_count`` from its
    preparation handle in the field matching the layout. Both fields remain
    ``None`` when preparation did not request counts.
    No count is read back to the host.
    """

    kind: DispatchLayout
    num_tokens_per_expert: Optional[torch.Tensor] = None
    num_tokens_per_rank: Optional[torch.Tensor] = None


@dataclass
class DispatchOutputInfo:
    """Layout and quantization metadata shared by an output and its handle."""

    layout: DispatchLayoutInfo
    quant: Optional[QuantConfig] = None


@dataclass
class DispatchOutput:
    """Capacity-sized activations and routing metadata for local expert compute.

    Runtime-owned views retain native storage even after slicing, but their
    contents are reused by later operations. Throughput uses separate dispatch
    and BF16 combine buffers, including when dispatch emits FP8. Except for
    latency EXPERT_MAJOR, write expert results into ``combine_input_buffer``.
    """

    tokens: torch.Tensor
    quant: Optional[QuantConfig]
    layout: DispatchLayoutInfo
    topk_ids: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    combine_input_buffer: Optional[torch.Tensor] = None


@dataclass(frozen=True, eq=False)
class PrepareHandle:
    """Opaque reusable throughput routing from ``communicator.prepare``.

    Keep routing IDs unchanged until all dispatches using this preparation have
    finished. A new preparation (including automatic preparation by dispatch)
    invalidates this handle. The native runtime validates freshness and matching
    routing pointer, token count, active capacity, and dispatch block count.
    Retains optional output counts, which dispatch exposes without copying.
    """

    _native: _cpp.PrepareHandle = field(repr=False)
    _runtime: _cpp.MoERuntime = field(repr=False)
    _topk_ids: torch.Tensor = field(repr=False)
    _stream: torch.cuda.Stream = field(repr=False)
    _output_count: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass(frozen=True, eq=False)
class DispatchHandle:
    """Opaque dispatch state consumed by ``communicator.combine``.

    Retains the native runtime and all tensors borrowed by the operation through
    combine. A successful later dispatch or preparation invalidates this handle.
    Retention is not synchronization: finish local and peer GPU use before
    releasing the last runtime/handle/view, including CUDA graph replays.
    """

    output_info: DispatchOutputInfo
    _native: _cpp.DispatchHandle = field(repr=False)
    _runtime: _cpp.MoERuntime = field(repr=False)
    _num_tokens: int = field(repr=False)
    _active_capacity: int = field(repr=False)
    _tensors: Tuple[torch.Tensor, ...] = field(repr=False)
    _stream: torch.cuda.Stream = field(repr=False)
