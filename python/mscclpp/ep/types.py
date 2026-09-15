# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Public configuration, GPU metadata, and opaque operation handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from mscclpp import CommGroup

from . import _cpp
from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode


@dataclass
class QuantConfig:
    """Payload format and optional FP32 scales, one per 128 hidden elements.

    Latency dispatch quantizes BF16 input and returns scales with logical shape
    ``[local_expert, row, hidden // 128]`` and transposed physical storage.
    Throughput FP8 dispatch requires contiguous input scales with shape
    ``[num_tokens, hidden // 128]``. BF16 dispatch does not use scales.
    """

    format: Optional[DispatchDataType] = None
    block_scales: Optional[torch.Tensor] = None


@dataclass
class MoECommunicatorConfig:
    """Fixed runtime configuration, which must agree across participating ranks.

    Experts are partitioned evenly into contiguous rank-local ranges. A scalar
    ``num_blocks=N`` resolves to ``(N, N - 2)`` for latency or ``(N, N)`` for
    throughput; either entry of a pair may be ``None`` for its mode default.
    ``quant`` supplies a default which a dispatch's explicit quant config overrides.
    """

    comm: Optional[CommGroup] = None
    device: Optional[Union[torch.device, int, str]] = None
    num_experts: int = 0
    num_local_experts: Optional[int] = None
    local_expert_start: Optional[int] = None
    hidden_size: int = 0
    topk: int = 0
    max_tokens_per_rank: int = 0
    mode: MoEMode = MoEMode.LATENCY
    output_layout: Optional[DispatchLayout] = None
    invalid_token_expert_id: Optional[int] = None
    quant: Optional[QuantConfig] = None
    num_blocks: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    combine_mode: CombineMode = CombineMode.RANK_LOCAL_REDUCE


@dataclass
class DispatchLayoutInfo:
    """GPU-resident counts describing valid rows inside capacity-sized tensors.

    TOKEN_MAJOR rows are dense only up to ``num_recv_tokens``; per-expert counts
    cannot be summed to obtain this value when tokens select multiple local
    experts. RANK_MAJOR uses ``num_tokens_per_rank`` to bound each source rank.
    EXPERT_MAJOR uses ``num_tokens_per_expert`` to bound each local expert.
    ``num_recv_tokens`` is a borrowed read-only scalar overwritten by preparation.
    Do not modify it; PyTorch cannot enforce read-only access to this storage.
    No count is read back to the host. ``offsets`` is reserved and is not populated.
    """

    kind: DispatchLayout
    num_tokens_per_expert: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None
    num_tokens_per_rank: Optional[torch.Tensor] = None
    num_recv_tokens: Optional[torch.Tensor] = None

    @property
    def num_tokens(self) -> Optional[torch.Tensor]:
        """Alias for the throughput runtime's scalar GPU receive-row count."""
        return self.num_recv_tokens


@dataclass
class DispatchOutputInfo:
    """Layout and quantization metadata shared by an output and its handle."""

    layout: DispatchLayoutInfo
    quant: Optional[QuantConfig] = None


@dataclass
class DispatchOutput:
    """Capacity-sized activations and routing metadata for local expert compute.

    Runtime-owned views retain native storage even after slicing, but their
    contents are reused by later operations. Throughput's BF16 combine buffer
    aliases its dispatch buffer, including when dispatch emits FP8. Consume FP8
    tokens before writing BF16 results into that allocation.
    """

    tokens: torch.Tensor
    quant: Optional[QuantConfig]
    layout: DispatchLayoutInfo
    topk_ids: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    combine_input_buffer: Optional[torch.Tensor] = None


@dataclass(frozen=True, eq=False)
class PrepareHandle:
    """Opaque reusable throughput routing; create with ``communicator.prepare``.

    Keep routing IDs unchanged until all dispatches using this preparation have
    finished. A new preparation (including automatic preparation by dispatch)
    invalidates this handle. The native runtime validates freshness.
    """

    _native: _cpp.PrepareHandle = field(repr=False)
    _runtime: _cpp.MoERuntime = field(repr=False)
    _config: tuple = field(repr=False)
    _topk_ids: torch.Tensor = field(repr=False)
    _topk_ptr: int = field(repr=False)
    _num_tokens: int = field(repr=False)
    _active_capacity: int = field(repr=False)
    _stream: torch.cuda.Stream = field(repr=False)


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
    _config: tuple = field(repr=False)
    _num_tokens: int = field(repr=False)
    _active_capacity: int = field(repr=False)
    _tensors: Tuple[torch.Tensor, ...] = field(repr=False)
    _combine_input_buffer: Optional[torch.Tensor] = field(repr=False)
    _stream: torch.cuda.Stream = field(repr=False)
