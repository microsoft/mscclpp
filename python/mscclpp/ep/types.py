# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Public data types for the expert-parallel Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import mscclpp
from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode

# Quantization metadata.


@dataclass
class QuantConfig:
    """Quantization metadata associated with an activation tensor.

    Latency FP8 dispatch returns ``block_scales`` with the activation's
    leading dimensions and a format-defined final scale dimension. ``FP8_E4M3``
    uses FP32 scales per 128 elements.
    """

    format: Optional[DispatchDataType] = None
    block_scales: Optional[torch.Tensor] = None


# Communicator construction.


@dataclass
class MoECommunicatorConfig:
    """Configuration for the high-level MoE dispatch/combine API."""

    comm: Optional[mscclpp.CommGroup] = None
    device: Optional[Union[torch.device, int]] = None

    # Expert topology
    num_experts: int = 0
    num_local_experts: Optional[int] = None
    local_expert_start: Optional[int] = None

    # Model shape and capacity
    hidden_size: int = 0
    topk: int = 0
    max_tokens_per_rank: int = 0

    # Runtime mode and output layout
    mode: MoEMode = MoEMode.LATENCY
    output_layout: Optional[DispatchLayout] = None
    # Latency rank-major sentinel; None resolves to num_experts.
    invalid_token_expert_id: Optional[int] = None

    # Quantization defaults
    quant: Optional[QuantConfig] = None

    # Launch tuning
    num_sms: int = 20
    low_latency_num_blocks: int = 130
    low_latency_combine_mode: CombineMode = CombineMode.RANK_LOCAL_REDUCE
    enable_overlap: bool = False

    # Overlap receive-pool tuning (advanced)
    expert_alignment: int = 1


# MLP-facing dispatch output.


@dataclass
class DispatchLayoutInfo:
    """Physical layout of dispatched tokens and optional rank/expert metadata."""

    kind: DispatchLayout
    num_tokens_per_expert: Optional[Union[torch.Tensor, List[int]]] = None
    offsets: Optional[torch.Tensor] = None
    num_tokens_per_rank: Optional[Union[torch.Tensor, List[int]]] = None


@dataclass
class DispatchOutputInfo:
    """Lightweight output metadata copied into both dispatch output and handle."""

    layout: DispatchLayoutInfo
    quant: Optional[QuantConfig] = None


@dataclass
class DispatchOutput:
    """Dispatch result consumed by the local MLP.

    ``RANK_MAJOR`` tensors alias runtime-owned registered buffers that are
    reused by every dispatch. Clone any result that must outlive the next call.
    """

    tokens: torch.Tensor
    quant: Optional[QuantConfig]
    layout: DispatchLayoutInfo
    topk_ids: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None


# Private combine-side context.


@dataclass
class _ExpertMajorCombineContext:
    """Combine context for expert-major dispatch output."""

    topk_ids: torch.Tensor
    weights: Optional[torch.Tensor]
    num_experts: int
    num_tokens: int
    hidden_size: int
    src_info: torch.Tensor
    layout_range: torch.Tensor


@dataclass
class _RankMajorCombineContext:
    """Combine context for fixed-stride rank-major output."""

    topk_ids: torch.Tensor
    num_experts: int
    num_tokens: int
    hidden_size: int
    max_tokens_per_rank: int


@dataclass
class _TokenMajorOverlapCombineContext:
    """Combine context for token-major overlap output."""

    recv_topk_weights: Optional[torch.Tensor]
    send_head: torch.Tensor


_CombineContext = Union[
    _ExpertMajorCombineContext,
    _RankMajorCombineContext,
    _TokenMajorOverlapCombineContext,
]


# Opaque dispatch handles returned by dispatch() and consumed by combine().


@dataclass
class DispatchHandle:
    """Opaque dispatch metadata consumed by :meth:`MoECommunicator.combine`."""

    output_info: DispatchOutputInfo
    _context: _CombineContext


# Optional async/overlap configuration.


@dataclass
class OperationOverlapConfig:
    """Operation-level communication overlap controls."""

    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    num_comm_sms: Optional[int] = None


@dataclass
class BlockOverlapConfig:
    """Block-level MLP/combine overlap controls."""

    block_size_m: int
    ready_signal: torch.Tensor
    ready_value: int = 1
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    num_comm_sms: Optional[int] = None


@dataclass
class CommOverlapConfig:
    """Mutually exclusive operation-level or block-level overlap configuration."""

    operation: Optional[OperationOverlapConfig] = None
    block: Optional[BlockOverlapConfig] = None

    def __post_init__(self) -> None:
        if (self.operation is None) == (self.block is None):
            raise ValueError("exactly one of operation or block overlap config must be set")

    @property
    def level(self) -> str:
        return "block" if self.block is not None else "op"
