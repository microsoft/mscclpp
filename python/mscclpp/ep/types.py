# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Public data types for the expert-parallel Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
import mscclpp
from ._cpp import CombineMode, DispatchLayout, MoEMode

# Quantization metadata.


@dataclass
class QuantConfig:
    """Quantization metadata associated with an activation tensor."""

    dtype: Optional[torch.dtype] = None
    block_scales: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    block_size: Optional[int] = None


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
    max_recv_tokens_per_rank: Optional[int] = None

    # Runtime mode and output layout
    mode: MoEMode = MoEMode.LOW_LATENCY
    output_layout: Optional[DispatchLayout] = None

    # Quantization defaults
    quant: Optional[QuantConfig] = None

    # Transport / launch tuning
    num_rdma_qps_per_rank: int = 12
    num_sms: int = 20
    low_latency_num_blocks: int = 130
    low_latency_combine_mode: CombineMode = CombineMode.RANK_LOCAL_REDUCE
    enable_overlap: bool = False

    # HT-only buffer/launch tuning (advanced)
    expert_alignment: int = 1
    nvl_chunked_send: int = 8
    nvl_chunked_recv: int = 256
    rdma_chunked_send: int = 16
    rdma_chunked_recv: int = 128


# MLP-facing dispatch output.


@dataclass
class DispatchLayoutInfo:
    """Physical layout of dispatched tokens and optional expert-group metadata."""

    kind: DispatchLayout
    num_tokens_per_expert: Optional[Union[torch.Tensor, List[int]]] = None
    offsets: Optional[torch.Tensor] = None


@dataclass
class DispatchOutputInfo:
    """Lightweight output metadata copied into both dispatch output and handle."""

    layout: DispatchLayoutInfo
    quant: Optional[QuantConfig] = None


@dataclass
class DispatchOutput:
    """Dispatch result consumed by the local MLP."""

    tokens: torch.Tensor
    quant: Optional[QuantConfig]
    layout: DispatchLayoutInfo


# Combine-side context. These objects are layout-specific and opaque to the MLP.


@dataclass
class ExpertMajorCombineContext:
    """Combine context for expert-major dispatch output."""

    topk_ids: torch.Tensor
    weights: Optional[torch.Tensor]
    num_experts: int
    num_tokens: int
    hidden_size: int
    src_info: torch.Tensor
    layout_range: torch.Tensor
    num_max_dispatch_tokens_per_rank: int


@dataclass
class RowMajorIntranodeCombineContext:
    """Combine context for row-major intranode dispatch output."""

    recv_topk_weights: Optional[torch.Tensor]
    src_idx: torch.Tensor
    rank_prefix_matrix: torch.Tensor
    recv_channel_prefix_matrix: torch.Tensor
    send_head: torch.Tensor


@dataclass
class RowMajorInternodeCombineContext:
    """Combine context for row-major internode dispatch output."""

    recv_topk_weights: Optional[torch.Tensor]
    src_meta: torch.Tensor
    is_token_in_rank: torch.Tensor
    recv_rdma_channel_prefix_matrix: torch.Tensor
    recv_rdma_rank_prefix_sum: torch.Tensor
    recv_gbl_channel_prefix_matrix: torch.Tensor
    send_rdma_head: torch.Tensor
    send_nvl_head: torch.Tensor


CombineContext = Union[ExpertMajorCombineContext, RowMajorIntranodeCombineContext, RowMajorInternodeCombineContext]


# Opaque dispatch handles returned by dispatch() and consumed by combine().


@dataclass
class DispatchHandle:
    """Base opaque dispatch metadata consumed by :meth:`MoECommunicator.combine`."""

    output_info: DispatchOutputInfo


@dataclass
class ExpertMajorDispatchHandle(DispatchHandle):
    combine_context: ExpertMajorCombineContext


@dataclass
class RowMajorIntranodeDispatchHandle(DispatchHandle):
    combine_context: RowMajorIntranodeCombineContext


@dataclass
class RowMajorInternodeDispatchHandle(DispatchHandle):
    combine_context: RowMajorInternodeCombineContext


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
