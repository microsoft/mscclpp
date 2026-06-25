# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""High-level MoE expert-parallel communicator (MoECommunicator).

This module implements the high-level API described in
``python/mscclpp/ext/ep/README.md`` on top of the low-level
:class:`mscclpp.ext.ep.Buffer` (DeepEP-style) runtime.

The first implementation covers ``mode="ht"`` (high-throughput,
``DispatchLayout.FLAT``) — the NVLink+RDMA dispatch/combine kernels that power
``test/python/ext/ep/test_internode_multirank.py`` and
``test/python/ext/ep/test_intranode_multirank.py``. Intranode (single NVLink
domain) vs internode (NVLink + RDMA) is selected internally from the world size;
the user never picks the transport.

The user passes the tensors the model owns (``input``, ``topk_ids``, ``weights``,
``scales``); :meth:`MoECommunicator.dispatch` returns MLP-ready FLAT tokens plus
a per-local-expert count, and :meth:`MoECommunicator.combine` reverses it from an
opaque :class:`DispatchHandle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .buffer import Buffer, Config


class DispatchLayout(str, Enum):
    """MLP input layout returned by :meth:`MoECommunicator.dispatch`."""

    #: ``[total_recv_tokens, hidden]`` rows grouped by local expert id (HT).
    FLAT = "flat"
    #: ``[num_local_experts, max_slots_per_expert, hidden]`` padded (LL).
    EXPERT_MAJOR = "expert_major"


@dataclass
class QuantScales:
    """Activation quantization metadata for ``input`` / dispatched tokens."""

    local: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    format: Optional[str] = None
    block_size: Optional[int] = None


@dataclass
class DispatchOutput:
    """MLP-ready dispatch output."""

    tokens: torch.Tensor
    scales: Optional[QuantScales]
    num_tokens_per_expert: Union[torch.Tensor, List[int]]
    expert_offsets: Optional[torch.Tensor] = None
    layout: DispatchLayout = DispatchLayout.FLAT


@dataclass
class DispatchHandle:
    """Opaque dispatch metadata consumed by :meth:`MoECommunicator.combine`.

    The MLP should not need to inspect this handle. For HT, the reverse-dispatch
    metadata is transport-specific, so it is kept in the ``combine_meta`` bundle
    (different keys for intranode vs internode); :meth:`MoECommunicator.combine`
    is the only reader.
    """

    topk_ids: torch.Tensor
    weights: torch.Tensor
    num_tokens: int
    hidden_size: int
    num_experts: int
    num_local_experts: int
    local_expert_start: int
    layout: DispatchLayout = DispatchLayout.FLAT
    output_scales: Optional[QuantScales] = None
    # HT reverse-dispatch metadata (transport-tagged).
    is_internode: bool = False
    combine_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommOverlapConfig:
    """Optional overlap configuration for ``*_async`` dispatch/combine."""

    op: str  # "dispatch" or "combine"
    level: str = "op"  # "op" or "block"
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    signal: Optional[torch.Tensor] = None
    num_comm_sms: Optional[int] = None
    block_m: Optional[int] = None
    block_ready_value: Optional[int] = None


@dataclass
class MoECommunicatorConfig:
    """Configuration for the high-level MoE dispatch/combine API."""

    # Communication
    group: Optional[dist.ProcessGroup] = None
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
    mode: str = "ht"
    output_layout: Optional[Union[DispatchLayout, str]] = None

    # Quantization defaults
    input_dtype: Optional[torch.dtype] = None
    quant_format: Optional[str] = None

    # Transport / launch tuning (advanced)
    num_sms: int = 20
    num_rdma_qps_per_rank: int = 12
    expert_alignment: int = 1
    nvl_chunked_send: int = 8
    nvl_chunked_recv: int = 256
    rdma_chunked_send: int = 16
    rdma_chunked_recv: int = 128

    # Streams and overlap
    comm_stream: Optional[torch.cuda.Stream] = None
    enable_overlap: bool = False


def _normalize_output_layout(
    output_layout: Optional[Union[DispatchLayout, str]], mode: str
) -> DispatchLayout:
    if output_layout is not None:
        return DispatchLayout(output_layout)
    return DispatchLayout.FLAT if mode == "ht" else DispatchLayout.EXPERT_MAJOR


def _exclusive_cumsum(counts: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    """``[0, c0, c0+c1, ...]`` expert offsets for the FLAT grouped layout."""
    if isinstance(counts, torch.Tensor):
        flat = counts.to(torch.int64).flatten()
        zero = torch.zeros(1, dtype=torch.int64, device=flat.device)
        return torch.cat([zero, torch.cumsum(flat, dim=0)])
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + int(c))
    return torch.tensor(offsets, dtype=torch.int64)


class _CompletionRequest:
    """Request handle returned by the ``*_async`` methods.

    Wraps the in-flight :class:`EventHandle` (when ``async_finish`` is used) and
    the result the corresponding blocking call would have returned. ``wait()``
    synchronizes the comm event onto the current stream and returns the result.
    """

    def __init__(self, event, result):
        self._event = event
        self._result = result

    def wait(self):
        if self._event is not None:
            try:
                self._event.current_stream_wait()
            except AttributeError:
                torch.cuda.current_stream().synchronize()
        return self._result


class MoECommunicator:
    """High-level MoE communicator for HT (FLAT) dispatch/combine.

    The communicator owns the communication setup, scratch buffers, expert
    placement, and layout choice. These are configured once instead of being
    passed to every ``dispatch`` / ``combine`` call.
    """

    def __init__(self, config: Optional[MoECommunicatorConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either MoECommunicatorConfig or keyword arguments, not both")
        if config is None:
            # Accept ``comm=`` as an alias for ``group=`` for README parity.
            if "comm" in kwargs and "group" not in kwargs:
                kwargs["group"] = kwargs.pop("comm")
            config = MoECommunicatorConfig(**kwargs)

        if config.device is not None:
            torch.cuda.set_device(config.device)

        group = config.group
        if group is None:
            raise ValueError("MoECommunicator requires a torch.distributed ProcessGroup via group=")

        self.mode = config.mode.lower()
        if self.mode != "ht":
            raise NotImplementedError("MoECommunicator currently supports only mode='ht'")

        self.output_layout = _normalize_output_layout(config.output_layout, self.mode)
        if self.output_layout != DispatchLayout.FLAT:
            raise NotImplementedError("HT mode currently supports only DispatchLayout.FLAT")

        self.group = group
        self.rank: int = group.rank()
        self.world_size: int = group.size()
        self.local_rank: int = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)

        # ---- model shape / placement ----
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        if self.num_experts <= 0 or self.hidden_size <= 0 or self.topk <= 0 or self.max_tokens_per_rank <= 0:
            raise ValueError("num_experts, hidden_size, topk, and max_tokens_per_rank must be positive")

        self.num_local_experts = config.num_local_experts
        if self.num_local_experts is None:
            if self.num_experts % self.world_size != 0:
                raise ValueError("num_experts must be divisible by world_size for even contiguous placement")
            self.num_local_experts = self.num_experts // self.world_size
        if self.num_local_experts * self.world_size != self.num_experts:
            raise NotImplementedError("only even contiguous expert placement is currently supported")

        self.local_expert_start = config.local_expert_start
        if self.local_expert_start is None:
            self.local_expert_start = self.rank * self.num_local_experts

        if config.input_dtype not in (None, torch.bfloat16):
            raise NotImplementedError("HT dispatch currently supports BF16 input only")
        if config.quant_format is not None:
            raise NotImplementedError("HT quantized dispatch (scales) is not implemented yet")

        # ---- launch / tuning ----
        self.expert_alignment = config.expert_alignment
        self.num_sms = config.num_sms
        self.comm_stream = config.comm_stream
        self.enable_overlap = config.enable_overlap

        # ---- kernel launch config + scratch sizing ----
        # Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv). The C++
        # size hints return 0 RDMA bytes when world_size <= NUM_MAX_NVL_PEERS,
        # which is exactly the intranode/internode boundary — so we derive the
        # transport from the hint instead of hardcoding the NVL peer count.
        self._cfg = Config(
            self.num_sms,
            config.nvl_chunked_send,
            config.nvl_chunked_recv,
            config.rdma_chunked_send,
            config.rdma_chunked_recv,
        )
        hidden_bytes = self.hidden_size * torch.tensor([], dtype=torch.bfloat16).element_size()
        num_nvl_bytes = self._cfg.get_nvl_buffer_size_hint(hidden_bytes, self.world_size)
        num_rdma_bytes = self._cfg.get_rdma_buffer_size_hint(hidden_bytes, self.world_size)
        self._is_internode = num_rdma_bytes > 0

        self._buffer = Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._buffer.is_available()

    def is_internode(self) -> bool:
        return self._is_internode

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        scales: Optional[QuantScales] = None,
        *,
        output_buffer: Optional[torch.Tensor] = None,
        async_finish: bool = False,
    ) -> Tuple[DispatchOutput, DispatchHandle]:
        """Dispatch local tokens to their expert owners (HT / FLAT).

        ``output_buffer`` is accepted for API parity but ignored in HT: the FLAT
        row count is data-dependent, so dispatch allocates ``tokens`` itself.
        """
        self._validate_dispatch_inputs(input, topk_ids, weights, scales)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        # 1. Metadata phase — counts + per-token destination membership.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _layout_event,
        ) = self._buffer.get_dispatch_layout(topk_ids, self.num_experts, None, False, False)

        # 2. Payload dispatch -> FLAT recv_x grouped by local expert id.
        if self._is_internode:
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
                event,
            ) = self._buffer.internode_dispatch(
                input,
                None,
                topk_ids,
                weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                self.expert_alignment,
                self._cfg,
                None,
                async_finish,
                False,
            )
            combine_meta = {
                "recv_topk_weights": recv_topk_weights,
                "src_meta": recv_src_meta,
                "is_token_in_rank": is_token_in_rank,
                "recv_rdma_channel_prefix_matrix": recv_rdma_channel_prefix_matrix,
                "recv_rdma_rank_prefix_sum": recv_rdma_rank_prefix_sum,
                "recv_gbl_channel_prefix_matrix": recv_gbl_channel_prefix_matrix,
                "send_rdma_head": send_rdma_head,
                "send_nvl_head": send_nvl_head,
            }
        else:
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                event,
            ) = self._buffer.intranode_dispatch(
                input,
                None,
                topk_ids,
                weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                None,
                None,
                self.expert_alignment,
                self._cfg,
                None,
                async_finish,
                False,
            )
            combine_meta = {
                "recv_topk_weights": recv_topk_weights,
                "src_idx": recv_src_idx,
                "rank_prefix_matrix": rank_prefix_matrix,
                "recv_channel_prefix_matrix": recv_channel_prefix_matrix,
                "send_head": send_head,
            }

        # 3. MLP-facing output (FLAT, grouped by local expert).
        dispatch_out = DispatchOutput(
            tokens=recv_x,
            scales=None,  # BF16 path; FP8 scales plumbed in a later increment.
            num_tokens_per_expert=num_recv_tokens_per_expert_list,
            expert_offsets=_exclusive_cumsum(num_recv_tokens_per_expert_list),
            layout=DispatchLayout.FLAT,
        )

        # 4. Opaque reverse-dispatch handle.
        handle = DispatchHandle(
            topk_ids=topk_ids,
            weights=weights,
            num_tokens=int(input.size(0)),
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            local_expert_start=self.local_expert_start,
            layout=DispatchLayout.FLAT,
            output_scales=None,
            is_internode=self._is_internode,
            combine_meta=combine_meta,
        )
        handle._event = event  # type: ignore[attr-defined]
        return dispatch_out, handle

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        async_finish: bool = False,
    ) -> torch.Tensor:
        """Reduce expert outputs back to local token-major order (HT / FLAT)."""
        self._validate_combine_inputs(expert_output, handle)
        m = handle.combine_meta
        if handle.is_internode:
            combined_x, _combined_w, _event = self._buffer.internode_combine(
                expert_output,
                m["recv_topk_weights"],
                m["src_meta"],
                m["is_token_in_rank"],
                m["recv_rdma_channel_prefix_matrix"],
                m["recv_rdma_rank_prefix_sum"],
                m["recv_gbl_channel_prefix_matrix"],
                m["send_rdma_head"],
                m["send_nvl_head"],
                self._cfg,
                None,
                async_finish,
                False,
            )
        else:
            combined_x, _combined_w, _event = self._buffer.intranode_combine(
                expert_output,
                m["recv_topk_weights"],
                m["src_idx"],
                m["rank_prefix_matrix"],
                m["recv_channel_prefix_matrix"],
                m["send_head"],
                self._cfg,
                None,
                async_finish,
                False,
            )
        if out is not None:
            out.copy_(combined_x)
            return out
        return combined_x

    # ------------------------------------------------------------------
    # Optional async / overlap APIs
    # ------------------------------------------------------------------

    def dispatch_async(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        scales: Optional[QuantScales] = None,
        *,
        output_buffer: Optional[torch.Tensor] = None,
        overlap_config: Optional[CommOverlapConfig] = None,
    ) -> _CompletionRequest:
        result = self.dispatch(
            input, topk_ids, weights, scales, output_buffer=output_buffer, async_finish=True
        )
        _dispatch_out, handle = result
        return _CompletionRequest(getattr(handle, "_event", None), result)

    def combine_async(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        overlap_config: Optional[CommOverlapConfig] = None,
    ) -> _CompletionRequest:
        combined = self.combine(expert_output, handle, out=out, async_finish=True)
        return _CompletionRequest(None, combined)

    def create_overlap_config(
        self,
        op: str,
        *,
        handle: Optional[DispatchHandle] = None,
        level: str = "op",
    ) -> CommOverlapConfig:
        if op not in ("dispatch", "combine"):
            raise ValueError("op must be 'dispatch' or 'combine'")
        if level != "op":
            raise NotImplementedError("block-level overlap is not implemented yet")
        if op == "combine" and handle is None:
            raise ValueError("combine overlap config requires a DispatchHandle")
        return CommOverlapConfig(op=op, level=level, stream=self.comm_stream)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dispatch_inputs(self, input, topk_ids, weights, scales) -> None:
        if scales is not None and (scales.local is not None or scales.global_scale is not None):
            raise NotImplementedError("HT dispatch does not support quantized input scales yet")
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("HT dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} != configured {self.hidden_size}")
        if input.size(0) > self.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), self.topk):
            raise ValueError("topk_ids shape must be [input.size(0), topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")

    def _validate_combine_inputs(self, expert_output, handle) -> None:
        if not isinstance(handle, DispatchHandle):
            raise TypeError("handle must be a DispatchHandle returned by dispatch")
        if expert_output.dim() != 2 or not expert_output.is_contiguous():
            raise ValueError("expert_output must be a contiguous [total_recv_tokens, hidden] tensor")
        if expert_output.size(1) != self.hidden_size:
            raise ValueError(f"expert_output hidden size {expert_output.size(1)} != configured {self.hidden_size}")
        if handle.is_internode != self._is_internode:
            raise ValueError("handle transport does not match this communicator")
