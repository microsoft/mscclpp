# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Low-latency backend for the high-level MoE communicator."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode, MoERuntime
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    ExpertMajorDispatchHandle,
    ExpertMajorCombineContext,
    MoECommunicatorConfig,
    QuantConfig,
    TokenMajorCombineContext,
    TokenMajorDispatchHandle,
)
from .utils import cuda_stream_ptr, resolve_expert_placement


def _resolve_dispatch_data_type(quant: Optional[QuantConfig]) -> DispatchDataType:
    if quant is None:
        return DispatchDataType.BF16

    quant_format = quant.format
    if quant_format is not None and not isinstance(quant_format, DispatchDataType):
        raise TypeError("quant.format must be a DispatchDataType")
    if quant_format is None:
        raise ValueError("quant.format is required")
    if quant_format not in (DispatchDataType.FP8_E4M3, DispatchDataType.MXFP8_E4M3):
        raise ValueError("unsupported low-latency quantization format")
    if quant.block_scales is not None:
        raise ValueError("communicator quant config must not contain precomputed scales")
    return quant_format


def _dispatch_scale_block_size(data_type: DispatchDataType) -> int:
    if data_type == DispatchDataType.FP8_E4M3:
        return 128
    if data_type == DispatchDataType.MXFP8_E4M3:
        return 32
    return 0


def _dispatch_scale_dtype(data_type: DispatchDataType) -> torch.dtype:
    if data_type == DispatchDataType.FP8_E4M3:
        return torch.float32
    if data_type == DispatchDataType.MXFP8_E4M3:
        return torch.uint8
    raise ValueError("BF16 dispatch does not have block scales")


class LowLatencyRuntime:
    """Private low-level low-latency runtime wrapper (wraps ``_cpp.MoERuntime``)."""

    num_sms: int = 128

    def __init__(
        self,
        comm: Any,
        max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        initialize_token_major_padding: bool,
    ) -> None:
        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.cpp_runtime = MoERuntime(
            comm.communicator,
            max_tokens_per_rank,
            hidden,
            num_experts,
            num_topk,
            initialize_token_major_padding,
        )

    def is_available(self) -> bool:
        return self.cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        return self.cpp_runtime.is_internode_available()


class LowLatencyBackend:
    """Backend implementation for ``MoEMode.LOW_LATENCY``."""

    def __init__(self, config: MoECommunicatorConfig, output_layout: DispatchLayout) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=LOW_LATENCY requires an mscclpp.CommGroup via comm=")

        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.mode = MoEMode.LOW_LATENCY
        self.output_layout = output_layout

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.num_blocks = config.low_latency_num_blocks
        self.num_sms = self.num_blocks - 2
        self.combine_mode = config.low_latency_combine_mode
        self.token_major_init_padding = config.token_major_init_padding
        self.invalid_token_expert_id = (
            self.num_experts if config.invalid_token_expert_id is None else config.invalid_token_expert_id
        )
        self.enable_overlap = config.enable_overlap

        if self.output_layout not in (DispatchLayout.EXPERT_MAJOR, DispatchLayout.TOKEN_MAJOR):
            raise NotImplementedError("low-latency mode supports EXPERT_MAJOR and TOKEN_MAJOR output")
        if self.num_experts % self.world_size != 0:
            raise ValueError("low-latency mode requires num_experts divisible by world_size")
        if not self.world_size + 2 <= self.num_blocks <= 130:
            raise ValueError("low_latency_num_blocks must be between world_size + 2 and 130")
        if not isinstance(self.combine_mode, CombineMode):
            raise TypeError("low_latency_combine_mode must be a CombineMode")
        if not isinstance(self.token_major_init_padding, bool):
            raise TypeError("token_major_init_padding must be a bool")
        if type(self.invalid_token_expert_id) is not int:
            raise TypeError("invalid_token_expert_id must be an int or None")
        if not -(1 << 31) <= self.invalid_token_expert_id < (1 << 31):
            raise ValueError("invalid_token_expert_id must fit in int32")
        if 0 <= self.invalid_token_expert_id < self.num_experts:
            raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
        if self.token_major_init_padding and self.output_layout != DispatchLayout.TOKEN_MAJOR:
            raise ValueError("token_major_init_padding requires TOKEN_MAJOR output")
        if self.output_layout == DispatchLayout.TOKEN_MAJOR and self.combine_mode != CombineMode.RANK_LOCAL_REDUCE:
            raise ValueError("TOKEN_MAJOR output requires RANK_LOCAL_REDUCE combine")

        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )

        self.dispatch_data_type = _resolve_dispatch_data_type(config.quant)

        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_topk_ids: Optional[torch.Tensor] = None
        self._dispatch_weights: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._runtime = LowLatencyRuntime(
            comm,
            max_tokens_per_rank=self.max_tokens_per_rank,
            hidden=self.hidden_size,
            num_experts=self.num_experts,
            num_topk=self.topk,
            initialize_token_major_padding=self.token_major_init_padding,
        )
        self._is_internode = self._runtime.is_internode_available()

    def is_available(self) -> bool:
        return self._runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._runtime.is_internode_available()

    def is_internode(self) -> bool:
        return self._is_internode

    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        *,
        output_buffer: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
        previous_handle: Optional[DispatchHandle],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        del previous_handle
        self._validate_dispatch_inputs(input, topk_ids, weights, quant, output_buffer)

        out_buf, scales, src_info, recv_topk_ids, recv_weights, layout_range, count = self._get_dispatch_output_tensors(
            output_buffer
        )
        self._runtime.cpp_runtime.dispatch(
            input.data_ptr(),
            topk_ids.data_ptr(),
            0 if weights is None else weights.data_ptr(),
            out_buf.data_ptr(),
            0 if scales is None else scales.data_ptr(),
            src_info.data_ptr(),
            0 if recv_topk_ids is None else recv_topk_ids.data_ptr(),
            0 if recv_weights is None else recv_weights.data_ptr(),
            0 if layout_range is None else layout_range.data_ptr(),
            count.data_ptr(),
            input.size(0),
            self.hidden_size,
            self.topk,
            self.max_tokens_per_rank,
            self.num_experts,
            self.invalid_token_expert_id,
            self.output_layout,
            self.dispatch_data_type,
            self.num_blocks,
            cuda_stream_ptr(stream),
        )
        output_quant = (
            None
            if scales is None
            else QuantConfig(
                format=self.dispatch_data_type,
                block_scales=scales,
            )
        )
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            layout_info = DispatchLayoutInfo(kind=self.output_layout, num_tokens_per_expert=count)
        elif self.output_layout == DispatchLayout.TOKEN_MAJOR:
            assert layout_range is not None
            layout_info = DispatchLayoutInfo(
                kind=self.output_layout,
                num_tokens_per_rank=count,
                offsets=layout_range,
            )
        else:
            raise ValueError(f"unsupported low-latency output layout: {self.output_layout}")
        output_info = DispatchOutputInfo(layout=layout_info, quant=output_quant)
        dispatch_out = DispatchOutput(
            tokens=out_buf,
            quant=output_info.quant,
            layout=output_info.layout,
            topk_ids=recv_topk_ids,
            weights=recv_weights,
        )
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            assert layout_range is not None
            handle: DispatchHandle = ExpertMajorDispatchHandle(
                output_info=output_info,
                combine_context=ExpertMajorCombineContext(
                    topk_ids=topk_ids,
                    weights=weights,
                    num_experts=self.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=self.hidden_size,
                    src_info=src_info,
                    layout_range=layout_range,
                    num_max_dispatch_tokens_per_rank=self.max_tokens_per_rank,
                ),
            )
        elif self.output_layout == DispatchLayout.TOKEN_MAJOR:
            assert layout_range is not None
            handle = TokenMajorDispatchHandle(
                output_info=output_info,
                combine_context=TokenMajorCombineContext(
                    topk_ids=topk_ids,
                    num_experts=self.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=self.hidden_size,
                    source_token_ids=src_info,
                    num_tokens_per_rank=count,
                    rank_offsets=layout_range,
                    num_max_dispatch_tokens_per_rank=self.max_tokens_per_rank,
                ),
            )
        else:
            raise ValueError(f"unsupported low-latency output layout: {self.output_layout}")
        return dispatch_out, handle

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        self._validate_combine_inputs(expert_output, handle, out)
        if isinstance(handle, ExpertMajorDispatchHandle):
            context = handle.combine_context
            topk_weights = context.weights
            src_info = context.src_info
            layout_range = context.layout_range
        elif isinstance(handle, TokenMajorDispatchHandle):
            context = handle.combine_context
            topk_weights = None
            src_info = context.source_token_ids
            layout_range = context.rank_offsets
        else:
            raise ValueError("DispatchHandle does not contain low-latency combine context")
        if out is None:
            out = torch.empty((context.num_tokens, self.hidden_size), dtype=torch.bfloat16, device=expert_output.device)
        self._runtime.cpp_runtime.combine(
            expert_output.data_ptr(),
            context.topk_ids.data_ptr(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            src_info.data_ptr(),
            0 if layout_range is None else layout_range.data_ptr(),
            out.data_ptr(),
            context.num_tokens,
            self.hidden_size,
            self.topk,
            context.num_max_dispatch_tokens_per_rank,
            context.num_experts,
            self.output_layout,
            self.dispatch_data_type,
            self.combine_mode,
            self.num_blocks - 2,
            cuda_stream_ptr(stream),
        )
        return out

    def _get_dispatch_output_tensors(self, output_buffer: torch.Tensor):
        device = output_buffer.device
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self._dispatch_src_info is None or self._dispatch_src_info.device != device:
            self._dispatch_scales = None
            self._dispatch_topk_ids = None
            self._dispatch_weights = None
            if self.output_layout == DispatchLayout.EXPERT_MAJOR:
                self._dispatch_src_info = torch.empty(
                    (self.num_local_experts, slots_per_expert), dtype=torch.int32, device=device
                )
                self._dispatch_layout_range = torch.empty(
                    (self.num_local_experts, self.world_size), dtype=torch.int64, device=device
                )
                self._dispatch_count = torch.empty((self.num_local_experts,), dtype=torch.int32, device=device)
                scale_block_size = _dispatch_scale_block_size(self.dispatch_data_type)
                if scale_block_size:
                    num_scales = self.hidden_size // scale_block_size
                    scale_storage = torch.empty(
                        (self.num_local_experts, num_scales, slots_per_expert),
                        dtype=_dispatch_scale_dtype(self.dispatch_data_type),
                        device=device,
                    )
                    self._dispatch_scales = scale_storage.transpose(1, 2)
            elif self.output_layout == DispatchLayout.TOKEN_MAJOR:
                token_capacity = self.world_size * self.max_tokens_per_rank
                self._dispatch_src_info = torch.empty((token_capacity,), dtype=torch.int32, device=device)
                self._dispatch_topk_ids = torch.empty((token_capacity, self.topk), dtype=torch.int32, device=device)
                self._dispatch_weights = torch.empty((token_capacity, self.topk), dtype=torch.float32, device=device)
                self._dispatch_layout_range = torch.empty((self.world_size + 1,), dtype=torch.int64, device=device)
                self._dispatch_count = torch.empty((self.world_size,), dtype=torch.int32, device=device)
                scale_block_size = _dispatch_scale_block_size(self.dispatch_data_type)
                if scale_block_size:
                    self._dispatch_scales = torch.empty(
                        (token_capacity, self.hidden_size // scale_block_size),
                        dtype=_dispatch_scale_dtype(self.dispatch_data_type),
                        device=device,
                    )
            else:
                raise ValueError(f"unsupported low-latency output layout: {self.output_layout}")
        assert self._dispatch_src_info is not None
        assert self._dispatch_count is not None
        return (
            output_buffer,
            self._dispatch_scales,
            self._dispatch_src_info,
            self._dispatch_topk_ids,
            self._dispatch_weights,
            self._dispatch_layout_range,
            self._dispatch_count,
        )

    def _validate_dispatch_inputs(self, input, topk_ids, weights, quant, output_buffer) -> None:
        if output_buffer is None:
            raise ValueError("output_buffer is required for low-latency dispatch")
        if quant is not None:
            raise NotImplementedError(
                "per-call input quant metadata is not supported; configure dispatch output quantization on the communicator"
            )
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden_size] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("low-latency dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} does not match configured {self.hidden_size}")
        if input.size(0) > self.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), self.topk):
            raise ValueError("topk_ids shape must match [input.size(0), configured topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        elif self.output_layout == DispatchLayout.TOKEN_MAJOR:
            expected_shape = (self.world_size * self.max_tokens_per_rank, self.hidden_size)
        else:
            raise ValueError(f"unsupported low-latency output layout: {self.output_layout}")
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {self.output_layout} tensor")
        expected_dtype = torch.bfloat16 if self.dispatch_data_type == DispatchDataType.BF16 else torch.float8_e4m3fn
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine_inputs(self, expert_output, handle, out) -> None:
        if not isinstance(handle, (ExpertMajorDispatchHandle, TokenMajorDispatchHandle)):
            raise ValueError("DispatchHandle does not contain low-latency combine context")
        context = handle.combine_context
        if context.num_experts != self.num_experts or context.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        if handle.output_info.layout.kind != self.output_layout:
            raise ValueError("DispatchHandle output layout does not match this MoECommunicator")
        output_quant = handle.output_info.quant
        handle_data_type = DispatchDataType.BF16 if output_quant is None else output_quant.format
        if handle_data_type != self.dispatch_data_type:
            raise ValueError("DispatchHandle quantization does not match this MoECommunicator configuration")
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if handle.output_info.layout.kind == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        elif handle.output_info.layout.kind == DispatchLayout.TOKEN_MAJOR:
            expected_shape = (self.world_size * self.max_tokens_per_rank, self.hidden_size)
            if handle.output_info.layout.offsets is None:
                raise ValueError("TOKEN_MAJOR DispatchHandle is missing compact rank offsets")
        else:
            raise ValueError(f"unsupported low-latency output layout: {handle.output_info.layout.kind}")
        if expert_output.dim() != len(expected_shape) or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous layout")
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        if expert_output.dtype != torch.bfloat16:
            raise ValueError("expert_output must be BF16")
        if out is not None:
            expected_out_shape = (context.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")
