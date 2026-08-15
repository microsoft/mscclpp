# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Latency-optimized backend for the high-level MoE communicator."""

from __future__ import annotations

from typing import Optional

import torch

from mscclpp.ep._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode
from mscclpp.ep.backend import Backend
from mscclpp.ep.runtime import Runtime
from mscclpp.ep.types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    QuantConfig,
    _ExpertMajorCombineContext,
    _RankMajorCombineContext,
)
from mscclpp.ep.utils import (
    DevicePointerArray,
    cuda_stream_ptr,
    dispatch_scale_block_size,
    dispatch_scale_dtype,
    resolve_expert_placement,
    resolve_dispatch_data_type,
    tensor_from_pointer,
)


class LatencyBackend(Backend):
    """Latency-mode backend."""

    def __init__(self, config: MoECommunicatorConfig) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=LATENCY requires an mscclpp.CommGroup via comm=")
        output_layout = config.output_layout
        if output_layout is None:
            output_layout = DispatchLayout.EXPERT_MAJOR
        runtime = Runtime(
            comm,
            MoEMode.LATENCY,
            max_tokens_per_rank=config.max_tokens_per_rank,
            hidden=config.hidden_size,
            num_experts=config.num_experts,
            num_topk=config.topk,
            output_layout=output_layout,
        )
        super().__init__(runtime)

        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.mode = MoEMode.LATENCY
        self.output_layout = output_layout

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.num_blocks = config.latency_num_blocks
        self.num_sms = self.num_blocks - 2
        self.combine_mode = config.combine_mode
        self.invalid_token_expert_id = (
            self.num_experts if config.invalid_token_expert_id is None else config.invalid_token_expert_id
        )
        self.enable_overlap = config.enable_overlap

        if self.output_layout not in (
            DispatchLayout.EXPERT_MAJOR,
            DispatchLayout.RANK_MAJOR,
        ):
            raise NotImplementedError("unsupported latency output layout")
        if self.num_experts % self.world_size != 0:
            raise ValueError("latency mode requires num_experts divisible by world_size")
        if not self.world_size + 2 <= self.num_blocks <= 130:
            raise ValueError("latency_num_blocks must be between world_size + 2 and 130")
        if not isinstance(self.combine_mode, CombineMode):
            raise TypeError("combine_mode must be a CombineMode")
        if type(self.invalid_token_expert_id) is not int:
            raise TypeError("invalid_token_expert_id must be an int or None")
        if not -(1 << 31) <= self.invalid_token_expert_id < (1 << 31):
            raise ValueError("invalid_token_expert_id must fit in int32")
        if 0 <= self.invalid_token_expert_id < self.num_experts:
            raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
        if self.output_layout == DispatchLayout.RANK_MAJOR:
            if self.combine_mode != CombineMode.RANK_LOCAL_REDUCE:
                raise ValueError("RANK_MAJOR output requires RANK_LOCAL_REDUCE combine")
            if self.enable_overlap:
                raise NotImplementedError("RANK_MAJOR output does not support overlapping calls yet")

        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )

        self.dispatch_data_type = resolve_dispatch_data_type(config.quant)
        if self.output_layout == DispatchLayout.RANK_MAJOR and self.dispatch_data_type != DispatchDataType.BF16:
            raise NotImplementedError("RANK_MAJOR output currently supports BF16 dispatch only")

        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_topk_ids: Optional[torch.Tensor] = None
        self._dispatch_weights: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._is_internode = self.runtime.is_internode_available()
        self._dispatch_output_owner: Optional[DevicePointerArray] = None
        self._combine_input_owner: Optional[DevicePointerArray] = None
        self._output_topk_ids_owner: Optional[DevicePointerArray] = None
        self._output_topk_weights_owner: Optional[DevicePointerArray] = None
        self._output_topk_ids: Optional[torch.Tensor] = None
        self._output_topk_weights: Optional[torch.Tensor] = None
        self.combine_input_buffer: Optional[torch.Tensor] = None
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            dispatch_shape = (
                self.num_local_experts,
                self.world_size * self.max_tokens_per_rank,
                self.hidden_size,
            )
        else:
            dispatch_shape = (self.world_size * self.max_tokens_per_rank, self.hidden_size)

        if self.dispatch_data_type == DispatchDataType.BF16:
            self._dispatch_output_owner, self.dispatch_output_buffer = tensor_from_pointer(
                self.runtime.cpp_runtime.dispatch_output_buffer_ptr(),
                dispatch_shape,
                torch.bfloat16,
                self.device,
                self.runtime,
            )
        else:
            self._dispatch_output_owner, self.dispatch_output_buffer = tensor_from_pointer(
                self.runtime.cpp_runtime.dispatch_output_buffer_ptr(),
                dispatch_shape,
                torch.float8_e4m3fn,
                self.device,
                self.runtime,
            )

        if self.output_layout == DispatchLayout.RANK_MAJOR:
            metadata_shape = (self.world_size * self.max_tokens_per_rank, self.topk)
            (
                self._output_topk_ids_owner,
                self._output_topk_ids,
            ) = tensor_from_pointer(
                self.runtime.cpp_runtime.output_topk_ids_buffer_ptr(),
                metadata_shape,
                torch.int32,
                self.device,
                self.runtime,
            )
            (
                self._output_topk_weights_owner,
                self._output_topk_weights,
            ) = tensor_from_pointer(
                self.runtime.cpp_runtime.output_topk_weights_buffer_ptr(),
                metadata_shape,
                torch.float32,
                self.device,
                self.runtime,
            )
            (
                self._combine_input_owner,
                self.combine_input_buffer,
            ) = tensor_from_pointer(
                self.runtime.cpp_runtime.combine_input_buffer_ptr(),
                dispatch_shape,
                torch.bfloat16,
                self.device,
                self.runtime,
            )

    def _resolve_runtime_max_tokens_per_rank(self, runtime_max_tokens_per_rank: Optional[int]) -> int:
        resolved = self.max_tokens_per_rank if runtime_max_tokens_per_rank is None else runtime_max_tokens_per_rank
        if type(resolved) is not int or not 0 < resolved <= self.max_tokens_per_rank:
            raise ValueError("runtime_max_tokens_per_rank must be positive and not exceed max_tokens_per_rank")
        if self.output_layout != DispatchLayout.RANK_MAJOR and resolved != self.max_tokens_per_rank:
            raise ValueError("runtime_max_tokens_per_rank is only supported by rank-major dispatch")
        return resolved

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
        runtime_max_tokens_per_rank: Optional[int],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        del previous_handle
        active_capacity = self._resolve_runtime_max_tokens_per_rank(runtime_max_tokens_per_rank)
        if output_buffer is None:
            output_buffer = self.dispatch_output_buffer
        self._validate_dispatch_inputs(input, topk_ids, weights, quant, output_buffer, active_capacity)

        out_buf, scales, src_info, recv_topk_ids, recv_weights, layout_range, count = self._get_dispatch_output_tensors(
            output_buffer
        )
        self.runtime.cpp_runtime.latency_dispatch(
            input.data_ptr(),
            topk_ids.data_ptr(),
            0 if weights is None else weights.data_ptr(),
            out_buf.data_ptr(),
            0 if scales is None else scales.data_ptr(),
            0 if src_info is None else src_info.data_ptr(),
            0 if recv_topk_ids is None else recv_topk_ids.data_ptr(),
            0 if recv_weights is None else recv_weights.data_ptr(),
            0 if layout_range is None else layout_range.data_ptr(),
            count.data_ptr(),
            input.size(0),
            self.hidden_size,
            self.topk,
            active_capacity,
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
        elif self.output_layout == DispatchLayout.RANK_MAJOR:
            layout_info = DispatchLayoutInfo(
                kind=self.output_layout,
                num_tokens_per_rank=count,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {self.output_layout}")
        output_info = DispatchOutputInfo(layout=layout_info, quant=output_quant)
        dispatch_out = DispatchOutput(
            tokens=out_buf,
            quant=output_info.quant,
            layout=output_info.layout,
            topk_ids=recv_topk_ids,
            weights=recv_weights,
            combine_input_buffer=(
                self.combine_input_buffer if self.output_layout == DispatchLayout.RANK_MAJOR else None
            ),
        )
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            assert layout_range is not None
            assert src_info is not None
            handle = DispatchHandle(
                output_info=output_info,
                _context=_ExpertMajorCombineContext(
                    topk_ids=topk_ids,
                    weights=weights,
                    num_experts=self.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=self.hidden_size,
                    src_info=src_info,
                    layout_range=layout_range,
                ),
            )
        elif self.output_layout == DispatchLayout.RANK_MAJOR:
            handle = DispatchHandle(
                output_info=output_info,
                _context=_RankMajorCombineContext(
                    topk_ids=topk_ids,
                    num_experts=self.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=self.hidden_size,
                    max_tokens_per_rank=active_capacity,
                ),
            )
        else:
            raise ValueError(f"unsupported latency output layout: {self.output_layout}")
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
        context = handle._context
        if isinstance(context, _ExpertMajorCombineContext):
            topk_weights = context.weights
            src_info = context.src_info
            layout_range = context.layout_range
            active_capacity = self.max_tokens_per_rank
        elif isinstance(context, _RankMajorCombineContext):
            active_capacity = context.max_tokens_per_rank
            topk_weights = None
            src_info = None
            layout_range = None
        else:
            raise ValueError("DispatchHandle does not contain latency combine context")
        if out is None:
            out = torch.empty(
                (context.num_tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device=expert_output.device,
            )
        self.runtime.cpp_runtime.latency_combine(
            expert_output.data_ptr(),
            context.topk_ids.data_ptr(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            0 if src_info is None else src_info.data_ptr(),
            0 if layout_range is None else layout_range.data_ptr(),
            out.data_ptr(),
            context.num_tokens,
            self.hidden_size,
            self.topk,
            active_capacity,
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
        if self._dispatch_count is None or self._dispatch_count.device != device:
            self._dispatch_scales = None
            self._dispatch_topk_ids = None
            self._dispatch_weights = None
            if self.output_layout == DispatchLayout.EXPERT_MAJOR:
                self._dispatch_src_info = torch.empty(
                    (self.num_local_experts, slots_per_expert),
                    dtype=torch.int32,
                    device=device,
                )
                self._dispatch_layout_range = torch.empty(
                    (self.num_local_experts, self.world_size),
                    dtype=torch.int64,
                    device=device,
                )
                self._dispatch_count = torch.empty((self.num_local_experts,), dtype=torch.int32, device=device)
                scale_block_size = dispatch_scale_block_size(self.dispatch_data_type)
                if scale_block_size:
                    num_scales = self.hidden_size // scale_block_size
                    scale_storage = torch.empty(
                        (self.num_local_experts, num_scales, slots_per_expert),
                        dtype=dispatch_scale_dtype(self.dispatch_data_type),
                        device=device,
                    )
                    self._dispatch_scales = scale_storage.transpose(1, 2)
            elif self.output_layout == DispatchLayout.RANK_MAJOR:
                self._dispatch_src_info = None
                assert self._output_topk_ids is not None
                assert self._output_topk_weights is not None
                self._dispatch_topk_ids = self._output_topk_ids
                self._dispatch_weights = self._output_topk_weights
                self._dispatch_layout_range = None
                self._dispatch_count = torch.empty((self.world_size,), dtype=torch.int32, device=device)
            else:
                raise ValueError(f"unsupported latency output layout: {self.output_layout}")
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

    def _validate_dispatch_inputs(self, input, topk_ids, weights, quant, output_buffer, active_capacity: int) -> None:
        if quant is not None:
            raise NotImplementedError(
                "per-call input quant metadata is not supported; configure dispatch output quantization on the communicator"
            )
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden_size] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("latency dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} does not match configured {self.hidden_size}")
        if input.size(0) > active_capacity:
            raise ValueError("input token count exceeds runtime_max_tokens_per_rank")
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
            expected_shape = (
                self.num_local_experts,
                slots_per_expert,
                self.hidden_size,
            )
        elif self.output_layout == DispatchLayout.RANK_MAJOR:
            expected_shape = (
                self.world_size * self.max_tokens_per_rank,
                self.hidden_size,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {self.output_layout}")
        if self.output_layout == DispatchLayout.RANK_MAJOR:
            if output_buffer.data_ptr() != self.dispatch_output_buffer.data_ptr():
                raise ValueError("RANK_MAJOR output uses the runtime-owned dispatch output buffer")
            return
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {self.output_layout} tensor")
        expected_dtype = torch.bfloat16 if self.dispatch_data_type == DispatchDataType.BF16 else torch.float8_e4m3fn
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine_inputs(self, expert_output, handle, out) -> None:
        if not isinstance(handle, DispatchHandle) or not isinstance(
            handle._context, (_ExpertMajorCombineContext, _RankMajorCombineContext)
        ):
            raise ValueError("DispatchHandle does not contain latency combine context")
        context = handle._context
        if context.num_experts != self.num_experts or context.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        if handle.output_info.layout.kind != self.output_layout:
            raise ValueError("DispatchHandle output layout does not match this MoECommunicator")
        output_quant = handle.output_info.quant
        handle_data_type = DispatchDataType.BF16 if output_quant is None else output_quant.format
        if handle_data_type != self.dispatch_data_type:
            raise ValueError("DispatchHandle quantization does not match this MoECommunicator configuration")
        active_capacity = (
            context.max_tokens_per_rank if isinstance(context, _RankMajorCombineContext) else self.max_tokens_per_rank
        )
        slots_per_expert = self.world_size * active_capacity
        if handle.output_info.layout.kind == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (
                self.num_local_experts,
                slots_per_expert,
                self.hidden_size,
            )
        elif handle.output_info.layout.kind == DispatchLayout.RANK_MAJOR:
            expected_shape = (
                self.world_size * active_capacity,
                self.hidden_size,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {handle.output_info.layout.kind}")
        if expert_output.dim() != len(expected_shape) or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous layout")
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        if expert_output.dtype != torch.bfloat16:
            raise ValueError("expert_output must be BF16")
        if handle.output_info.layout.kind == DispatchLayout.RANK_MAJOR:
            assert self.combine_input_buffer is not None
            if expert_output.data_ptr() != self.combine_input_buffer.data_ptr():
                raise ValueError("RANK_MAJOR combine requires the runtime-owned combine input buffer")
        if out is not None:
            expected_out_shape = (context.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")
