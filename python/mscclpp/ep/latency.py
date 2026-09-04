# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Latency-mode context."""

from __future__ import annotations

from typing import Optional

import torch

from mscclpp.ep._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode, create_moe_runtime
from mscclpp.ep.context import Context
from mscclpp.ep.runtime import Runtime, requires_initialized
from mscclpp.ep.types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    QuantConfig,
    _ExpertMajorCombineContext,
    _RankMajorCombineContext,
    _TokenMajorCombineContext,
)
from mscclpp.ep.utils import (
    DevicePointerArray,
    combine_tensor_dtype,
    cuda_stream_ptr,
    dispatch_scale_block_size,
    dispatch_scale_dtype,
    dispatch_tensor_dtype,
    resolve_expert_placement,
    resolve_dispatch_data_type,
    resolve_num_blocks,
    tensor_from_pointer,
)


class LatencyContext(Context):
    """Latency-mode context."""

    def __init__(self, config: MoECommunicatorConfig) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=LATENCY requires an mscclpp.CommGroup via comm=")
        output_layout = config.output_layout
        if output_layout is None:
            output_layout = DispatchLayout.EXPERT_MAJOR
        dispatch_blocks, combine_blocks = resolve_num_blocks(
            config.num_blocks,
            default=(130, 128),
            scalar_combine_offset=-2,
        )

        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.initialized = False
        self.mode = MoEMode.LATENCY
        self.output_layout = output_layout

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.dispatch_blocks = dispatch_blocks
        self.combine_blocks = combine_blocks
        self.combine_mode = config.combine_mode
        self.invalid_token_expert_id = (
            self.num_experts if config.invalid_token_expert_id is None else config.invalid_token_expert_id
        )
        self.enable_overlap = config.enable_overlap

        if self.output_layout not in (
            DispatchLayout.EXPERT_MAJOR,
            DispatchLayout.RANK_MAJOR,
            DispatchLayout.TOKEN_MAJOR,
        ):
            raise NotImplementedError("unsupported latency output layout")
        if self.num_experts % self.world_size != 0:
            raise ValueError("latency mode requires num_experts divisible by world_size")
        if not self.world_size + 2 <= dispatch_blocks <= 130:
            raise ValueError("dispatch block count must be between world_size + 2 and 130 in latency mode")
        if not 0 < combine_blocks <= 128:
            raise ValueError("combine block count must be between 1 and 128 in latency mode")
        if not isinstance(self.combine_mode, CombineMode):
            raise TypeError("combine_mode must be a CombineMode")
        if type(self.invalid_token_expert_id) is not int:
            raise TypeError("invalid_token_expert_id must be an int or None")
        if not -(1 << 31) <= self.invalid_token_expert_id < (1 << 31):
            raise ValueError("invalid_token_expert_id must fit in int32")
        if 0 <= self.invalid_token_expert_id < self.num_experts:
            raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
        if self.output_layout == DispatchLayout.RANK_MAJOR:
            if self.combine_mode not in (
                CombineMode.RANK_LOCAL_REDUCE,
                CombineMode.DIRECT_SEND,
            ):
                raise ValueError("RANK_MAJOR output requires a supported combine mode")
            if self.enable_overlap:
                raise NotImplementedError("RANK_MAJOR output does not support overlapping calls yet")
        if self.output_layout == DispatchLayout.TOKEN_MAJOR:
            if self.combine_mode != CombineMode.RANK_LOCAL_REDUCE:
                raise ValueError("TOKEN_MAJOR output requires RANK_LOCAL_REDUCE combine")
            if self.enable_overlap:
                raise NotImplementedError("TOKEN_MAJOR output does not support overlapping calls yet")

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
        if self.output_layout == DispatchLayout.TOKEN_MAJOR and self.dispatch_data_type != DispatchDataType.BF16:
            raise NotImplementedError("TOKEN_MAJOR output currently supports BF16 dispatch only")

        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_topk_ids: Optional[torch.Tensor] = None
        self._dispatch_weights: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._dispatch_output_owner: Optional[DevicePointerArray] = None
        self._combine_input_owner: Optional[DevicePointerArray] = None
        self._output_topk_ids_owner: Optional[DevicePointerArray] = None
        self._output_topk_weights_owner: Optional[DevicePointerArray] = None
        self._output_topk_ids: Optional[torch.Tensor] = None
        self._output_topk_weights: Optional[torch.Tensor] = None
        self.dispatch_output_buffer: Optional[torch.Tensor] = None
        self.combine_input_buffer: Optional[torch.Tensor] = None


class LatencyRuntime(Runtime):
    """Latency-optimized runtime."""

    context: LatencyContext

    def __init__(self, context: LatencyContext) -> None:
        cpp_runtime = create_moe_runtime(
            context.comm.communicator,
            context.mode,
            max_tokens_per_rank=context.max_tokens_per_rank,
            hidden=context.hidden_size,
            num_experts=context.num_experts,
            num_topk=context.topk,
            output_layout=context.output_layout,
            combine_mode=context.combine_mode,
        )
        super().__init__(context, cpp_runtime)

    @requires_initialized
    def get_dispatch_output_buffer(self) -> torch.Tensor:
        """Return the stable runtime-owned dispatch output buffer."""
        if self.context.dispatch_output_buffer is None:
            raise RuntimeError("latency dispatch output buffer is unavailable")
        return self.context.dispatch_output_buffer

    @requires_initialized
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
        mode_context = self.context
        del previous_handle
        active_capacity = self._resolve_capacity(runtime_max_tokens_per_rank)
        if output_buffer is None:
            assert mode_context.dispatch_output_buffer is not None
            output_buffer = mode_context.dispatch_output_buffer
        self._validate_dispatch(input, topk_ids, weights, quant, output_buffer, active_capacity)

        out_buf, scales, src_info, recv_topk_ids, recv_weights, layout_range, count = self._dispatch_outputs(
            output_buffer
        )
        self.cpp_runtime.dispatch(
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
            mode_context.hidden_size,
            mode_context.topk,
            active_capacity,
            mode_context.num_experts,
            mode_context.invalid_token_expert_id,
            mode_context.output_layout,
            mode_context.dispatch_data_type,
            mode_context.dispatch_blocks,
            cuda_stream_ptr(stream),
        )
        output_quant = (
            None
            if scales is None
            else QuantConfig(
                format=mode_context.dispatch_data_type,
                block_scales=scales,
            )
        )
        if mode_context.output_layout == DispatchLayout.EXPERT_MAJOR:
            layout_info = DispatchLayoutInfo(kind=mode_context.output_layout, num_tokens_per_expert=count)
        elif mode_context.output_layout == DispatchLayout.RANK_MAJOR:
            layout_info = DispatchLayoutInfo(
                kind=mode_context.output_layout,
                num_tokens_per_rank=count,
            )
        elif mode_context.output_layout == DispatchLayout.TOKEN_MAJOR:
            layout_info = DispatchLayoutInfo(
                kind=mode_context.output_layout,
                num_tokens_per_rank=count,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {mode_context.output_layout}")
        output_info = DispatchOutputInfo(layout=layout_info, quant=output_quant)
        combine_input_buffer = mode_context.combine_input_buffer
        if mode_context.output_layout in (DispatchLayout.RANK_MAJOR, DispatchLayout.TOKEN_MAJOR):
            assert combine_input_buffer is not None
            rows = mode_context.world_size * active_capacity
            if mode_context.output_layout == DispatchLayout.TOKEN_MAJOR:
                rows *= mode_context.topk
        dispatch_out = DispatchOutput(
            tokens=out_buf,
            quant=output_info.quant,
            layout=output_info.layout,
            topk_ids=recv_topk_ids,
            weights=recv_weights,
            combine_input_buffer=(
                combine_input_buffer[:rows]
                if mode_context.output_layout in (DispatchLayout.RANK_MAJOR, DispatchLayout.TOKEN_MAJOR)
                else None
            ),
        )
        if mode_context.output_layout == DispatchLayout.EXPERT_MAJOR:
            assert layout_range is not None
            assert src_info is not None
            handle = DispatchHandle(
                output_info=output_info,
                _context=_ExpertMajorCombineContext(
                    topk_ids=topk_ids,
                    weights=weights,
                    num_experts=mode_context.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=mode_context.hidden_size,
                    src_info=src_info,
                    layout_range=layout_range,
                ),
            )
        elif mode_context.output_layout == DispatchLayout.RANK_MAJOR:
            handle = DispatchHandle(
                output_info=output_info,
                _context=_RankMajorCombineContext(
                    topk_ids=topk_ids,
                    num_experts=mode_context.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=mode_context.hidden_size,
                    max_tokens_per_rank=active_capacity,
                ),
            )
        elif mode_context.output_layout == DispatchLayout.TOKEN_MAJOR:
            handle = DispatchHandle(
                output_info=output_info,
                _context=_TokenMajorCombineContext(
                    topk_ids=topk_ids,
                    weights=weights,
                    num_experts=mode_context.num_experts,
                    num_tokens=input.size(0),
                    hidden_size=mode_context.hidden_size,
                    max_tokens_per_rank=active_capacity,
                ),
            )
        else:
            raise ValueError(f"unsupported latency output layout: {mode_context.output_layout}")
        return dispatch_out, handle

    @requires_initialized
    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        mode_context = self.context
        self._validate_combine(expert_output, handle, out)
        context = handle._context
        if isinstance(context, _ExpertMajorCombineContext):
            topk_weights = context.weights
            src_info = context.src_info
            layout_range = context.layout_range
            active_capacity = mode_context.max_tokens_per_rank
        elif isinstance(context, _RankMajorCombineContext):
            active_capacity = context.max_tokens_per_rank
            topk_weights = None
            src_info = None
            layout_range = None
        elif isinstance(context, _TokenMajorCombineContext):
            active_capacity = context.max_tokens_per_rank
            topk_weights = context.weights
            src_info = None
            layout_range = None
        else:
            raise ValueError("DispatchHandle does not contain latency combine context")
        if out is None:
            out = torch.empty(
                (context.num_tokens, mode_context.hidden_size),
                dtype=combine_tensor_dtype(mode_context.dispatch_data_type),
                device=expert_output.device,
            )
        self.cpp_runtime.combine(
            expert_output.data_ptr(),
            context.topk_ids.data_ptr(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            0 if src_info is None else src_info.data_ptr(),
            0 if layout_range is None else layout_range.data_ptr(),
            out.data_ptr(),
            context.num_tokens,
            mode_context.hidden_size,
            mode_context.topk,
            active_capacity,
            context.num_experts,
            mode_context.output_layout,
            mode_context.dispatch_data_type,
            mode_context.combine_mode,
            mode_context.combine_blocks,
            cuda_stream_ptr(stream),
        )
        return out

    def initialize(self) -> None:
        """Collectively initialize and bind latency communication resources."""
        super().initialize()
        if self.context.dispatch_output_buffer is None:
            self._bind_buffers()

    def _bind_buffers(self) -> None:
        """Create tensor views over runtime-owned latency buffers."""
        context = self.context
        if context.dispatch_output_buffer is not None:
            return
        if context.output_layout == DispatchLayout.EXPERT_MAJOR:
            dispatch_shape = (
                context.num_local_experts,
                context.world_size * context.max_tokens_per_rank,
                context.hidden_size,
            )
        elif context.output_layout == DispatchLayout.TOKEN_MAJOR:
            dispatch_shape = (
                context.world_size * context.max_tokens_per_rank * context.topk,
                context.hidden_size,
            )
        else:
            dispatch_shape = (context.world_size * context.max_tokens_per_rank, context.hidden_size)

        dispatch_dtype = dispatch_tensor_dtype(context.dispatch_data_type)
        context._dispatch_output_owner, context.dispatch_output_buffer = tensor_from_pointer(
            self.cpp_runtime.dispatch_output_buffer_ptr(),
            dispatch_shape,
            dispatch_dtype,
            context.device,
            self.cpp_runtime,
        )

        if context.output_layout not in (DispatchLayout.RANK_MAJOR, DispatchLayout.TOKEN_MAJOR):
            return

        metadata_shape = (context.world_size * context.max_tokens_per_rank, context.topk)
        context._output_topk_ids_owner, context._output_topk_ids = tensor_from_pointer(
            self.cpp_runtime.output_topk_ids_buffer_ptr(),
            metadata_shape,
            torch.int32,
            context.device,
            self.cpp_runtime,
        )
        context._output_topk_weights_owner, context._output_topk_weights = tensor_from_pointer(
            self.cpp_runtime.output_topk_weights_buffer_ptr(),
            metadata_shape,
            torch.float32,
            context.device,
            self.cpp_runtime,
        )
        if context.combine_mode == CombineMode.RANK_LOCAL_REDUCE:
            context._combine_input_owner = context._dispatch_output_owner
            context.combine_input_buffer = context.dispatch_output_buffer
        else:
            context._combine_input_owner, context.combine_input_buffer = tensor_from_pointer(
                self.cpp_runtime.combine_input_buffer_ptr(),
                (
                    context.world_size * context.max_tokens_per_rank,
                    context.topk,
                    context.hidden_size,
                ),
                torch.bfloat16,
                context.device,
                self.cpp_runtime,
            )

    def _resolve_capacity(self, runtime_max_tokens_per_rank: Optional[int]) -> int:
        mode_context = self.context
        resolved = (
            mode_context.max_tokens_per_rank if runtime_max_tokens_per_rank is None else runtime_max_tokens_per_rank
        )
        if type(resolved) is not int or not 0 < resolved <= mode_context.max_tokens_per_rank:
            raise ValueError("runtime_max_tokens_per_rank must be positive and not exceed max_tokens_per_rank")
        if (
            mode_context.output_layout
            not in (
                DispatchLayout.RANK_MAJOR,
                DispatchLayout.TOKEN_MAJOR,
            )
            and resolved != mode_context.max_tokens_per_rank
        ):
            raise ValueError("runtime_max_tokens_per_rank is only supported by rank-major dispatch")
        return resolved

    def _dispatch_outputs(self, output_buffer: torch.Tensor):
        mode_context = self.context
        device = output_buffer.device
        slots_per_expert = mode_context.world_size * mode_context.max_tokens_per_rank
        if mode_context._dispatch_count is None or mode_context._dispatch_count.device != device:
            mode_context._dispatch_scales = None
            mode_context._dispatch_topk_ids = None
            mode_context._dispatch_weights = None
            if mode_context.output_layout == DispatchLayout.EXPERT_MAJOR:
                mode_context._dispatch_src_info = torch.empty(
                    (mode_context.num_local_experts, slots_per_expert),
                    dtype=torch.int32,
                    device=device,
                )
                mode_context._dispatch_layout_range = torch.empty(
                    (mode_context.num_local_experts, mode_context.world_size),
                    dtype=torch.int64,
                    device=device,
                )
                mode_context._dispatch_count = torch.empty(
                    (mode_context.num_local_experts,), dtype=torch.int32, device=device
                )
                scale_block_size = dispatch_scale_block_size(mode_context.dispatch_data_type)
                if scale_block_size:
                    num_scales = mode_context.hidden_size // scale_block_size
                    scale_storage = torch.empty(
                        (mode_context.num_local_experts, num_scales, slots_per_expert),
                        dtype=dispatch_scale_dtype(mode_context.dispatch_data_type),
                        device=device,
                    )
                    mode_context._dispatch_scales = scale_storage.transpose(1, 2)
            elif mode_context.output_layout == DispatchLayout.RANK_MAJOR:
                mode_context._dispatch_src_info = None
                assert mode_context._output_topk_ids is not None
                assert mode_context._output_topk_weights is not None
                mode_context._dispatch_topk_ids = mode_context._output_topk_ids
                mode_context._dispatch_weights = mode_context._output_topk_weights
                mode_context._dispatch_layout_range = None
                mode_context._dispatch_count = torch.empty((mode_context.world_size,), dtype=torch.int32, device=device)
            elif mode_context.output_layout == DispatchLayout.TOKEN_MAJOR:
                mode_context._dispatch_src_info = None
                assert mode_context._output_topk_ids is not None
                assert mode_context._output_topk_weights is not None
                mode_context._dispatch_topk_ids = mode_context._output_topk_ids
                mode_context._dispatch_weights = mode_context._output_topk_weights
                mode_context._dispatch_layout_range = None
                mode_context._dispatch_count = torch.empty((mode_context.world_size,), dtype=torch.int32, device=device)
            else:
                raise ValueError(f"unsupported latency output layout: {mode_context.output_layout}")
        assert mode_context._dispatch_count is not None
        return (
            output_buffer,
            mode_context._dispatch_scales,
            mode_context._dispatch_src_info,
            mode_context._dispatch_topk_ids,
            mode_context._dispatch_weights,
            mode_context._dispatch_layout_range,
            mode_context._dispatch_count,
        )

    def _validate_dispatch(self, input, topk_ids, weights, quant, output_buffer, active_capacity: int) -> None:
        mode_context = self.context
        if quant is not None:
            raise NotImplementedError(
                "per-call input quant metadata is not supported; configure dispatch output quantization on the communicator"
            )
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden_size] tensor")
        expected_input_dtype = dispatch_tensor_dtype(mode_context.dispatch_data_type)
        if input.device.type != "cuda" or input.dtype != expected_input_dtype:
            raise ValueError(f"latency dispatch input must be a CUDA {expected_input_dtype} tensor")
        if input.size(1) != mode_context.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} does not match configured {mode_context.hidden_size}")
        if input.size(0) > active_capacity:
            raise ValueError("input token count exceeds runtime_max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), mode_context.topk):
            raise ValueError("topk_ids shape must match [input.size(0), configured topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")
        slots_per_expert = mode_context.world_size * mode_context.max_tokens_per_rank
        if mode_context.output_layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (
                mode_context.num_local_experts,
                slots_per_expert,
                mode_context.hidden_size,
            )
        elif mode_context.output_layout == DispatchLayout.RANK_MAJOR:
            expected_shape = (
                mode_context.world_size * mode_context.max_tokens_per_rank,
                mode_context.hidden_size,
            )
        elif mode_context.output_layout == DispatchLayout.TOKEN_MAJOR:
            expected_shape = (
                mode_context.world_size * mode_context.max_tokens_per_rank * mode_context.topk,
                mode_context.hidden_size,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {mode_context.output_layout}")
        if mode_context.output_layout in (DispatchLayout.RANK_MAJOR, DispatchLayout.TOKEN_MAJOR):
            assert mode_context.dispatch_output_buffer is not None
            if output_buffer.data_ptr() != mode_context.dispatch_output_buffer.data_ptr():
                raise ValueError(f"{mode_context.output_layout} output uses the runtime-owned dispatch output buffer")
            return
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {mode_context.output_layout} tensor")
        expected_dtype = dispatch_tensor_dtype(mode_context.dispatch_data_type)
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine(self, expert_output, handle, out) -> None:
        mode_context = self.context
        if not isinstance(handle, DispatchHandle) or not isinstance(
            handle._context, (_ExpertMajorCombineContext, _RankMajorCombineContext, _TokenMajorCombineContext)
        ):
            raise ValueError("DispatchHandle does not contain latency combine context")
        context = handle._context
        if context.num_experts != mode_context.num_experts or context.hidden_size != mode_context.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        if handle.output_info.layout.kind != mode_context.output_layout:
            raise ValueError("DispatchHandle output layout does not match this MoECommunicator")
        output_quant = handle.output_info.quant
        handle_data_type = DispatchDataType.BF16 if output_quant is None else output_quant.format
        if handle_data_type != mode_context.dispatch_data_type:
            raise ValueError("DispatchHandle quantization does not match this MoECommunicator configuration")
        active_capacity = (
            context.max_tokens_per_rank
            if isinstance(context, (_RankMajorCombineContext, _TokenMajorCombineContext))
            else mode_context.max_tokens_per_rank
        )
        slots_per_expert = mode_context.world_size * active_capacity
        if handle.output_info.layout.kind == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (
                mode_context.num_local_experts,
                slots_per_expert,
                mode_context.hidden_size,
            )
        elif (
            handle.output_info.layout.kind == DispatchLayout.RANK_MAJOR
            and mode_context.combine_mode == CombineMode.DIRECT_SEND
        ):
            expected_shape = (
                mode_context.world_size * active_capacity,
                mode_context.topk,
                mode_context.hidden_size,
            )
        elif handle.output_info.layout.kind == DispatchLayout.RANK_MAJOR:
            expected_shape = (
                mode_context.world_size * active_capacity,
                mode_context.hidden_size,
            )
        elif handle.output_info.layout.kind == DispatchLayout.TOKEN_MAJOR:
            expected_shape = (
                mode_context.world_size * active_capacity * mode_context.topk,
                mode_context.hidden_size,
            )
        else:
            raise ValueError(f"unsupported latency output layout: {handle.output_info.layout.kind}")
        if expert_output.dim() != len(expected_shape) or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous layout")
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        expected_dtype = combine_tensor_dtype(mode_context.dispatch_data_type)
        if expert_output.dtype != expected_dtype:
            raise ValueError(f"expert_output must be {expected_dtype}")
        if handle.output_info.layout.kind in (DispatchLayout.RANK_MAJOR, DispatchLayout.TOKEN_MAJOR):
            assert mode_context.combine_input_buffer is not None
            if expert_output.data_ptr() != mode_context.combine_input_buffer.data_ptr():
                raise ValueError(
                    f"{handle.output_info.layout.kind} combine requires the runtime-owned combine input buffer"
                )
        if out is not None:
            expected_out_shape = (context.num_tokens, mode_context.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != expected_dtype or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous {expected_dtype} tensor with shape {expected_out_shape}")
