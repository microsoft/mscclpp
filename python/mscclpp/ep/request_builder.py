# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build validated tensor-backed requests without enqueueing communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from . import _cpp
from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode
from .types import DispatchHandle, DispatchLayoutInfo, DispatchOutput, MoECommunicatorConfig, PrepareHandle, QuantConfig
from .utils import (
    check_tensor,
    dispatch_shape,
    payload_dtype,
    ptr,
    resolve_dispatch_format,
    tensor_from_pointer,
    verify_active_capacity,
)


@dataclass
class _DispatchRequest:
    arguments: dict[str, Union[int, DispatchDataType, _cpp.PrepareHandle]]
    output: DispatchOutput
    num_tokens: int
    active_capacity: int
    tensors: Tuple[torch.Tensor, ...]


@dataclass
class _CombineRequest:
    arguments: dict[str, Union[int, _cpp.DispatchHandle]]
    output: torch.Tensor


class RequestBuilder:
    """Construct mode-specific requests using the communicator's config and runtime.

    Configuration and native storage are borrowed, not copied. The communicator
    owns initialization, stream binding, handle validation, and native execution.
    Dispatch requests retain every tensor borrowed by the native operation.
    """

    def __init__(self, config: MoECommunicatorConfig, runtime: _cpp.MoERuntime) -> None:
        self._config = config
        self._runtime = runtime

    def build_prepare(
        self,
        topk_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        runtime_max_tokens_per_rank: Optional[int],
        output_count: Optional[torch.Tensor],
    ) -> dict[str, int]:
        config = self._config
        if config.mode != MoEMode.THROUGHPUT:
            raise ValueError("prepare() is supported only in THROUGHPUT mode")
        active = verify_active_capacity(runtime_max_tokens_per_rank, config.max_tokens_per_rank)
        self._check(topk_ids, "topk_ids", (None, config.topk), torch.int64, 8)
        if output_count is not None:
            count_shape = (
                (self._runtime.num_ranks,)
                if config.output_layout == DispatchLayout.RANK_MAJOR
                else (config.num_local_experts,)
            )
            self._check(output_count, "output_count", count_shape, torch.int32, 4)
        num_tokens = topk_ids.shape[0]
        if num_tokens > active:
            raise ValueError("topk_ids token count exceeds runtime_max_tokens_per_rank")
        return dict(
            topk_idx_ptr=ptr(topk_ids),
            num_tokens=num_tokens,
            max_tokens_per_rank=active,
            num_blocks=config.num_blocks[0],
            stream_ptr=stream.cuda_stream,
            output_count_ptr=ptr(output_count),
        )

    def build_latency_dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        output_buffer: Optional[torch.Tensor],
        stream: torch.cuda.Stream,
        runtime_max_tokens_per_rank: Optional[int],
    ) -> _DispatchRequest:
        config = self._config
        active = verify_active_capacity(runtime_max_tokens_per_rank, config.max_tokens_per_rank)
        data_type, _ = resolve_dispatch_format(config, quant)
        num_tokens = self._validate_dispatch_input(input, topk_ids, weights, torch.bfloat16, active)
        output_shape = dispatch_shape(config, self._runtime.num_ranks, active)
        tokens = self._dispatch_buffer(
            output_buffer,
            output_shape,
            payload_dtype(data_type),
            runtime_owned=config.output_layout != DispatchLayout.EXPERT_MAJOR,
        )
        rank_major = config.output_layout == DispatchLayout.RANK_MAJOR
        count = torch.empty(
            (self._runtime.num_ranks if rank_major else config.num_local_experts,),
            dtype=torch.int32,
            device=config.device,
        )
        src_info = layout_range = recv_ids = recv_weights = scales = combine_input = None
        if rank_major:
            recv_ids, recv_weights = self._routing_views(output_shape)
            combine_input = self._view(
                self._runtime.combine_input_buffer_ptr(), self._combine_shape(active), torch.bfloat16
            )
        else:
            src_info = torch.empty(output_shape[:-1], dtype=torch.int32, device=config.device)
            # Each entry packs the source rank's count and offset for one local expert.
            layout_range = torch.empty(
                (config.num_local_experts, self._runtime.num_ranks), dtype=torch.int64, device=config.device
            )
        if data_type == DispatchDataType.FP8_E4M3:
            scales = torch.empty(
                (config.num_local_experts, config.hidden_size // 128, self._runtime.num_ranks * active),
                dtype=torch.float32,
                device=config.device,
            ).transpose(1, 2)
        output = DispatchOutput(
            tokens=tokens,
            quant=None if scales is None else QuantConfig(format=data_type, block_scales=scales),
            layout=DispatchLayoutInfo(
                kind=config.output_layout,
                num_tokens_per_expert=None if rank_major else count,
                num_tokens_per_rank=count if rank_major else None,
            ),
            topk_ids=recv_ids,
            weights=recv_weights,
            combine_input_buffer=combine_input,
        )
        return _DispatchRequest(
            arguments=dict(
                input_ptr=ptr(input),
                topk_idx_ptr=ptr(topk_ids),
                topk_weights_ptr=ptr(weights),
                output_ptr=ptr(tokens),
                output_scales_ptr=ptr(scales),
                output_src_info_ptr=ptr(src_info),
                output_topk_idx_ptr=ptr(recv_ids),
                output_topk_weights_ptr=ptr(recv_weights),
                output_layout_range_ptr=ptr(layout_range),
                output_count_ptr=ptr(count),
                num_tokens=num_tokens,
                max_tokens_per_rank=active,
                invalid_token_expert_id=config.invalid_token_expert_id,
                dispatch_data_type=data_type,
                num_blocks=config.num_blocks[0],
                stream_ptr=stream.cuda_stream,
            ),
            output=output,
            num_tokens=num_tokens,
            active_capacity=active,
            tensors=_retain(
                input,
                topk_ids,
                weights,
                tokens,
                scales,
                src_info,
                layout_range,
                recv_ids,
                recv_weights,
                count,
                combine_input,
            ),
        )

    def build_throughput_dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        output_buffer: Optional[torch.Tensor],
        stream: torch.cuda.Stream,
        prepare_handle: Optional[PrepareHandle],
        runtime_max_tokens_per_rank: Optional[int],
    ) -> _DispatchRequest:
        config = self._config
        active = verify_active_capacity(runtime_max_tokens_per_rank, config.max_tokens_per_rank)
        data_type, input_scales = resolve_dispatch_format(config, quant)
        num_tokens = self._validate_dispatch_input(input, topk_ids, weights, payload_dtype(data_type), active)
        if data_type == DispatchDataType.FP8_E4M3:
            if input_scales is None:
                raise ValueError("throughput FP8 dispatch requires quant.block_scales")
            self._check(input_scales, "quant.block_scales", (num_tokens, config.hidden_size // 128), torch.float32, 4)
        output_shape = dispatch_shape(config, self._runtime.num_ranks, active)
        tokens = self._dispatch_buffer(output_buffer, output_shape, payload_dtype(data_type), runtime_owned=True)
        recv_ids, recv_weights = self._routing_views(output_shape)
        scales = (
            self._view(
                self._runtime.output_scales_buffer_ptr(),
                output_shape[:-1] + (config.hidden_size // 128,),
                torch.float32,
            )
            if data_type == DispatchDataType.FP8_E4M3
            else None
        )
        combine_input = self._view(
            self._runtime.combine_input_buffer_ptr(), self._combine_shape(active), torch.bfloat16
        )
        output = DispatchOutput(
            tokens=tokens,
            quant=None if scales is None else QuantConfig(format=data_type, block_scales=scales),
            layout=DispatchLayoutInfo(kind=config.output_layout),
            topk_ids=recv_ids,
            weights=recv_weights,
            combine_input_buffer=combine_input,
        )
        return _DispatchRequest(
            arguments=dict(
                input_ptr=ptr(input),
                input_scales_ptr=ptr(input_scales),
                topk_idx_ptr=ptr(topk_ids),
                topk_weights_ptr=ptr(weights),
                num_tokens=num_tokens,
                max_tokens_per_rank=active,
                dispatch_data_type=data_type,
                num_blocks=config.num_blocks[0],
                stream_ptr=stream.cuda_stream,
                prepare_handle=_cpp.PrepareHandle() if prepare_handle is None else prepare_handle._native,
            ),
            output=output,
            num_tokens=num_tokens,
            active_capacity=active,
            tensors=_retain(
                input, topk_ids, weights, input_scales, tokens, scales, recv_ids, recv_weights, combine_input
            ),
        )

    def build_latency_combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        out: Optional[torch.Tensor],
        stream: torch.cuda.Stream,
        apply_router_weights: Optional[bool],
    ) -> _CombineRequest:
        if apply_router_weights is None:
            apply_router_weights = self._config.combine_mode == CombineMode.DIRECT_SEND
        output = self._combine_output(expert_output, handle, out)
        return _CombineRequest(
            arguments=dict(
                expert_output_ptr=ptr(expert_output),
                output_ptr=ptr(output),
                handle=handle._native,
                num_blocks=self._config.num_blocks[1],
                stream_ptr=stream.cuda_stream,
                apply_router_weights=apply_router_weights,
            ),
            output=output,
        )

    def build_throughput_combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        out: Optional[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> _CombineRequest:
        output = self._combine_output(expert_output, handle, out)
        return _CombineRequest(
            arguments=dict(
                output_ptr=ptr(output),
                handle=handle._native,
                num_blocks=self._config.num_blocks[1],
                stream_ptr=stream.cuda_stream,
            ),
            output=output,
        )

    def _validate_dispatch_input(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        dtype: torch.dtype,
        active: int,
    ) -> int:
        config = self._config
        self._check(input, "input", (None, config.hidden_size), dtype)
        num_tokens = input.shape[0]
        if num_tokens > active:
            raise ValueError("input token count exceeds runtime_max_tokens_per_rank")
        self._check(topk_ids, "topk_ids", (num_tokens, config.topk), torch.int64, 8)
        if weights is not None:
            self._check(weights, "weights", (num_tokens, config.topk), torch.float32, 4)
        return num_tokens

    def _dispatch_buffer(
        self, output_buffer: Optional[torch.Tensor], shape: tuple, dtype: torch.dtype, *, runtime_owned: bool
    ) -> torch.Tensor:
        if output_buffer is not None:
            self._check(output_buffer, "output_buffer", shape, dtype)
        runtime_output_ptr = self._runtime.dispatch_output_buffer_ptr()
        if runtime_owned and output_buffer is not None and ptr(output_buffer) != runtime_output_ptr:
            raise ValueError("output_buffer must be the runtime-owned dispatch buffer for this mode/layout")
        if output_buffer is None or ptr(output_buffer) == runtime_output_ptr:
            return self._view(runtime_output_ptr, shape, dtype)
        return output_buffer

    def _routing_views(self, output_shape: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        metadata_shape = output_shape[:-1] + (self._config.topk,)
        return (
            self._view(self._runtime.output_topk_ids_buffer_ptr(), metadata_shape, torch.int32),
            self._view(self._runtime.output_topk_weights_buffer_ptr(), metadata_shape, torch.float32),
        )

    def _combine_output(
        self, expert_output: torch.Tensor, handle: DispatchHandle, out: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self._check(expert_output, "expert_output", self._combine_shape(handle._active_capacity), torch.bfloat16)
        if (
            self._config.output_layout != DispatchLayout.EXPERT_MAJOR
            and ptr(expert_output) != self._runtime.combine_input_buffer_ptr()
        ):
            raise ValueError("expert_output must be DispatchOutput.combine_input_buffer for this mode/layout")
        output_shape = (handle._num_tokens, self._config.hidden_size)
        if out is not None:
            self._check(out, "out", output_shape, torch.bfloat16)
            return out
        return torch.empty(output_shape, dtype=torch.bfloat16, device=self._config.device)

    def _combine_shape(self, active: int) -> tuple:
        config = self._config
        if (
            config.mode == MoEMode.LATENCY
            and config.output_layout == DispatchLayout.RANK_MAJOR
            and config.combine_mode == CombineMode.DIRECT_SEND
        ):
            return self._runtime.num_ranks, active, config.topk, config.hidden_size
        return dispatch_shape(config, self._runtime.num_ranks, active)

    def _check(self, tensor, name, shape, dtype, alignment=16) -> None:
        check_tensor(tensor, name, shape=shape, dtype=dtype, device=self._config.device, alignment=alignment)

    def _view(self, pointer: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        return tensor_from_pointer(pointer, shape, dtype, self._config.device, self._runtime)[1]


def _retain(*tensors: Optional[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor for tensor in tensors if tensor is not None)
