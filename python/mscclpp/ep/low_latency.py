# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Low-latency backend for the high-level MoE communicator."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ._cpp import DispatchLayout, MoEMode, _cpp, get_low_latency_rdma_size_hint
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    ExpertMajorDispatchHandle,
    ExpertMajorCombineContext,
    MoECommunicatorConfig,
    QuantConfig,
)
from .utils import cuda_stream_ptr, requires_dequantization, resolve_expert_placement


class LowLatencyRuntime:
    """Private low-level low-latency runtime wrapper (wraps ``_cpp.MoERuntime``)."""

    num_sms: int = 20

    def __init__(
        self,
        comm: Any,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        mode: MoEMode = MoEMode.LOW_LATENCY,
        num_qps_per_rank: int = 12,
    ) -> None:
        if not isinstance(mode, MoEMode):
            raise TypeError("mode must be a MoEMode")
        if mode != MoEMode.LOW_LATENCY:
            raise NotImplementedError("LowLatencyRuntime supports only MoEMode.LOW_LATENCY")
        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

        self.mode = mode
        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.num_qps_per_rank = num_qps_per_rank
        self.cpp_runtime = _cpp.MoERuntime(comm.communicator, num_nvl_bytes, num_rdma_bytes, mode)

    def is_available(self) -> bool:
        return self.cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        return self.cpp_runtime.is_internode_available()

    def get_local_device_id(self) -> int:
        return self.cpp_runtime.get_local_device_id()

    def get_num_rdma_ranks(self) -> int:
        return self.cpp_runtime.get_num_rdma_ranks()

    def get_rdma_rank(self) -> int:
        return self.cpp_runtime.get_rdma_rank()

    def get_root_rdma_rank(self, global_: bool) -> int:
        return self.cpp_runtime.get_root_rdma_rank(global_)


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
        self.num_sms = config.num_sms
        self.enable_overlap = config.enable_overlap

        if self.output_layout != DispatchLayout.EXPERT_MAJOR:
            raise NotImplementedError("low-latency mode currently supports only DispatchLayout.EXPERT_MAJOR")
        if self.num_experts % self.world_size != 0:
            raise ValueError("low-latency mode requires num_experts divisible by world_size")

        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )

        if config.max_recv_tokens_per_rank not in (None, self.max_tokens_per_rank):
            raise NotImplementedError("low-latency mode currently uses max_tokens_per_rank as recv capacity")
        self.quant = config.quant
        self.quant_dtype = None if self.quant is None else self.quant.dtype
        if self.quant is not None and self.quant_dtype is None:
            raise ValueError("quant.dtype is required when quant is provided")
        if self.quant_dtype not in (None, torch.float8_e4m3fn):
            raise NotImplementedError(f"unsupported low-latency quant dtype: {self.quant_dtype}")
        self.dispatch_requires_quantization = self.quant_dtype is not None

        num_rdma_bytes = get_low_latency_rdma_size_hint(
            self.max_tokens_per_rank, self.hidden_size, self.world_size, self.num_experts, self.topk
        )
        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._runtime = LowLatencyRuntime(
            comm,
            num_nvl_bytes=0,
            num_rdma_bytes=num_rdma_bytes,
            mode=self.mode,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )
        # LL uses the registered symmetric buffer for both IPC/NVLink and RDMA-backed
        # modes. A single-node LL job is not internode topology-wise.
        # num_rdma_ranks > 1 iff world_size spans more than one local NVLink domain.
        self._is_internode = self._runtime.get_num_rdma_ranks() > 1

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
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        out_buf, packed_scales, src_info, layout_range, count = self._get_dispatch_output_tensors(output_buffer)
        self._runtime.cpp_runtime.dispatch(
            input.data_ptr(),
            topk_ids.data_ptr(),
            weights.data_ptr(),
            out_buf.data_ptr(),
            0 if packed_scales is None else packed_scales.data_ptr(),
            src_info.data_ptr(),
            layout_range.data_ptr(),
            count.data_ptr(),
            input.size(0),
            self.hidden_size,
            self.topk,
            self.max_tokens_per_rank,
            self.num_experts,
            self.dispatch_requires_quantization,
            self.output_layout,
            cuda_stream_ptr(stream),
        )
        dispatched_quant = None
        if packed_scales is not None:
            dispatched_quant = QuantConfig(dtype=self.quant_dtype, block_scales=packed_scales, block_size=128)
        output_info = DispatchOutputInfo(
            layout=DispatchLayoutInfo(kind=self.output_layout, num_tokens_per_expert=count),
            quant=dispatched_quant,
        )
        dispatch_out = DispatchOutput(
            tokens=out_buf,
            quant=output_info.quant,
            layout=output_info.layout,
        )
        handle = ExpertMajorDispatchHandle(
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
        if not isinstance(handle, ExpertMajorDispatchHandle):
            raise ValueError("DispatchHandle does not contain expert-major combine context")
        context = handle.combine_context
        combine_requires_dequantization = requires_dequantization(expert_output)
        x_scales = None
        if combine_requires_dequantization:
            if handle.output_info.quant is None or handle.output_info.quant.block_scales is None:
                raise ValueError("FP8 expert_output requires scales captured in the dispatch handle")
            x_scales = handle.output_info.quant.block_scales
        if out is None:
            out = torch.empty((context.num_tokens, self.hidden_size), dtype=torch.bfloat16, device=expert_output.device)
        self._runtime.cpp_runtime.combine(
            expert_output.data_ptr(),
            0 if x_scales is None else x_scales.data_ptr(),
            context.topk_ids.data_ptr(),
            context.weights.data_ptr(),
            context.src_info.data_ptr(),
            context.layout_range.data_ptr(),
            out.data_ptr(),
            context.num_tokens,
            self.hidden_size,
            context.weights.size(1),
            context.num_max_dispatch_tokens_per_rank,
            context.num_experts,
            combine_requires_dequantization,
            cuda_stream_ptr(stream),
        )
        return out

    def _get_dispatch_output_tensors(self, output_buffer: torch.Tensor):
        device = output_buffer.device
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self._dispatch_src_info is None or self._dispatch_src_info.device != device:
            self._dispatch_src_info = torch.empty(
                (self.num_local_experts, slots_per_expert), dtype=torch.int32, device=device
            )
            self._dispatch_layout_range = torch.empty(
                (self.num_local_experts, self.world_size), dtype=torch.int64, device=device
            )
            self._dispatch_count = torch.empty((self.num_local_experts,), dtype=torch.int32, device=device)
            self._dispatch_scales = None
            if self.dispatch_requires_quantization:
                num_scales = self.hidden_size // 128
                scales_storage = torch.empty(
                    (self.num_local_experts, num_scales, slots_per_expert), dtype=torch.float32, device=device
                )
                self._dispatch_scales = scales_storage.transpose(1, 2)
        return (
            output_buffer,
            self._dispatch_scales,
            self._dispatch_src_info,
            self._dispatch_layout_range,
            self._dispatch_count,
        )

    def _validate_dispatch_inputs(self, input, topk_ids, weights, quant, output_buffer) -> None:
        if output_buffer is None:
            raise ValueError("output_buffer is required for low-latency dispatch")
        if quant is not None and (quant.block_scales is not None or quant.global_scale is not None):
            raise NotImplementedError("low-latency dispatch does not support quantized input scales yet")
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
        expected_dtype = torch.float8_e4m3fn if self.dispatch_requires_quantization else torch.bfloat16
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        else:
            expected_shape = (self.num_local_experts * slots_per_expert, self.hidden_size)
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {self.output_layout} tensor")
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine_inputs(self, expert_output, handle, out) -> None:
        if not isinstance(handle, ExpertMajorDispatchHandle):
            raise ValueError("DispatchHandle does not contain expert-major combine context")
        context = handle.combine_context
        if context.num_experts != self.num_experts or context.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if handle.output_info.layout.kind == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        else:
            expected_shape = (self.num_local_experts * slots_per_expert, self.hidden_size)
        if expert_output.dim() != len(expected_shape) or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous layout")
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        if expert_output.dtype not in (torch.bfloat16, getattr(torch, "float8_e4m3fn", None)):
            raise ValueError("expert_output must be BF16 or FP8 E4M3")
        if out is not None:
            expected_out_shape = (context.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")
