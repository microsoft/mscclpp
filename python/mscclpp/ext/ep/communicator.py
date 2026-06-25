# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Python frontend for the MSCCL++ Expert-Parallel extension.

This is a thin wrapper around the nanobind extension ``mscclpp_ep_cpp``.
``MoECommunicator`` is the high-level API. ``MoERuntime`` is a lower-level
low-latency runtime wrapper used by the high-level API.

Current status (see ``src/ext/ep/README.md``):

* Intranode (NVLink-only) dispatch and combine: ported and validated on
  one node with 8 GPUs.
* ``get_dispatch_layout``: ported.
* Internode HT (MSCCL++ PortChannel + MemoryChannel) dispatch and combine:
  ported and validated on 2 nodes x 8 H100 GPUs with
  ``test/python/ext/ep/test_internode_multirank.py``.
* Low-latency kernels (RDMA + CUDA IPC paths):
  ported and validated on intra-node and 2 nodes x 8 H100 GPUs with
  ``test/python/ext/ep/test_low_latency_multirank.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from mscclpp._core import CommGroup

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with -DMSCCLPP_BUILD_EXT_EP=ON "
        "or install with `pip install .[ep]`."
    ) from exc


class DispatchLayout(str, Enum):
    FLAT = "flat"
    EXPERT_MAJOR = "expert_major"


@dataclass
class MoECommunicatorConfig:
    """Configuration for the high-level MoE dispatch/combine API."""

    comm: Optional[CommGroup] = None
    device: Optional[torch.device | int] = None
    num_experts: int = 0
    num_local_experts: Optional[int] = None
    local_expert_start: Optional[int] = None
    hidden_size: int = 0
    topk: int = 0
    max_tokens_per_rank: int = 0
    max_recv_tokens_per_rank: Optional[int] = None
    mode: str = "ll"
    output_layout: Optional[DispatchLayout | str] = None
    input_dtype: Optional[torch.dtype] = None
    quant_format: Optional[str] = None
    num_rdma_qps_per_rank: int = 12
    num_sms: int = 20
    comm_stream: Optional[torch.cuda.Stream] = None
    enable_overlap: bool = False


@dataclass
class QuantScales:
    local: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    format: Optional[str] = None
    block_size: Optional[int] = None


@dataclass
class DispatchOutput:
    tokens: torch.Tensor
    scales: Optional[QuantScales]
    num_tokens_per_expert: torch.Tensor | list[int]
    expert_offsets: Optional[torch.Tensor] = None
    layout: DispatchLayout = DispatchLayout.FLAT


@dataclass
class DispatchHandle:
    """Opaque dispatch metadata consumed by :meth:`MoECommunicator.combine`."""

    topk_ids: torch.Tensor
    weights: torch.Tensor
    src_info: torch.Tensor
    layout_range: torch.Tensor
    num_max_dispatch_tokens_per_rank: int
    num_experts: int
    num_tokens: int
    hidden_size: int
    num_local_experts: int
    local_expert_start: int
    layout: DispatchLayout
    output_scales: Optional[QuantScales] = None


@dataclass
class CommOverlapConfig:
    op: str
    level: str = "op"
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    signal: Optional[torch.Tensor] = None
    num_comm_sms: Optional[int] = None
    block_m: Optional[int] = None
    block_ready_value: Optional[int] = None


class MoERuntime:
    """Low-level MoE communication runtime wrapper.

    Parameters
    ----------
    comm:
    The :class:`mscclpp.CommGroup`. Used only for out-of-band
        exchange of IPC handles and the MSCCL++ unique id.
    num_nvl_bytes:
        Size of the NVLink-accessible scratch buffer (shared via CUDA IPC).
    num_rdma_bytes:
        Size of the RDMA scratch buffer. Required (>0) for internode HT and
        low-latency modes.
    low_latency_mode:
        Enable low-latency buffer setup for :class:`MoECommunicator`. New
        callers should use :class:`MoECommunicator` instead of direct LL
        methods on this low-level runtime.
    num_qps_per_rank:
        Ignored for intranode mode.
    """

    #: Default number of SMs reserved for comms kernels. Matches DeepEP.
    num_sms: int = 20

    def __init__(
        self,
        comm: CommGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
    ) -> None:
        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank

        self._cpp_runtime = _cpp.MoERuntime(comm.communicator, num_nvl_bytes, num_rdma_bytes, low_latency_mode)

        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")
        self._cpp_runtime.sync()

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._cpp_runtime.is_internode_available()

    def get_local_device_id(self) -> int:
        return self._cpp_runtime.get_local_device_id()

    def get_num_rdma_ranks(self) -> int:
        return self._cpp_runtime.get_num_rdma_ranks()

    def get_rdma_rank(self) -> int:
        return self._cpp_runtime.get_rdma_rank()

    def get_root_rdma_rank(self, global_: bool) -> int:
        return self._cpp_runtime.get_root_rdma_rank(global_)


class MoECommunicator:
    """High-level MoE communicator API for dispatch/combine.

    The first implementation supports the low-latency backend.
    """

    def __init__(self, config: Optional[MoECommunicatorConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either MoECommunicatorConfig or keyword arguments, not both")
        if config is None:
            if "group" in kwargs and "comm" not in kwargs:
                kwargs["comm"] = kwargs.pop("group")
            config = MoECommunicatorConfig(**kwargs)

        if config.device is not None:
            torch.cuda.set_device(config.device)

        comm = config.comm
        if comm is None:
            raise ValueError("MoECommunicator requires an mscclpp.CommGroup")

        self.comm = comm
        self.rank: int = comm.my_rank
        self.world_size: int = comm.nranks
        self.local_rank: int = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)

        self.mode = config.mode.lower()
        if self.mode != "ll":
            raise NotImplementedError("MoECommunicator currently supports only mode='ll'")

        self.output_layout = _normalize_output_layout(config.output_layout, self.mode)

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        if self.num_experts <= 0 or self.hidden_size <= 0 or self.topk <= 0 or self.max_tokens_per_rank <= 0:
            raise ValueError("num_experts, hidden_size, topk, and max_tokens_per_rank must be positive")
        if self.num_experts % self.world_size != 0:
            raise ValueError("low-latency mode requires num_experts divisible by world_size")

        self.num_local_experts = config.num_local_experts
        if self.num_local_experts is None:
            self.num_local_experts = self.num_experts // self.world_size
        if self.num_local_experts * self.world_size != self.num_experts:
            raise NotImplementedError("only even contiguous expert placement is currently supported")

        self.local_expert_start = config.local_expert_start
        if self.local_expert_start is None:
            self.local_expert_start = self.rank * self.num_local_experts

        if config.max_recv_tokens_per_rank not in (None, self.max_tokens_per_rank):
            raise NotImplementedError("low-latency mode currently uses max_tokens_per_rank as recv capacity")
        if config.input_dtype not in (None, torch.bfloat16):
            raise NotImplementedError("low-latency dispatch currently supports BF16 input only")

        self.quant_format = _normalize_quant_format(config.quant_format)
        self.dispatch_use_fp8 = self.quant_format == "fp8_e4m3"
        if self.quant_format not in (None, "fp8_e4m3"):
            raise NotImplementedError(f"unsupported low-latency quant_format: {config.quant_format}")

        num_nvl_bytes = 0
        num_rdma_bytes = _get_low_latency_rdma_size_hint(
            self.max_tokens_per_rank, self.hidden_size, self.world_size, self.num_experts
        )

        self.comm_stream = config.comm_stream
        if self.comm_stream is not None:
            raise NotImplementedError("custom comm_stream is not wired into the low-latency runtime yet")
        self.enable_overlap = config.enable_overlap
        self.num_sms = config.num_sms
        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._runtime = MoERuntime(
            comm,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )

    def is_available(self) -> bool:
        return self._runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._runtime.is_internode_available()

    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        scales: Optional[QuantScales] = None,
        *,
        output_buffer: torch.Tensor,
    ) -> tuple[DispatchOutput, DispatchHandle]:
        self._validate_dispatch_inputs(input, topk_ids, weights, scales, output_buffer)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        output_tensors = self._get_dispatch_output_tensors(output_buffer)
        output_buffer, packed_scales, src_info, layout_range, num_tokens_per_expert = output_tensors
        self._runtime._cpp_runtime.dispatch(
            input.data_ptr(),
            topk_ids.data_ptr(),
            output_buffer.data_ptr(),
            0 if packed_scales is None else packed_scales.data_ptr(),
            src_info.data_ptr(),
            layout_range.data_ptr(),
            num_tokens_per_expert.data_ptr(),
            input.size(0),
            self.hidden_size,
            self.topk,
            self.max_tokens_per_rank,
            self.num_experts,
            self.dispatch_use_fp8,
            _cpp_dispatch_layout(self.output_layout),
            torch.cuda.current_stream().cuda_stream,
        )
        output_scales = None
        if packed_scales is not None:
            output_scales = QuantScales(local=packed_scales, format="fp8_e4m3", block_size=128)

        dispatch_out = DispatchOutput(
            tokens=output_buffer,
            scales=output_scales,
            num_tokens_per_expert=num_tokens_per_expert,
            expert_offsets=None,
            layout=self.output_layout,
        )
        handle = DispatchHandle(
            topk_ids=topk_ids,
            weights=weights,
            src_info=src_info,
            layout_range=layout_range,
            num_max_dispatch_tokens_per_rank=self.max_tokens_per_rank,
            num_experts=self.num_experts,
            num_tokens=input.size(0),
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
            local_expert_start=self.local_expert_start,
            layout=self.output_layout,
            output_scales=output_scales,
        )
        return dispatch_out, handle

    def _get_dispatch_output_tensors(self, output_buffer: torch.Tensor) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = output_buffer.device
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self._dispatch_src_info is None or self._dispatch_src_info.device != device:
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
            self._dispatch_count = torch.empty(
                (self.num_local_experts,),
                dtype=torch.int32,
                device=device,
            )
            self._dispatch_scales = None
            if self.dispatch_use_fp8:
                num_scales = self.hidden_size // 128
                scales_storage = torch.empty(
                    (self.num_local_experts, num_scales, slots_per_expert),
                    dtype=torch.float32,
                    device=device,
                )
                self._dispatch_scales = scales_storage.transpose(1, 2)

        assert self._dispatch_src_info is not None
        assert self._dispatch_layout_range is not None
        assert self._dispatch_count is not None
        return (
            output_buffer,
            self._dispatch_scales,
            self._dispatch_src_info,
            self._dispatch_layout_range,
            self._dispatch_count,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_combine_inputs(expert_output, handle, out)
        x_scales = None
        if _is_fp8_e4m3_tensor(expert_output):
            if handle.output_scales is None or handle.output_scales.local is None:
                raise ValueError("FP8 expert_output requires scales captured in the dispatch handle")
            x_scales = handle.output_scales.local

        if out is None:
            out = torch.empty((handle.num_tokens, self.hidden_size), dtype=torch.bfloat16, device=expert_output.device)
        self._runtime._cpp_runtime.combine(
            expert_output.data_ptr(),
            0 if x_scales is None else x_scales.data_ptr(),
            handle.topk_ids.data_ptr(),
            handle.weights.data_ptr(),
            handle.src_info.data_ptr(),
            handle.layout_range.data_ptr(),
            out.data_ptr(),
            handle.num_tokens,
            self.hidden_size,
            handle.weights.size(1),
            handle.num_max_dispatch_tokens_per_rank,
            handle.num_experts,
            _is_fp8_e4m3_tensor(expert_output),
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def dispatch_async(self, *args, **kwargs):
        raise NotImplementedError("dispatch_async is not implemented for MoECommunicator yet")

    def combine_async(self, *args, **kwargs):
        raise NotImplementedError("combine_async is not implemented for MoECommunicator yet")

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

    def _validate_dispatch_inputs(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        scales: Optional[QuantScales],
        output_buffer: torch.Tensor,
    ) -> None:
        if output_buffer is None:
            raise ValueError("output_buffer is required for low-latency dispatch")
        if scales is not None and (scales.local is not None or scales.global_scale is not None):
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
        expected_dtype = torch.float8_e4m3fn if self.dispatch_use_fp8 else torch.bfloat16
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        else:
            expected_shape = (self.num_local_experts * slots_per_expert, self.hidden_size)
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {self.output_layout.value} tensor")
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine_inputs(
        self, expert_output: torch.Tensor, handle: DispatchHandle, out: Optional[torch.Tensor]
    ) -> None:
        if handle.num_experts != self.num_experts or handle.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if handle.layout == DispatchLayout.EXPERT_MAJOR:
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
            expected_out_shape = (handle.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")


def _normalize_output_layout(layout: Optional[DispatchLayout | str], mode: str) -> DispatchLayout:
    if layout is None:
        return DispatchLayout.EXPERT_MAJOR if mode == "ll" else DispatchLayout.FLAT
    if isinstance(layout, DispatchLayout):
        return layout

    normalized = layout.lower().replace("-", "_")
    if normalized in ("flat", "flat_2d", "flat_expert_major"):
        return DispatchLayout.FLAT
    if normalized in ("expert_major", "expert_major_3d", "padded_expert_major"):
        return DispatchLayout.EXPERT_MAJOR
    raise ValueError(f"unsupported dispatch output layout: {layout}")


def _cpp_dispatch_layout(layout: DispatchLayout) -> int:
    if layout == DispatchLayout.EXPERT_MAJOR:
        return 0
    if layout == DispatchLayout.FLAT:
        return 1
    raise ValueError(f"unsupported dispatch output layout: {layout}")


def _normalize_quant_format(fmt: Optional[str]) -> Optional[str]:
    if fmt is None:
        return None
    normalized = fmt.lower().replace("-", "_")
    if normalized in ("fp8", "fp8_e4m3", "f8e4m3", "float8_e4m3fn"):
        return "fp8_e4m3"
    return normalized


def _is_fp8_e4m3_tensor(tensor: torch.Tensor) -> bool:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    return fp8_dtype is not None and tensor.dtype == fp8_dtype


def _get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
) -> int:
    return _cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
