# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Python frontend for the MSCCL++ Expert-Parallel extension.

This is a thin wrapper around the nanobind extension ``mscclpp_ep_cpp``.
``MoECommunicator`` is the high-level API. ``ExpertParallelRuntime`` is a
lower-level runtime wrapper used by legacy HT/intranode tests and advanced
callers.

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
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from mscclpp._core import CommGroup

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON (and ensure PyTorch's CMake prefix is on "
        "CMAKE_PREFIX_PATH) or install via `pip install` after the build."
    ) from exc

Config = _cpp.Config
EventHandle = _cpp.EventHandle


def _send_bytes(comm: CommGroup, data: bytes, peer: int, tag: int) -> None:
    size = np.array([len(data)], dtype=np.int64)
    comm.send(size, peer, tag)
    if data:
        payload = np.frombuffer(data, dtype=np.uint8).copy()
        comm.send(payload, peer, tag + 1)


def _recv_bytes(comm: CommGroup, peer: int, tag: int) -> bytes:
    size = np.empty(1, dtype=np.int64)
    comm.recv(size, peer, tag)
    payload_size = int(size[0])
    if payload_size == 0:
        return b""
    payload = np.empty(payload_size, dtype=np.uint8)
    comm.recv(payload, peer, tag + 1)
    return payload.tobytes()


def _all_gather_object(comm: CommGroup, obj, tag: int):
    rank = comm.my_rank
    world_size = comm.nranks
    if rank == 0:
        values = [obj]
        for peer in range(1, world_size):
            values.append(pickle.loads(_recv_bytes(comm, peer, tag)))
        payload = pickle.dumps(values)
        for peer in range(1, world_size):
            _send_bytes(comm, payload, peer, tag + 2)
        return values

    _send_bytes(comm, pickle.dumps(obj), 0, tag)
    return pickle.loads(_recv_bytes(comm, 0, tag + 2))


def _broadcast_object(comm: CommGroup, obj, src: int, tag: int):
    rank = comm.my_rank
    world_size = comm.nranks
    if rank == src:
        payload = pickle.dumps(obj)
        for peer in range(world_size):
            if peer != src:
                _send_bytes(comm, payload, peer, tag)
        return obj

    return pickle.loads(_recv_bytes(comm, src, tag))


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
    output_layout: Optional[str] = None
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
    layout: str = "flat_expert_major"


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
    layout: str
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


class ExpertParallelRuntime:
    """Low-level expert-parallel runtime wrapper.

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

        self._cpp_buffer = _cpp.ExpertParallelRuntime(
            self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode
        )

        # Exchange device IDs + IPC handles + (for RDMA) the MSCCL++ unique id.
        local_device_id = self._cpp_buffer.get_local_device_id()
        device_ids = _all_gather_object(comm, local_device_id, tag=1000)

        local_ipc_handle = self._cpp_buffer.get_local_ipc_handle()
        ipc_handles = _all_gather_object(comm, local_ipc_handle, tag=1010)

        root_unique_id: Optional[bytes] = None
        # MSCCL++ requires a bootstrapped Communicator even for pure-NVLink
        # setups because the C++ runtime sync uses `communicator->connect(ipc)`
        # to build MemoryChannels. We always exchange a unique id.
        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

        if self.rank == 0:
            root_unique_id = self._cpp_buffer.create_unique_id()
        root_unique_id = _broadcast_object(comm, root_unique_id, src=0, tag=1020)
        assert root_unique_id is not None
        self._cpp_buffer.connect(root_unique_id)

        # sync() expects Sequence[bytearray | None] / bytearray | None.
        ipc_handles_ba = [bytearray(h) if h is not None else None for h in ipc_handles]
        self._cpp_buffer.sync(device_ids, ipc_handles_ba, bytearray(root_unique_id))

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._cpp_buffer.is_available()

    def is_internode_available(self) -> bool:
        return self._cpp_buffer.is_internode_available()

    def get_local_device_id(self) -> int:
        return self._cpp_buffer.get_local_device_id()

    def get_num_rdma_ranks(self) -> int:
        return self._cpp_buffer.get_num_rdma_ranks()

    def get_rdma_rank(self) -> int:
        return self._cpp_buffer.get_rdma_rank()

    def get_root_rdma_rank(self, global_: bool) -> int:
        return self._cpp_buffer.get_root_rdma_rank(global_)

    # ------------------------------------------------------------------
    # Layout / dispatch / combine (thin pass-through wrappers).
    # These are low-level runtime APIs for compatibility tests and advanced
    # callers. New MoE code should prefer MoECommunicator.
    # ------------------------------------------------------------------

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        return self._cpp_buffer.get_dispatch_layout(
            topk_idx, num_experts, previous_event, async_finish, allocate_on_comm_stream
        )

    def intranode_dispatch(self, *args, **kwargs):
        return self._cpp_buffer.intranode_dispatch(*args, **kwargs)

    def intranode_combine(self, *args, **kwargs):
        return self._cpp_buffer.intranode_combine(*args, **kwargs)

    def internode_dispatch(self, *args, **kwargs):
        return self._cpp_buffer.internode_dispatch(*args, **kwargs)

    def internode_combine(self, *args, **kwargs):
        return self._cpp_buffer.internode_combine(*args, **kwargs)

    def get_local_buffer_tensor(
        self, dtype: torch.dtype, offset: int = 0, use_rdma_buffer: bool = False
    ) -> torch.Tensor:
        return self._cpp_buffer.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)


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

        self.output_layout = config.output_layout or "padded_expert_major"
        if self.output_layout != "padded_expert_major":
            raise NotImplementedError("low-latency mode currently supports only padded_expert_major output")

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
        self._dispatch_tokens: Optional[torch.Tensor] = None
        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._runtime = ExpertParallelRuntime(
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
    ) -> tuple[DispatchOutput, DispatchHandle]:
        self._validate_dispatch_inputs(input, topk_ids, weights, scales)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        output_tensors = self._get_dispatch_output_tensors(input.device)
        packed_tokens, packed_scales, num_tokens_per_expert, src_info, layout_range, _event, _hook = (
            self._runtime._cpp_buffer.low_latency_dispatch(
                input,
                topk_ids,
                self.max_tokens_per_rank,
                self.num_experts,
                self.dispatch_use_fp8,
                False,
                False,
                *output_tensors,
            )
        )
        output_scales = None
        if packed_scales is not None:
            output_scales = QuantScales(local=packed_scales, format="fp8_e4m3", block_size=128)

        dispatch_out = DispatchOutput(
            tokens=packed_tokens,
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

    def _get_dispatch_output_tensors(self, device: torch.device) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        token_dtype = torch.float8_e4m3fn if self.dispatch_use_fp8 else torch.bfloat16
        if (
            self._dispatch_tokens is None
            or self._dispatch_tokens.device != device
            or self._dispatch_tokens.dtype != token_dtype
        ):
            self._dispatch_tokens = torch.empty(
                (self.num_local_experts, slots_per_expert, self.hidden_size),
                dtype=token_dtype,
                device=device,
            )
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

        assert self._dispatch_tokens is not None
        assert self._dispatch_src_info is not None
        assert self._dispatch_layout_range is not None
        assert self._dispatch_count is not None
        return (
            self._dispatch_tokens,
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

        combined, _event, _hook = self._runtime._cpp_buffer.low_latency_combine(
            expert_output,
            x_scales,
            handle.topk_ids,
            handle.weights,
            handle.src_info,
            handle.layout_range,
            handle.num_max_dispatch_tokens_per_rank,
            handle.num_experts,
            False,
            False,
            False,
            out,
        )
        return combined

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
    ) -> None:
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

    def _validate_combine_inputs(
        self, expert_output: torch.Tensor, handle: DispatchHandle, out: Optional[torch.Tensor]
    ) -> None:
        if handle.num_experts != self.num_experts or handle.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        if expert_output.dim() != 3 or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous padded expert-major layout")
        expected_shape = (self.num_local_experts, self.world_size * self.max_tokens_per_rank, self.hidden_size)
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        if expert_output.dtype not in (torch.bfloat16, getattr(torch, "float8_e4m3fn", None)):
            raise ValueError("expert_output must be BF16 or FP8 E4M3")
        if out is not None:
            expected_out_shape = (handle.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")


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
