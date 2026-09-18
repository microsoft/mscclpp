# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyTorch tensor boundary for the native expert-parallel runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch

from mscclpp import CommGroup

from . import _cpp
from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    PrepareHandle,
    QuantConfig,
)
from .utils import (
    check_tensor,
    ptr,
    record_stream,
    requires_initialized,
    resolve_dispatch_data_type,
    resolve_num_blocks,
    tensor_from_pointer,
)


class MoECommunicator:
    """Collective MoE dispatch/combine on one caller-owned CUDA stream.

    LATENCY defaults to EXPERT_MAJOR; THROUGHPUT defaults to TOKEN_MAJOR.
    Construction and collective initialization must precede graph capture.
    Operations otherwise initialize lazily. Host calls must be serialized, and
    expert computation must use the same stream as communication. This interface
    supplies no autograd backward.
    """

    def __init__(self, config: Optional[MoECommunicatorConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("pass either MoECommunicatorConfig or keyword arguments, not both")
        if config is None:
            config = MoECommunicatorConfig(**kwargs)
        self._config = _resolve_config(config)
        self._initialized = False
        self._stream: Optional[torch.cuda.Stream] = None
        with torch.cuda.device(self.device):
            self._runtime = _cpp.create_moe_runtime(
                comm=self.comm.communicator,
                mode=self.mode,
                max_tokens_per_rank=self.max_tokens_per_rank,
                hidden=self.hidden_size,
                num_experts=self.num_experts,
                num_topk=self.topk,
                output_layout=self.output_layout,
                combine_mode=self.combine_mode,
            )
        self._signature = (
            self.device,
            self.mode,
            self.output_layout,
            self.num_experts,
            self.hidden_size,
            self.topk,
            self.max_tokens_per_rank,
            self.num_blocks,
            self.combine_mode,
            self.invalid_token_expert_id,
        )

    @property
    def comm(self) -> CommGroup:
        return self._config.comm

    @property
    def rank(self) -> int:
        return self._runtime.rank

    @property
    def world_size(self) -> int:
        return self._runtime.num_ranks

    @property
    def num_ranks_per_ipc_domain(self) -> int:
        return self._runtime.num_ranks_per_ipc_domain

    @property
    def local_rank(self) -> int:
        return self.device.index

    @property
    def device(self) -> torch.device:
        return self._config.device

    @property
    def mode(self) -> MoEMode:
        return self._config.mode

    @property
    def output_layout(self) -> DispatchLayout:
        return self._config.output_layout

    @property
    def num_experts(self) -> int:
        return self._config.num_experts

    @property
    def num_local_experts(self) -> int:
        return self._config.num_local_experts

    @property
    def local_expert_start(self) -> int:
        return self._config.local_expert_start

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def topk(self) -> int:
        return self._config.topk

    @property
    def max_tokens_per_rank(self) -> int:
        return self._config.max_tokens_per_rank

    @property
    def num_blocks(self) -> Tuple[int, int]:
        """Resolved total grid sizes for dispatch and combine, respectively."""
        return self._config.num_blocks

    @property
    def combine_mode(self) -> CombineMode:
        return self._config.combine_mode

    @property
    def invalid_token_expert_id(self) -> int:
        return self._config.invalid_token_expert_id

    def is_available(self) -> bool:
        """Return native support for this runtime's mode and topology."""
        return self._runtime.is_available()

    def is_initialized(self) -> bool:
        """Return whether collective resource initialization has completed."""
        return self._initialized

    def initialize(self) -> None:
        """Collectively initialize once, outside capture; later calls are no-ops."""
        if self._initialized:
            return
        with torch.cuda.device(self.device):
            self._runtime.initialize()
            self._initialized = True

    @requires_initialized
    def get_dispatch_output_buffer(
        self,
        *,
        quant: Optional[QuantConfig] = None,
        runtime_max_tokens_per_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the runtime's dispatch storage with the requested active strides.

        ``quant`` selects the view's format, defaulting to the communicator's
        format. It does not perform quantization. The allocation is reused by
        subsequent operations; a view does not reserve a separate output slot.
        """
        active = self._capacity(runtime_max_tokens_per_rank)
        data_type, _ = self._dispatch_format(quant)
        with torch.cuda.device(self.device):
            return self._view(
                self._runtime.dispatch_output_buffer_ptr(), self._dispatch_shape(active), _payload_dtype(data_type)
            )

    @requires_initialized
    def prepare(
        self,
        topk_ids: torch.Tensor,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        runtime_max_tokens_per_rank: Optional[int] = None,
    ) -> PrepareHandle:
        """Prepare throughput routing entirely on the GPU, without moving payloads.

        Reuse the returned handle only with unchanged routing, token count,
        active capacity, and stream. Every rank must make the same reuse choice.
        """
        if self.mode != MoEMode.THROUGHPUT:
            raise ValueError("prepare() is supported only in THROUGHPUT mode")
        active = self._capacity(runtime_max_tokens_per_rank)
        self._check(topk_ids, "topk_ids", (None, self.topk), torch.int64, 8)
        num_tokens = topk_ids.shape[0]
        if num_tokens > active:
            raise ValueError("topk_ids token count exceeds runtime_max_tokens_per_rank")
        caller_stream = self._resolve_stream(stream)
        with torch.cuda.stream(caller_stream):
            record_stream((topk_ids,), caller_stream)
            native = self._runtime.prepare(
                topk_idx_ptr=ptr(topk_ids),
                num_tokens=num_tokens,
                max_tokens_per_rank=active,
                num_blocks=self.num_blocks[0],
                stream_ptr=caller_stream.cuda_stream,
            )
            self._bind_stream(caller_stream)
        return PrepareHandle(
            _native=native,
            _runtime=self._runtime,
            _config=self._signature,
            _topk_ids=topk_ids,
            _topk_ptr=ptr(topk_ids),
            _num_tokens=num_tokens,
            _active_capacity=active,
            _stream=caller_stream,
        )

    @requires_initialized
    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        quant: Optional[QuantConfig] = None,
        *,
        output_buffer: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        prepare_handle: Optional[PrepareHandle] = None,
        runtime_max_tokens_per_rank: Optional[int] = None,
    ) -> Tuple[DispatchOutput, DispatchHandle]:
        """Dispatch tokens and return capacity-sized GPU data plus a combine handle.

        Input IDs are contiguous CUDA int64 ``[num_tokens, topk]`` containing
        global expert IDs or negative dropped routes. Optional weights are
        CUDA FP32 with the same shape; omitted weights mean one per valid route.
        Routing values are not copied to or inspected on the host.
        Inputs must not overlap the output or runtime receive storage written
        by this operation. Aliasing is a caller precondition and is not checked.
        """
        active = self._capacity(runtime_max_tokens_per_rank)
        data_type, input_scales = self._dispatch_format(quant)
        output_dtype = _payload_dtype(data_type)
        input_dtype = torch.bfloat16 if self.mode == MoEMode.LATENCY else output_dtype
        self._check(input, "input", (None, self.hidden_size), input_dtype)
        num_tokens = input.shape[0]
        if num_tokens > active:
            raise ValueError("input token count exceeds runtime_max_tokens_per_rank")
        self._check(topk_ids, "topk_ids", (num_tokens, self.topk), torch.int64, 8)
        if weights is not None:
            self._check(weights, "weights", (num_tokens, self.topk), torch.float32, 4)
        if self.mode == MoEMode.THROUGHPUT and data_type == DispatchDataType.FP8_E4M3:
            if input_scales is None:
                raise ValueError("throughput FP8 dispatch requires quant.block_scales")
            self._check(input_scales, "quant.block_scales", (num_tokens, self.hidden_size // 128), torch.float32, 4)
        output_shape = self._dispatch_shape(active)
        if output_buffer is not None:
            self._check(output_buffer, "output_buffer", output_shape, output_dtype)
        if prepare_handle is not None:
            if self.mode != MoEMode.THROUGHPUT:
                raise ValueError("prepare_handle is supported only in THROUGHPUT mode")
            self._validate_handle(prepare_handle, PrepareHandle)
            if (
                prepare_handle._topk_ptr != ptr(topk_ids)
                or prepare_handle._num_tokens != num_tokens
                or prepare_handle._active_capacity != active
            ):
                raise ValueError("prepare_handle routing pointer, token count, and active capacity must match dispatch")
        caller_stream = self._resolve_stream(stream)

        with torch.cuda.stream(caller_stream):
            runtime_output_ptr = self._runtime.dispatch_output_buffer_ptr()
            if (
                self.mode == MoEMode.LATENCY
                and self.output_layout == DispatchLayout.RANK_MAJOR
                and output_buffer is not None
                and ptr(output_buffer) != runtime_output_ptr
            ):
                raise ValueError("latency RANK_MAJOR output_buffer must be the runtime-owned dispatch buffer")
            tokens = (
                self._view(runtime_output_ptr, output_shape, output_dtype)
                if output_buffer is None or ptr(output_buffer) == runtime_output_ptr
                else output_buffer
            )
            rank_major = self.output_layout == DispatchLayout.RANK_MAJOR
            count = torch.empty(
                (self.world_size if rank_major else self.num_local_experts,), dtype=torch.int32, device=self.device
            )
            src_info = layout_range = recv_ids = recv_weights = scales = None
            if self.output_layout == DispatchLayout.EXPERT_MAJOR:
                src_info = torch.empty(output_shape[:-1], dtype=torch.int32, device=self.device)
                # Each entry packs the source rank's count and offset for one local expert.
                layout_range = torch.empty(
                    (self.num_local_experts, self.world_size), dtype=torch.int64, device=self.device
                )
            else:
                metadata_shape = output_shape[:-1] + (self.topk,)
                if self.mode == MoEMode.LATENCY:
                    recv_ids = self._view(self._runtime.output_topk_ids_buffer_ptr(), metadata_shape, torch.int32)
                    recv_weights = self._view(
                        self._runtime.output_topk_weights_buffer_ptr(), metadata_shape, torch.float32
                    )
                else:
                    recv_ids = torch.empty(metadata_shape, dtype=torch.int32, device=self.device)
                    recv_weights = torch.empty(metadata_shape, dtype=torch.float32, device=self.device)
            if data_type == DispatchDataType.FP8_E4M3:
                num_scales = self.hidden_size // 128
                if self.mode == MoEMode.LATENCY:
                    scales = torch.empty(
                        (self.num_local_experts, num_scales, self.world_size * active),
                        dtype=torch.float32,
                        device=self.device,
                    ).transpose(1, 2)
                else:
                    scales = torch.empty(output_shape[:-1] + (num_scales,), dtype=torch.float32, device=self.device)
            combine_input = (
                None
                if self.output_layout == DispatchLayout.EXPERT_MAJOR
                else self._view(self._runtime.combine_input_buffer_ptr(), self._combine_shape(active), torch.bfloat16)
            )
            retained = tuple(
                tensor
                for tensor in (
                    input,
                    topk_ids,
                    weights,
                    input_scales,
                    tokens,
                    scales,
                    src_info,
                    layout_range,
                    recv_ids,
                    recv_weights,
                    count,
                    combine_input,
                )
                if tensor is not None
            )
            record_stream(retained, caller_stream)
            if self.mode == MoEMode.LATENCY:
                native = self._runtime.dispatch_latency(
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
                    invalid_token_expert_id=self.invalid_token_expert_id,
                    dispatch_data_type=data_type,
                    num_blocks=self.num_blocks[0],
                    stream_ptr=caller_stream.cuda_stream,
                )
            else:
                native = self._runtime.dispatch_throughput(
                    input_ptr=ptr(input),
                    input_scales_ptr=ptr(input_scales),
                    topk_idx_ptr=ptr(topk_ids),
                    topk_weights_ptr=ptr(weights),
                    output_ptr=ptr(tokens),
                    output_scales_ptr=ptr(scales),
                    output_topk_idx_ptr=ptr(recv_ids),
                    output_topk_weights_ptr=ptr(recv_weights),
                    output_count_ptr=ptr(count),
                    num_tokens=num_tokens,
                    max_tokens_per_rank=active,
                    dispatch_data_type=data_type,
                    num_blocks=self.num_blocks[0],
                    stream_ptr=caller_stream.cuda_stream,
                    prepare_handle=_cpp.PrepareHandle() if prepare_handle is None else prepare_handle._native,
                )
            self._bind_stream(caller_stream)
            layout = DispatchLayoutInfo(
                kind=self.output_layout,
                num_tokens_per_expert=None if rank_major else count,
                num_tokens_per_rank=count if rank_major else None,
            )
            output_quant = None if scales is None else QuantConfig(format=data_type, block_scales=scales)
            output_info = DispatchOutputInfo(layout=layout, quant=output_quant)

        output = DispatchOutput(
            tokens=tokens,
            quant=output_quant,
            layout=layout,
            topk_ids=recv_ids,
            weights=recv_weights,
            combine_input_buffer=combine_input,
        )
        handle = DispatchHandle(
            output_info=output_info,
            _native=native,
            _runtime=self._runtime,
            _config=self._signature,
            _num_tokens=num_tokens,
            _active_capacity=active,
            _tensors=retained,
            _stream=caller_stream,
        )
        return output, handle

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        output_topk_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine BF16 expert outputs into ``[local_num_tokens, hidden_size]``.

        EXPERT_MAJOR applies the dispatch's original weights natively. Other
        layouts require already-weighted local expert sums (per-topk results
        for latency DIRECT_SEND). Throughput optionally writes combined weights
        into ``output_topk_weights``, a CUDA FP32 ``[local_num_tokens, topk]`` tensor.
        Outputs must not overlap each other, expert inputs, or runtime combine
        storage. Expert inputs may exactly reuse ``combine_input_buffer``, but
        partial overlap is unsupported. The caller must ensure these conditions;
        buffer aliasing is not checked.
        """
        self._validate_handle(handle, DispatchHandle)
        self._check(expert_output, "expert_output", self._combine_shape(handle._active_capacity), torch.bfloat16)
        output_shape = (handle._num_tokens, self.hidden_size)
        if out is not None:
            self._check(out, "out", output_shape, torch.bfloat16)
        if output_topk_weights is not None:
            if self.mode != MoEMode.THROUGHPUT:
                raise ValueError("output_topk_weights is supported only in THROUGHPUT mode")
            self._check(output_topk_weights, "output_topk_weights", (handle._num_tokens, self.topk), torch.float32, 4)
        caller_stream = self._resolve_stream(stream)
        with torch.cuda.stream(caller_stream):
            if out is None:
                out = torch.empty(output_shape, dtype=torch.bfloat16, device=self.device)
            record_stream((*handle._tensors, expert_output, out, output_topk_weights), caller_stream)
            if self.mode == MoEMode.LATENCY:
                self._runtime.combine_latency(
                    expert_output_ptr=ptr(expert_output),
                    output_ptr=ptr(out),
                    handle=handle._native,
                    num_blocks=self.num_blocks[1],
                    stream_ptr=caller_stream.cuda_stream,
                )
            else:
                self._runtime.combine_throughput(
                    expert_output_ptr=ptr(expert_output),
                    output_ptr=ptr(out),
                    output_topk_weights_ptr=ptr(output_topk_weights),
                    handle=handle._native,
                    num_blocks=self.num_blocks[1],
                    stream_ptr=caller_stream.cuda_stream,
                )
            self._bind_stream(caller_stream)
        return out

    def _capacity(self, active: Optional[int]) -> int:
        active = self.max_tokens_per_rank if active is None else active
        if type(active) is not int or not 0 < active <= self.max_tokens_per_rank:
            raise ValueError("runtime_max_tokens_per_rank must be positive and not exceed configured capacity")
        return active

    def _dispatch_shape(self, active: int) -> tuple:
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            return self.num_local_experts, self.world_size * active, self.hidden_size
        if self.output_layout == DispatchLayout.RANK_MAJOR:
            return self.world_size, active, self.hidden_size
        return self.world_size * active, self.hidden_size

    def _combine_shape(self, active: int) -> tuple:
        if (
            self.mode == MoEMode.LATENCY
            and self.output_layout == DispatchLayout.RANK_MAJOR
            and self.combine_mode == CombineMode.DIRECT_SEND
        ):
            return self.world_size, active, self.topk, self.hidden_size
        return self._dispatch_shape(active)

    def _dispatch_format(self, quant: Optional[QuantConfig]) -> Tuple[DispatchDataType, Optional[torch.Tensor]]:
        quant = self._config.quant if quant is None else quant
        data_type = _validate_quant(quant, self.mode, self.output_layout, self.hidden_size)
        scales = None if quant is None else quant.block_scales
        return data_type, scales

    def _check(self, tensor, name, shape, dtype, alignment=16) -> None:
        check_tensor(tensor, name, shape=shape, dtype=dtype, device=self.device, alignment=alignment)

    def _view(self, pointer: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        return tensor_from_pointer(pointer, shape, dtype, self.device, self._runtime)[1]

    def _resolve_stream(self, stream: Optional[torch.cuda.Stream]) -> torch.cuda.Stream:
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be a torch.cuda.Stream or None")
        if stream.device != self.device:
            raise ValueError(f"stream must be on {self.device}, got {stream.device}")
        if self._stream is not None and stream.cuda_stream != self._stream.cuda_stream:
            raise ValueError(
                "this MoECommunicator is bound to another CUDA stream; "
                "prepare, dispatch, expert compute, and combine must use the same caller stream"
            )
        return stream

    def _bind_stream(self, stream: torch.cuda.Stream) -> None:
        if self._stream is None:
            self._stream = stream

    def _validate_handle(self, handle: Union[PrepareHandle, DispatchHandle], expected_type: type) -> None:
        if not isinstance(handle, expected_type):
            raise TypeError(f"handle must be a Python {expected_type.__name__}")
        if handle._runtime is not self._runtime or handle._config != self._signature:
            raise ValueError(f"{expected_type.__name__} belongs to another MoECommunicator or configuration")
        native_type = _cpp.PrepareHandle if expected_type is PrepareHandle else _cpp.DispatchHandle
        if not isinstance(handle._native, native_type):
            raise ValueError(f"{expected_type.__name__} does not contain a native handle")
        if (
            type(handle._num_tokens) is not int
            or type(handle._active_capacity) is not int
            or not 0 < handle._active_capacity <= self.max_tokens_per_rank
            or not 0 <= handle._num_tokens <= handle._active_capacity
        ):
            raise ValueError(f"{expected_type.__name__} has an invalid token count or capacity")
        if isinstance(handle, DispatchHandle):
            if (
                not isinstance(handle.output_info, DispatchOutputInfo)
                or not isinstance(handle.output_info.layout, DispatchLayoutInfo)
                or handle.output_info.layout.kind != self.output_layout
            ):
                raise ValueError("DispatchHandle output layout does not match this MoECommunicator")


def _payload_dtype(data_type: DispatchDataType) -> torch.dtype:
    return torch.bfloat16 if data_type == DispatchDataType.BF16 else torch.float8_e4m3fn


def _validate_quant(quant, mode, layout, hidden) -> DispatchDataType:
    data_type = resolve_dispatch_data_type(quant)
    if mode == MoEMode.LATENCY and quant is not None and quant.block_scales is not None:
        raise ValueError("latency dispatch quantizes BF16 input; precomputed block_scales are unsupported")
    if data_type == DispatchDataType.FP8_E4M3:
        if hidden % 128:
            raise ValueError("FP8 dispatch requires hidden_size to be a multiple of 128")
        if mode == MoEMode.LATENCY and layout == DispatchLayout.RANK_MAJOR:
            raise ValueError("latency RANK_MAJOR supports BF16 dispatch only")
    return data_type


def _resolve_config(config: MoECommunicatorConfig) -> MoECommunicatorConfig:
    if not isinstance(config, MoECommunicatorConfig):
        raise TypeError("config must be a MoECommunicatorConfig")
    if config.comm is None:
        raise ValueError("MoECommunicator requires an mscclpp.CommGroup via comm=")
    if not isinstance(config.comm, CommGroup):
        raise TypeError("comm must be an mscclpp.CommGroup")
    for name in ("num_experts", "hidden_size", "topk", "max_tokens_per_rank"):
        value = getattr(config, name)
        if type(value) is not int or not 0 < value < (1 << 31):
            raise ValueError(f"{name} must be a positive int32")
    if config.topk > 8:
        raise ValueError("topk must be in [1, 8]")
    if not isinstance(config.mode, MoEMode):
        raise TypeError("mode must be a MoEMode")
    if not isinstance(config.combine_mode, CombineMode):
        raise TypeError("combine_mode must be a CombineMode")
    latency = config.mode == MoEMode.LATENCY
    layout = config.output_layout
    if layout is None:
        layout = DispatchLayout.EXPERT_MAJOR if latency else DispatchLayout.TOKEN_MAJOR
    if not isinstance(layout, DispatchLayout):
        raise TypeError("output_layout must be a DispatchLayout")
    supported = (
        (DispatchLayout.EXPERT_MAJOR, DispatchLayout.RANK_MAJOR)
        if latency
        else (DispatchLayout.TOKEN_MAJOR, DispatchLayout.RANK_MAJOR)
    )
    if layout not in supported:
        raise ValueError("output_layout is unsupported for the selected mode")
    if not latency and config.combine_mode != CombineMode.RANK_LOCAL_REDUCE:
        raise ValueError("THROUGHPUT supports only RANK_LOCAL_REDUCE combine")
    world_size = config.comm.nranks
    if not 1 <= world_size <= 64 or config.comm.nranks_per_ipc_domain != world_size:
        raise ValueError("EP requires 1-64 ranks in one CUDA IPC domain")
    if config.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    num_local = config.num_experts // world_size
    start = config.comm.my_rank * num_local
    for name, expected in (("num_local_experts", num_local), ("local_expert_start", start)):
        value = getattr(config, name)
        if value is not None and (type(value) is not int or value != expected):
            raise ValueError(f"{name} must be {expected}; only even contiguous expert placement is supported")
    if latency:
        if config.hidden_size not in (4096, 4352, 6656, 7168, 8192, 8704, 9216):
            raise ValueError("latency hidden_size must be one of 4096, 4352, 6656, 7168, 8192, 8704, 9216")
    elif num_local > 128 or config.hidden_size % 8:
        raise ValueError("throughput requires at most 128 experts per rank and 16-byte-aligned BF16 rows")
    invalid_id = config.invalid_token_expert_id
    if not latency and invalid_id is not None:
        raise ValueError("invalid_token_expert_id is supported only in LATENCY mode; throughput uses -1")
    if invalid_id is None:
        invalid_id = config.num_experts if latency else -1
    if type(invalid_id) is not int or not -(1 << 31) <= invalid_id < (1 << 31):
        raise ValueError("invalid_token_expert_id must fit in int32")
    if 0 <= invalid_id < config.num_experts:
        raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
    _validate_quant(config.quant, config.mode, layout, config.hidden_size)
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise RuntimeError("MSCCL++ EP requires CUDA and an SM90 or newer GPU")
    device = config.device
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif type(device) is int:
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 9:
        raise RuntimeError("MSCCL++ EP requires an SM90 or newer GPU")
    num_sms = properties.multi_processor_count
    blocks = resolve_num_blocks(
        config.num_blocks,
        default=(min(130, num_sms), min(128, num_sms)) if latency else (min(24, num_sms), min(32, num_sms)),
        scalar_combine_offset=-2 if latency else 0,
    )
    if not (world_size + 2 if latency else 1) <= blocks[0] <= 130 or not 1 <= blocks[1] <= 128:
        raise ValueError(
            "num_blocks must satisfy dispatch <= 130 and combine <= 128; "
            "latency dispatch requires at least world_size + 2 blocks, and all counts must be positive"
        )
    return replace(
        config,
        device=device,
        output_layout=layout,
        num_local_experts=num_local,
        local_expert_start=start,
        invalid_token_expert_id=invalid_id,
        num_blocks=blocks,
        quant=None if config.quant is None else replace(config.quant),
    )
