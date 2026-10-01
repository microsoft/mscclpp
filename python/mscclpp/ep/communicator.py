# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyTorch tensor boundary for the native expert-parallel runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch

from mscclpp import CommGroup

from . import _cpp
from ._cpp import CombineMode, DispatchLayout, MoEMode
from .request_builder import RequestBuilder
from .types import (
    DispatchHandle,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    PrepareHandle,
    QuantConfig,
)
from .utils import (
    dispatch_shape,
    payload_dtype,
    record_stream,
    requires_initialized,
    resolve_dispatch_format,
    resolve_num_blocks,
    tensor_from_pointer,
    validate_quant,
    verify_active_capacity,
)


class MoECommunicator:
    """Collective MoE dispatch/combine on one caller-owned CUDA stream.

    LATENCY defaults to EXPERT_MAJOR; THROUGHPUT defaults to TOKEN_MAJOR.
    Construction and collective initialization must precede graph capture.
    Operations otherwise initialize lazily. Host calls must be serialized, and
    expert computation must use the same stream as communication. This interface
    supplies no autograd backward.
    """

    def __init__(self, config: MoECommunicatorConfig) -> None:
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
        self._request_builder = RequestBuilder(self._config, self._runtime)

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
        active = verify_active_capacity(runtime_max_tokens_per_rank, self.max_tokens_per_rank)
        data_type, _ = resolve_dispatch_format(self._config, quant)
        with torch.cuda.device(self.device):
            return tensor_from_pointer(
                self._runtime.dispatch_output_buffer_ptr(),
                dispatch_shape(self._config, self.world_size, active),
                payload_dtype(data_type),
                self.device,
                self._runtime,
            )[1]

    @requires_initialized
    def prepare(
        self,
        topk_ids: torch.Tensor,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        runtime_max_tokens_per_rank: Optional[int] = None,
        output_count: Optional[torch.Tensor] = None,
    ) -> PrepareHandle:
        """Prepare throughput routing entirely on the GPU, without moving payloads.

        Optional ``output_count`` receives GPU counts with shape
        ``[num_local_experts]`` for TOKEN_MAJOR or ``[world_size]`` for
        RANK_MAJOR.
        Reuse the returned handle only with unchanged routing, token count,
        active capacity, and stream. Every rank must make the same reuse choice.
        """
        caller_stream = self._resolve_stream(stream)
        with torch.cuda.stream(caller_stream):
            arguments = self._request_builder.build_prepare(
                topk_ids, caller_stream, runtime_max_tokens_per_rank, output_count
            )
            record_stream((topk_ids, output_count), caller_stream)
            native = self._runtime.prepare(**arguments)
            self._bind_stream(caller_stream)
        return PrepareHandle(
            _native=native,
            _runtime=self._runtime,
            _topk_ids=topk_ids,
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
        Only latency EXPERT_MAJOR accepts an external ``output_buffer``; other
        layouts require the view returned by ``get_dispatch_output_buffer``.
        Inputs must not overlap the output or runtime receive storage written
        by this operation. Aliasing is a caller precondition and is not checked.
        """
        if prepare_handle is not None:
            if self.mode != MoEMode.THROUGHPUT:
                raise ValueError("prepare_handle is supported only in THROUGHPUT mode")
            self._validate_handle(prepare_handle, PrepareHandle)
        caller_stream = self._resolve_stream(stream)

        with torch.cuda.stream(caller_stream):
            if self.mode == MoEMode.LATENCY:
                request = self._request_builder.build_latency_dispatch(
                    input, topk_ids, weights, quant, output_buffer, caller_stream, runtime_max_tokens_per_rank
                )
                launch = self._runtime.dispatch_latency
            else:
                request = self._request_builder.build_throughput_dispatch(
                    input,
                    topk_ids,
                    weights,
                    quant,
                    output_buffer,
                    caller_stream,
                    prepare_handle,
                    runtime_max_tokens_per_rank,
                )
                launch = self._runtime.dispatch_throughput
            record_stream(request.tensors, caller_stream)
            native = launch(**request.arguments)
            self._bind_stream(caller_stream)

        handle = DispatchHandle(
            output_info=DispatchOutputInfo(layout=request.output.layout, quant=request.output.quant),
            _native=native,
            _runtime=self._runtime,
            _num_tokens=request.num_tokens,
            _active_capacity=request.active_capacity,
            _tensors=request.tensors,
            _stream=caller_stream,
        )
        return request.output, handle

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        apply_router_weights: Optional[bool] = None,
    ) -> torch.Tensor:
        """Combine BF16 expert outputs into ``[local_num_tokens, hidden_size]``.

        EXPERT_MAJOR and latency RANK_MAJOR DIRECT_SEND apply routing weights
        natively. The latter expects unweighted per-topk rows; other rank/token
        layouts expect already-weighted local expert sums. Non-EXPERT_MAJOR
        inputs must use ``DispatchOutput.combine_input_buffer``.
        For latency DIRECT_SEND, set ``apply_router_weights=False`` for
        preweighted expert outputs. By default it is enabled only for
        DIRECT_SEND; other modes ignore this option.
        There is no implicit copy from external expert-output tensors.
        The output must not overlap expert inputs or runtime combine storage.
        The caller must ensure these conditions; buffer aliasing is not checked.
        """
        self._validate_handle(handle, DispatchHandle)
        caller_stream = self._resolve_stream(stream)
        with torch.cuda.stream(caller_stream):
            if self.mode == MoEMode.LATENCY:
                request = self._request_builder.build_latency_combine(
                    expert_output, handle, out, caller_stream, apply_router_weights
                )
                launch = self._runtime.combine_latency
            else:
                request = self._request_builder.build_throughput_combine(expert_output, handle, out, caller_stream)
                launch = self._runtime.combine_throughput
            record_stream((*handle._tensors, expert_output, request.output), caller_stream)
            launch(**request.arguments)
            self._bind_stream(caller_stream)
        return request.output

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
        if handle._runtime is not self._runtime:
            raise ValueError(f"{expected_type.__name__} belongs to another MoECommunicator")
        if isinstance(handle, DispatchHandle) and (
            type(handle._num_tokens) is not int
            or type(handle._active_capacity) is not int
            or not 0 < handle._active_capacity <= self.max_tokens_per_rank
            or not 0 <= handle._num_tokens <= handle._active_capacity
        ):
            raise ValueError(f"{expected_type.__name__} has an invalid token count or capacity")


def _resolve_config(config: MoECommunicatorConfig) -> MoECommunicatorConfig:
    if not isinstance(config, MoECommunicatorConfig):
        raise TypeError("config must be a MoECommunicatorConfig")
    if config.comm is None:
        raise ValueError("MoECommunicator requires an mscclpp.CommGroup via config.comm")
    if not isinstance(config.comm, CommGroup):
        raise TypeError("comm must be an mscclpp.CommGroup")
    latency = config.mode == MoEMode.LATENCY
    layout = config.output_layout
    if layout is None:
        layout = DispatchLayout.EXPERT_MAJOR if latency else DispatchLayout.TOKEN_MAJOR
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
    if not latency and num_local > 128:
        raise ValueError("throughput requires at most 128 experts per rank")
    invalid_id = config.invalid_token_expert_id
    if not latency and invalid_id is not None:
        raise ValueError("invalid_token_expert_id is supported only in LATENCY mode; throughput uses -1")
    if invalid_id is None:
        invalid_id = config.num_experts if latency else -1
    if type(invalid_id) is not int or not -(1 << 31) <= invalid_id < (1 << 31):
        raise ValueError("invalid_token_expert_id must fit in int32")
    if 0 <= invalid_id < config.num_experts:
        raise ValueError("invalid_token_expert_id must not overlap a valid global expert ID")
    validate_quant(config.quant, config.mode, layout, config.hidden_size)
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
