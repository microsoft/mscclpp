# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tensor validation and ownership helpers for the optional EP frontend."""

from __future__ import annotations

from functools import wraps
import math
from numbers import Real
from typing import Any, Iterable, Optional, Tuple, Union

import torch

from ._cpp import DispatchDataType, DispatchLayout, MoEMode, StaticFp8InputMode
from .types import MoECommunicatorConfig, QuantConfig


def requires_initialized(method):
    """Initialize the communicator once before entering an operation."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self.initialize()
        return method(self, *args, **kwargs)

    return wrapped


def resolve_num_blocks(
    num_blocks: Optional[Union[int, Tuple[Optional[int], Optional[int]]]],
    *,
    default: Tuple[int, int],
    scalar_combine_offset: int,
) -> Tuple[int, int]:
    """Resolve a scalar launch budget or a ``(dispatch, combine)`` pair."""
    if num_blocks is None:
        return default
    if type(num_blocks) is int:
        return num_blocks, num_blocks + scalar_combine_offset
    if not isinstance(num_blocks, tuple):
        raise TypeError("num_blocks must be an int or a (dispatch, combine) tuple")
    if len(num_blocks) != 2:
        raise ValueError("num_blocks must contain exactly (dispatch, combine)")
    if any(value is not None and type(value) is not int for value in num_blocks):
        raise TypeError("num_blocks tuple entries must be ints or None")
    return tuple(fallback if value is None else value for value, fallback in zip(num_blocks, default))


def resolve_dispatch_data_type(quant: Optional[QuantConfig]) -> DispatchDataType:
    """Resolve an explicit payload format without inferring or converting tensors."""
    if quant is None:
        return DispatchDataType.BF16
    if not isinstance(quant, QuantConfig):
        raise TypeError("quant must be a QuantConfig or None")
    data_type = DispatchDataType.BF16 if quant.format is None else quant.format
    if not isinstance(data_type, DispatchDataType):
        raise TypeError("quant.format must be a DispatchDataType")
    if data_type == DispatchDataType.BF16 and quant.block_scales is not None:
        raise ValueError("BF16 dispatch does not accept block_scales")
    return data_type


def validate_quant(
    quant: Optional[QuantConfig], mode: MoEMode, layout: DispatchLayout, hidden: int
) -> DispatchDataType:
    """Validate the mode-specific dispatch and combine quantization contract."""
    data_type = resolve_dispatch_data_type(quant)
    block_scales = None if quant is None else quant.block_scales
    scale = None if quant is None else quant.scale
    input_mode = StaticFp8InputMode.PREQUANTIZED_E4M3 if quant is None else quant.input_mode
    combine_type = DispatchDataType.BF16 if quant is None else quant.combine_format
    combine_scale = None if quant is None else quant.combine_scale
    if not isinstance(input_mode, StaticFp8InputMode):
        raise TypeError("quant.input_mode must be a StaticFp8InputMode")
    if not isinstance(combine_type, DispatchDataType):
        raise TypeError("quant.combine_format must be a DispatchDataType")
    if combine_type not in (DispatchDataType.BF16, DispatchDataType.FP8_E4M3):
        raise ValueError("quant.combine_format supports only BF16 or FP8_E4M3")
    static_fp8 = (
        data_type == DispatchDataType.FP8_E4M3 and mode == MoEMode.LATENCY and layout == DispatchLayout.RANK_MAJOR
    )
    if input_mode != StaticFp8InputMode.PREQUANTIZED_E4M3 and not static_fp8:
        raise ValueError("quant.input_mode is supported only by latency RANK_MAJOR static FP8")
    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise TypeError("quant.scale must be a real number")
        if not math.isfinite(float(scale)) or scale <= 0:
            raise ValueError("quant.scale must be finite and positive")
    if combine_scale is not None:
        if isinstance(combine_scale, bool) or not isinstance(combine_scale, Real):
            raise TypeError("quant.combine_scale must be a real number")
        if not math.isfinite(float(combine_scale)) or combine_scale <= 0:
            raise ValueError("quant.combine_scale must be finite and positive")
    if combine_type == DispatchDataType.FP8_E4M3:
        if mode != MoEMode.LATENCY or layout != DispatchLayout.RANK_MAJOR:
            raise ValueError("FP8 combine input requires latency RANK_MAJOR")
        if combine_scale is None:
            raise ValueError("FP8 combine input requires quant.combine_scale")
    elif combine_scale is not None:
        raise ValueError("quant.combine_scale requires FP8_E4M3 combine input")
    if data_type == DispatchDataType.BF16:
        if block_scales is not None:
            raise ValueError("BF16 dispatch does not accept block_scales")
        if scale is not None:
            raise ValueError("BF16 dispatch does not accept a scalar scale")
        return data_type
    if data_type == DispatchDataType.FP8_E4M3:
        if hidden % 128:
            raise ValueError("FP8 dispatch requires hidden_size to be a multiple of 128")
        if mode == MoEMode.LATENCY and layout == DispatchLayout.RANK_MAJOR:
            if block_scales is not None:
                raise ValueError("latency RANK_MAJOR static FP8 does not accept block_scales")
            if scale is None:
                raise ValueError("latency RANK_MAJOR FP8 requires quant.scale")
        elif scale is not None:
            raise ValueError("quant.scale is supported only by latency RANK_MAJOR FP8")
        if mode == MoEMode.LATENCY and layout != DispatchLayout.RANK_MAJOR and block_scales is not None:
            raise ValueError("latency EXPERT_MAJOR quantizes BF16 input; precomputed block_scales are unsupported")
    elif data_type == DispatchDataType.MXFP8_E4M3:
        if mode != MoEMode.LATENCY or layout != DispatchLayout.RANK_MAJOR:
            raise ValueError("MXFP8 dispatch requires latency RANK_MAJOR")
        if hidden % 32:
            raise ValueError("MXFP8 dispatch requires hidden_size to be a multiple of 32")
        if block_scales is not None:
            raise ValueError("latency MXFP8 dispatch quantizes BF16 input; precomputed block_scales are unsupported")
        if scale is not None:
            raise ValueError("MXFP8 dispatch does not accept a scalar scale")
    return data_type


def resolve_dispatch_format(config: MoECommunicatorConfig, quant: Optional[QuantConfig]) -> Tuple[
    DispatchDataType,
    Optional[torch.Tensor],
    Optional[float],
    StaticFp8InputMode,
    DispatchDataType,
    Optional[float],
]:
    """Resolve a per-call format override or the communicator's quantization default."""
    quant = config.quant if quant is None else quant
    data_type = validate_quant(quant, config.mode, config.output_layout, config.hidden_size)
    return (
        data_type,
        None if quant is None else quant.block_scales,
        None if quant is None or quant.scale is None else float(quant.scale),
        StaticFp8InputMode.PREQUANTIZED_E4M3 if quant is None else quant.input_mode,
        DispatchDataType.BF16 if quant is None else quant.combine_format,
        None if quant is None or quant.combine_scale is None else float(quant.combine_scale),
    )


def verify_active_capacity(active: Optional[int], capacity: int) -> int:
    """Verify active capacity against the configured allocation limit."""
    active = capacity if active is None else active
    if type(active) is not int or not 0 < active <= capacity:
        raise ValueError("runtime_max_tokens_per_rank must be positive and not exceed configured capacity")
    return active


def dispatch_shape(config: MoECommunicatorConfig, num_ranks: int, active: int) -> tuple:
    """Return the dispatch shape for a resolved configuration and active capacity."""
    if config.output_layout == DispatchLayout.EXPERT_MAJOR:
        return config.num_local_experts, num_ranks * active, config.hidden_size
    if config.output_layout == DispatchLayout.RANK_MAJOR:
        return num_ranks, active, config.hidden_size
    return num_ranks * active, config.hidden_size


def payload_dtype(data_type: DispatchDataType) -> torch.dtype:
    """Return the PyTorch dtype of the dispatched payload."""
    return torch.bfloat16 if data_type == DispatchDataType.BF16 else torch.float8_e4m3fn


def ptr(tensor: Optional[torch.Tensor]) -> int:
    """Return a tensor's device pointer, or the native null pointer for ``None``."""
    return 0 if tensor is None else tensor.data_ptr()


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: Tuple[Optional[int], ...],
    dtype: torch.dtype,
    device: torch.device,
    alignment: int,
) -> None:
    """Check host-visible tensor metadata before submitting communication work."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(f"{name} must not have unresolved conjugate or negative view bits")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(tensor.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    pointer = tensor.data_ptr()
    if tensor.numel() and not pointer:
        raise ValueError(f"{name} must have allocated CUDA storage")
    if pointer % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def record_stream(tensors: Iterable[Optional[torch.Tensor]], stream: torch.cuda.Stream) -> None:
    """Protect allocator-backed storage used by native asynchronous GPU work.

    PyTorch's allocator ignores foreign storage. Runtime buffers instead rely on
    their native owner and the caller's obligation to finish local and peer use.
    This does not provide producer/consumer ordering across streams.
    """
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


class DevicePointerArray:
    """CUDA array interface whose storage owner is retained by PyTorch's deleter."""

    def __init__(self, pointer: int, shape: Tuple[int, ...], typestr: str, owner: Any) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {
            "data": (pointer, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


def tensor_from_pointer(
    pointer: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    owner: Any,
) -> Tuple[DevicePointerArray, torch.Tensor]:
    """Make a zero-copy CUDA view whose underlying storage keeps ``owner`` alive.

    ``torch.as_tensor`` retains the CUDA array-interface object in its storage
    deleter. Consequently slices and dtype/shape views retain the native owner,
    without attaching fragile attributes to individual tensors or forming cycles.
    """
    storage_types = {
        torch.bfloat16: "<i2",
        torch.float8_e4m3fn: "|u1",
        torch.uint8: "|u1",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.float32: "<f4",
    }
    if dtype not in storage_types:
        raise ValueError(f"unsupported CUDA pointer view dtype: {dtype}")
    if not pointer:
        raise RuntimeError("the native runtime did not provide an initialized buffer")
    buffer_view = DevicePointerArray(pointer, shape, storage_types[dtype], owner)
    tensor = torch.as_tensor(buffer_view, device=device).view(dtype)
    return buffer_view, tensor
