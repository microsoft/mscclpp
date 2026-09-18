# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tensor validation and ownership helpers for the optional EP frontend."""

from __future__ import annotations

from functools import wraps
from typing import Any, Iterable, Optional, Tuple, Union

import torch

from ._cpp import DispatchDataType
from .types import QuantConfig


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
