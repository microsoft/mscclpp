# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import Union, Tuple

import cupy as cp
import numpy as np
from mscclpp._mscclpp import CppRawGpuBuffer, CppGpuBufferGranularity

__all__ = ["GpuBuffer", "GpuBufferGranularity"]

GpuBufferGranularity = CppGpuBufferGranularity


class GpuBuffer(cp.ndarray):
    def __new__(
        cls,
        shape: Union[int, Tuple[int]],
        dtype: cp.dtype = float,
        strides: Tuple[int] = None,
        order: str = "C",
        granularity: CppGpuBufferGranularity = CppGpuBufferGranularity.MultiCastMinimum,
    ):
        # Check if `shape` is valid
        if isinstance(shape, int):
            shape = (shape,)
        try:
            shape = tuple(shape)
        except TypeError:
            raise ValueError("Shape must be a tuple-like or an integer.")
        if any(s <= 0 for s in shape):
            raise ValueError("Shape must be positive.")
        # Create the buffer
        buffer = CppRawGpuBuffer(np.prod(shape) * np.dtype(dtype).itemsize, granularity)
        memptr = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buffer.data(), buffer.bytes(), buffer), 0)
        return cp.ndarray(shape, dtype=dtype, strides=strides, order=order, memptr=memptr)
