# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import Union, Tuple

import cupy as cp
import numpy as np
from mscclpp._mscclpp import CppRawGpuBuffer

__all__ = ["GpuBuffer"]


class GpuBuffer(cp.ndarray):
    def __new__(
        cls, shape: Union[int, Tuple[int]], dtype: cp.dtype = float, strides: Tuple[int] = None, order: str = "C"
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
        buffer = CppRawGpuBuffer(np.prod(shape) * np.dtype(dtype).itemsize)
        memptr = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buffer.data(), buffer.bytes(), buffer), 0)
        return cp.ndarray(shape, dtype=dtype, strides=strides, order=order, memptr=memptr)
