# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import Optional, Union, Tuple

import cupy as cp
import numpy as np
from mscclpp._mscclpp import (
    CppRawGpuBuffer,
    CppRawGpuBufferPool,
    CppRawGpuBufferPoolAllocation,
    CppGpuBufferGranularity,
)

__all__ = ["GpuBuffer", "GpuBufferPool", "GpuBufferGranularity"]

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


class GpuBufferPool:
    """A GPU buffer pool that returns raw buffer allocations backed by one communication-friendly allocation.

    All ranks should create the same-sized pool and call :meth:`allocate` in the same order to get matching offsets.
    Pass ``alloc_id`` when a buffer must be reused deterministically across ranks: the id reserves its offset for the
    lifetime of the pool and can be allocated again after the previous raw allocation using the same id is destroyed.
    """

    def __init__(
        self,
        nbytes: int,
        granularity: CppGpuBufferGranularity = CppGpuBufferGranularity.MultiCastMinimum,
    ):
        if nbytes <= 0:
            raise ValueError("Pool size must be positive.")
        self._pool = CppRawGpuBufferPool(int(nbytes), granularity)

    @property
    def bytes(self) -> int:
        """Number of bytes in the underlying pool allocation."""
        return self._pool.bytes()

    @property
    def free_bytes(self) -> int:
        """Number of bytes available for new non-persistent allocations."""
        return self._pool.free_bytes()

    @property
    def active_bytes(self) -> int:
        """Number of bytes currently held by live raw allocations."""
        return self._pool.active_bytes()

    @property
    def data(self) -> int:
        """Device pointer to the pool base allocation."""
        return self._pool.data()

    @property
    def device_id(self) -> int:
        """CUDA/HIP device id of the pool allocation."""
        return self._pool.device_id()

    def allocate(
        self,
        nbytes: int,
        alignment: int = 256,
        alloc_id: Optional[int] = None,
    ) -> CppRawGpuBufferPoolAllocation:
        """Allocate a raw buffer from the pool.

        Args:
            nbytes: Number of bytes to allocate.
            alignment: Required byte alignment of the allocation offset from the pool base.
            alloc_id: Optional persistent allocation id for deterministic reuse across ranks.
        """
        if nbytes <= 0:
            raise ValueError("Allocation size must be positive.")
        if alignment <= 0:
            raise ValueError("Alignment must be positive.")

        return self._pool.allocate(int(nbytes), int(alignment), alloc_id)
