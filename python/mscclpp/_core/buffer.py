# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from contextlib import contextmanager
import math
import operator
import sys

import numpy as np
from mscclpp._mscclpp import CppRawGpuBuffer, CppGpuBufferGranularity, is_hip

__all__ = ["GpuBuffer", "GpuBufferGranularity"]

GpuBufferGranularity = CppGpuBufferGranularity


def _runtime():
    if is_hip:
        from hip import hip

        return hip, "hip"
    from cuda.bindings import runtime

    return runtime, "cuda"


def _call(func, *args):
    error, *values = func(*args)
    if int(error) != 0:
        raise RuntimeError(f"{func.__name__} failed with error {int(error)}")
    return values


class GpuBuffer:
    """Native-owned, contiguous GPU storage with explicit host transfers.

    Unlike an array library, this class does not perform arithmetic or implicit
    device-to-host copies. One-dimensional unit-step slices share the allocation.
    Transfers are synchronous and must be performed outside GPU graph capture.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        dtype=float,
        strides: tuple[int, ...] | None = None,
        order: str = "C",
        granularity: CppGpuBufferGranularity = CppGpuBufferGranularity.MultiCastMinimum,
    ):
        if isinstance(shape, (int, np.integer)):
            shape = (shape,)
        try:
            self._shape = tuple(operator.index(s) for s in shape)
        except TypeError as exc:
            raise ValueError("Shape must be an integer or a sequence of integers.") from exc
        if any(s < 0 for s in self.shape):
            raise ValueError("Shape dimensions must be non-negative.")
        self._dtype = np.dtype(dtype)
        if self.dtype.kind not in "biufc" or not self.dtype.isnative:
            raise ValueError("GpuBuffer requires a native-endian numeric or boolean dtype.")
        if order not in ("C", "F"):
            raise ValueError("Order must be 'C' or 'F'.")
        self._order = order
        self._size = math.prod(self.shape)
        if self.nbytes > sys.maxsize:
            raise OverflowError("Buffer size exceeds the addressable range.")
        dims = range(self.ndim - 1, -1, -1) if order == "C" else range(self.ndim)
        contiguous_strides = [0] * self.ndim
        stride = self.itemsize
        for dim in dims:
            contiguous_strides[dim] = stride
            stride *= max(1, self.shape[dim])
        self._strides = tuple(contiguous_strides)
        if strides is not None and tuple(strides) != self.strides:
            raise ValueError("GpuBuffer supports only contiguous strides.")
        self._allocation = CppRawGpuBuffer(max(1, self.nbytes), granularity)
        self._offset = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return self._size

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def data(self) -> "GpuBuffer":
        """Pointer interface compatible with ``buffer.data.ptr`` consumers."""
        return self

    @property
    def ptr(self) -> int:
        return self._allocation.data() + self._offset

    @property
    def allocation_size(self) -> int:
        """Total native allocation size, for binding an unsliced buffer to NVLS."""
        return self._allocation.bytes()

    @property
    def device_id(self) -> int:
        return self._allocation.device_id()

    def __getitem__(self, key: slice) -> "GpuBuffer":
        if self.ndim != 1 or not isinstance(key, slice):
            raise TypeError("GpuBuffer supports only one-dimensional slices; use to_numpy() for host indexing.")
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("GpuBuffer slices must have unit step.")
        view = object.__new__(type(self))
        view._allocation = self._allocation
        view._offset = self._offset + start * self.itemsize
        view._shape = (max(0, stop - start),)
        view._size = view._shape[0]
        view._dtype = self.dtype
        view._order = self._order
        view._strides = (self.itemsize,)
        return view

    def __array__(self, dtype=None, copy=None):
        raise TypeError("GpuBuffer does not convert implicitly to host memory; use to_numpy().")

    @contextmanager
    def _on_device(self):
        runtime, prefix = _runtime()
        (previous,) = _call(getattr(runtime, prefix + "GetDevice"))
        if previous != self.device_id:
            _call(getattr(runtime, prefix + "SetDevice"), self.device_id)
        try:
            yield runtime, prefix
        finally:
            if previous != self.device_id:
                _call(getattr(runtime, prefix + "SetDevice"), previous)

    def _copy(self, dst: int, src: int, direction: str) -> None:
        if self.nbytes == 0:
            return
        with self._on_device() as (runtime, prefix):
            # Include writes on non-default streams before exposing data to the host.
            _call(getattr(runtime, prefix + "DeviceSynchronize"))
            kind = getattr(getattr(runtime, prefix + "MemcpyKind"), prefix + "Memcpy" + direction)
            _call(getattr(runtime, prefix + "Memcpy"), dst, src, self.nbytes, kind)
            _call(getattr(runtime, prefix + "DeviceSynchronize"))

    def to_numpy(self) -> np.ndarray:
        """Synchronously copy this buffer to a new NumPy array."""
        result = np.empty(self.shape, dtype=self.dtype, order=self._order)
        self._copy(result.ctypes.data, self.ptr, "DeviceToHost")
        return result

    def copy_from_numpy(self, values) -> None:
        """Synchronously upload host values with the same shape and dtype."""
        array = np.asarray(values, order=self._order)
        if array.shape != self.shape or array.dtype != self.dtype:
            raise ValueError("Source shape and dtype must match the GPU buffer.")
        self._copy(self.ptr, array.ctypes.data, "HostToDevice")

    @classmethod
    def from_numpy(cls, values, dtype=None) -> "GpuBuffer":
        """Allocate GPU storage and synchronously upload host values."""
        array = np.asarray(values, dtype=dtype)
        result = cls(array.shape, dtype=array.dtype, order="F" if np.isfortran(array) else "C")
        result.copy_from_numpy(array)
        return result

    def fill(self, value) -> None:
        """Synchronously fill storage with a scalar, outside graph capture."""
        if self.nbytes == 0:
            return
        scalar = np.asarray(value, dtype=self.dtype)
        if scalar.ndim != 0:
            raise ValueError("Fill value must be a scalar.")
        if not scalar.tobytes().strip(b"\0"):
            with self._on_device() as (runtime, prefix):
                _call(getattr(runtime, prefix + "DeviceSynchronize"))
                _call(getattr(runtime, prefix + "Memset"), self.ptr, 0, self.nbytes)
                _call(getattr(runtime, prefix + "DeviceSynchronize"))
        else:
            self.copy_from_numpy(np.full(self.shape, value, dtype=self.dtype, order=self._order))
