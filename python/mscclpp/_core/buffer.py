# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from contextlib import contextmanager
import math
import operator
import sys

import numpy as np
from mscclpp import gpu
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


class _ExternalGpuAllocation:
    def __init__(self, ptr: int, nbytes: int, owner: object):
        self._ptr = ptr
        self._nbytes = nbytes
        self._owner = owner
        self._device_id = None

    def data(self) -> int:
        return self._ptr

    def bytes(self) -> int:
        return self._nbytes

    def device_id(self) -> int:
        if self._device_id is None:
            runtime, prefix = _runtime()
            (attributes,) = _call(getattr(runtime, prefix + "PointerGetAttributes"), self._ptr)
            self._device_id = int(attributes.device)
        return self._device_id


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
        self._set_layout(shape, dtype, strides, order)
        self._allocation = CppRawGpuBuffer(max(1, self.nbytes), granularity)
        self._offset = 0

    def _set_layout(self, shape, dtype, strides, order):
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

    @classmethod
    def from_ptr(cls, ptr: int, nbytes: int, owner: object, dtype=np.uint8) -> "GpuBuffer":
        """Wrap existing GPU memory as a one-dimensional buffer without allocating or copying.

        ``ptr`` must point to at least ``nbytes`` bytes of valid GPU storage.
        ``owner`` must keep that storage alive; this buffer and its slices retain it.
        The wrapper never frees the pointer itself. The byte size must be a multiple
        of the dtype's item size. Transfers query the pointer's device, not the current
        device at construction.
        """
        ptr = operator.index(ptr)
        nbytes = operator.index(nbytes)
        if nbytes < 0:
            raise ValueError("Buffer byte size must be non-negative.")
        address_limit = 2 * (sys.maxsize + 1)
        if not 0 <= ptr < address_limit or ptr + nbytes > address_limit or (ptr == 0 and nbytes):
            raise ValueError("Invalid GPU memory pointer or address range.")
        if owner is None:
            raise ValueError("An owner is required to keep GPU memory alive.")
        dtype = np.dtype(dtype)
        if dtype.itemsize == 0 or nbytes % dtype.itemsize:
            raise ValueError("Buffer byte size must be a multiple of the dtype item size.")
        result = cls.__new__(cls)
        result._set_layout(nbytes // dtype.itemsize, dtype, None, "C")
        result._allocation = _ExternalGpuAllocation(ptr, nbytes, owner)
        result._offset = 0
        return result

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
        previous = gpu.current_device()
        if previous != self.device_id:
            gpu.set_device(self.device_id)
        try:
            yield runtime, prefix
        finally:
            if previous != self.device_id:
                gpu.set_device(previous)

    def _copy(self, dst: int, src: int, direction: str) -> None:
        if self.nbytes == 0:
            return
        with self._on_device() as (runtime, prefix):
            # Include writes on non-default streams before exposing data to the host.
            gpu.device_synchronize()
            kind = getattr(getattr(runtime, prefix + "MemcpyKind"), prefix + "Memcpy" + direction)
            _call(getattr(runtime, prefix + "Memcpy"), dst, src, self.nbytes, kind)
            gpu.device_synchronize()

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
                gpu.device_synchronize()
                _call(getattr(runtime, prefix + "Memset"), self.ptr, 0, self.nbytes)
                gpu.device_synchronize()
        else:
            self.copy_from_numpy(np.full(self.shape, value, dtype=self.dtype, order=self._order))
