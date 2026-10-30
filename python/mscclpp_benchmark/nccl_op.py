# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
from functools import lru_cache
import os

import numpy as np
from mscclpp import GpuBuffer
from mscclpp._mscclpp import is_hip


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


@lru_cache(maxsize=1)
def _library():
    name = os.environ.get("MSCCLPP_BENCH_NCCL_LIBRARY", "librccl.so.1" if is_hip else "libnccl.so.2")
    try:
        lib = ctypes.CDLL(name)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load {name}. Install NCCL/RCCL or set MSCCLPP_BENCH_NCCL_LIBRARY to its shared library path."
        ) from exc
    signatures = {
        "ncclGetUniqueId": [ctypes.POINTER(_UniqueId)],
        "ncclCommInitRank": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int],
        "ncclCommDestroy": [ctypes.c_void_p],
        "ncclCommAbort": [ctypes.c_void_p],
        "ncclAllReduce": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
    }
    for symbol, args in signatures.items():
        func = getattr(lib, symbol)
        func.argtypes = args
        func.restype = ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib


def _check(lib, func, *args):
    status = func(*args)
    if status != 0:
        message = lib.ncclGetErrorString(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"{func.__name__} failed ({status}): {message}")


def get_unique_id() -> bytes:
    """Generate the opaque NCCL/RCCL ID to broadcast to every MPI rank."""
    lib = _library()
    uid = _UniqueId()
    _check(lib, lib.ncclGetUniqueId, ctypes.byref(uid))
    return bytes(uid)


class NcclCommunicator:
    """A blocking NCCL/RCCL communicator; use as a context manager for cleanup."""

    def __init__(self, nranks: int, uid: bytes, rank: int):
        if nranks <= 0 or not 0 <= rank < nranks:
            raise ValueError("Rank must be in [0, nranks).")
        if len(uid) != ctypes.sizeof(_UniqueId):
            raise ValueError("NCCL unique IDs must contain exactly 128 bytes.")
        self._lib = _library()
        self._handle = ctypes.c_void_p()
        _check(
            self._lib,
            self._lib.ncclCommInitRank,
            ctypes.byref(self._handle),
            nranks,
            _UniqueId.from_buffer_copy(uid),
            rank,
        )

    def all_reduce(self, memory: GpuBuffer, dtype: int, stream: int) -> None:
        if not self._handle.value:
            raise RuntimeError("NCCL communicator is closed.")
        _check(
            self._lib,
            self._lib.ncclAllReduce,
            memory.data.ptr,
            memory.data.ptr,
            memory.size,
            dtype,
            0,  # ncclSum
            self._handle,
            stream,
        )

    def close(self, *, abort: bool = False) -> None:
        if self._handle.value:
            func = self._lib.ncclCommAbort if abort else self._lib.ncclCommDestroy
            _check(self._lib, func, self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(abort=exc_type is not None)


class NcclAllReduce:
    def __init__(self, nccl_comm: NcclCommunicator, memory: GpuBuffer):
        self.nccl_comm = nccl_comm
        self.memory = memory
        if memory.dtype == np.float32:
            self.nccl_dtype = 7  # ncclFloat32
        elif memory.dtype == np.float16:
            self.nccl_dtype = 6  # ncclFloat16
        elif memory.dtype == np.int32:
            self.nccl_dtype = 2  # ncclInt32
        else:
            raise RuntimeError("Make sure that the data type is mapped to the correct NCCL data type")

    def __call__(self, stream: int | None):
        stream_ptr = 0 if stream is None else stream
        self.nccl_comm.all_reduce(self.memory, self.nccl_dtype, stream_ptr)
        return self.memory
