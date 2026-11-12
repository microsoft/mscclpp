# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional GPU array operations for tests and benchmarks, not the runtime."""

import importlib.util

import numpy as np


def array_module(array=None):
    """Keep host references on NumPy; use matching CuPy for GPU workloads."""
    if isinstance(array, np.ndarray):
        return np
    if importlib.util.find_spec("cupy") is None:
        return np
    import cupy
    from mscclpp._mscclpp import is_hip

    if bool(cupy.cuda.runtime.is_hip) != is_hip:
        raise RuntimeError("CuPy and MSCCL++ must use the same CUDA or HIP backend")
    return cupy if array is None else cupy.get_array_module(array)


def buffer_array(buffer):
    """Return a GPU storage view retaining its owner, or a host copy without CuPy."""
    xp = array_module()
    if xp is np:
        return buffer.to_numpy()
    from mscclpp.gpu import device_synchronize

    name = buffer.dtype.name
    dtype = np.dtype("uint8" if name.startswith("float8") else "uint16" if name == "bfloat16" else name)
    with xp.cuda.Device(buffer.device_id):
        device_synchronize()
        if buffer.ptr == 0:
            return xp.empty(buffer.shape, dtype=dtype)
        memory = xp.cuda.UnownedMemory(buffer.ptr, buffer.nbytes, buffer, device_id=buffer.device_id)
        return xp.ndarray(buffer.shape, dtype=dtype, memptr=xp.cuda.MemoryPointer(memory, 0), strides=buffer.strides)


def copy_to(buffer, values):
    """Write storage values and finish the write before native communication."""
    xp = array_module()
    if xp is np:
        buffer.copy_from_numpy(values)
        return
    with xp.cuda.Device(buffer.device_id):
        buffer_array(buffer)[...] = xp.asarray(values)
        xp.cuda.get_current_stream().synchronize()


def buffer_from_array(values):
    from mscclpp import GpuBuffer

    buffer = GpuBuffer(values.shape, dtype=values.dtype)
    copy_to(buffer, values)
    return buffer
