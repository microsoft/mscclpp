# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import os
import struct
import subprocess
import tempfile
from typing import Any, Type

from cuda import cuda, nvrtc, cudart
import cupy as cp
import numpy as np

try:
    import torch

    _use_torch = True
    torchTensor = torch.Tensor
except ImportError:
    _use_torch = False
    torchTensor = Type[Any]


def _check_cuda_errors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cuda_get_error(result[0])})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def _cuda_get_error(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


class Kernel:
    def __init__(self, ptx: bytes, kernel_name: str, device_id: int):
        self._context = _check_cuda_errors(cuda.cuCtxGetCurrent())
        assert self._context is not None
        self._module = _check_cuda_errors(cuda.cuModuleLoadData(ptx))
        self._kernel = _check_cuda_errors(cuda.cuModuleGetFunction(self._module, kernel_name.encode()))

    def launch_kernel(
        self,
        params: bytes,
        nblocks: int,
        nthreads: int,
        shared: int,
        stream: Type[cuda.CUstream] or Type[cudart.cudaStream_t],
    ):
        buffer = (ctypes.c_byte * len(params)).from_buffer_copy(params)
        buffer_size = ctypes.c_size_t(len(params))
        config = np.array(
            [
                cuda.CU_LAUNCH_PARAM_BUFFER_POINTER,
                ctypes.addressof(buffer),
                cuda.CU_LAUNCH_PARAM_BUFFER_SIZE,
                ctypes.addressof(buffer_size),
                cuda.CU_LAUNCH_PARAM_END,
            ],
            dtype=np.uint64,
        )
        _check_cuda_errors(
            cuda.cuLaunchKernel(self._kernel, nblocks, 1, 1, nthreads, 1, 1, shared, stream, 0, config.ctypes.data)
        )

    def __del__(self):
        cuda.cuModuleUnload(self._module)


class KernelBuilder:
    kernel_map: dict = {}

    def get_key(self, kernel_name, macro_dict):
        return kernel_name + "-".join(f"{key}={macro_dict[key]}" for key in sorted(macro_dict))

    def __init__(self, file: str, kernel_name: str, file_dir: str = None, macro_dict: dict = {}):
        kernel_key = self.get_key(kernel_name, macro_dict)
        if kernel_key in self.kernel_map:
            self._kernel = self.kernel_map[kernel_key]
            return
        self._tempdir = tempfile.TemporaryDirectory(suffix=f"{os.getpid()}")
        self._current_file_dir = file_dir if file_dir else os.path.dirname(os.path.abspath(__file__))
        self.macros = None
        if file_dir:
            self.macros = ["-D{}={}".format(macro, value) for macro, value in macro_dict.items()]
        device_id = cp.cuda.Device().id
        ptx = self._compile_cuda(os.path.join(self._current_file_dir, file), f"{kernel_name}.ptx", device_id)
        self._kernel = Kernel(ptx, kernel_name, device_id)
        self.kernel_map[kernel_key] = self._kernel

    def _compile_cuda(self, source_file, output_file, device_id, std_version="c++17"):
        mscclpp_home = os.environ.get("MSCCLPP_HOME", "/usr/local/mscclpp")
        include_dir = os.path.join(mscclpp_home, "include")
        major = _check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id)
        )
        minor = _check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id)
        )
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc = os.path.join(cuda_home, "bin/nvcc") if cuda_home else "nvcc"
        command = [
            nvcc,
            f"-std={std_version}",
            "-ptx",
            "-Xcompiler",
            "-Wall,-Wextra",
            f"-I{include_dir}",
            f"{source_file}",
            f"--gpu-architecture=compute_{major}{minor}",
            f"--gpu-code=sm_{major}{minor},compute_{major}{minor}",
            "-o",
            f"{self._tempdir.name}/{output_file}",
        ]
        if self.macros:
            command += self.macros
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, bufsize=1)
            with open(f"{self._tempdir.name}/{output_file}", "rb") as f:
                return f.read()
        except subprocess.CalledProcessError as e:
            print(e.stderr, end="")
            raise RuntimeError("Compilation failed: ", " ".join(command))

    def get_compiled_kernel(self):
        return self._kernel

    def __del__(self):
        if hasattr(self, "_tempdir"):
            self._tempdir.cleanup()


def pack(*args):
    res = b""
    for arg in list(args):
        if isinstance(arg, int):
            res += struct.pack("i", arg)
        elif isinstance(arg, ctypes.c_size_t):
            res += struct.pack("N", arg.value)
        elif isinstance(arg, np.ndarray):
            res += struct.pack("P", arg.ctypes.data)
        elif isinstance(arg, cp.ndarray):
            res += struct.pack("P", arg.data.ptr)
        elif is_torch_tensor(arg):
            res += struct.pack("P", arg.data_ptr())
        # use int to represent bool, which can avoid CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES error
        elif isinstance(arg, bool):
            res += struct.pack("i", arg)
        elif isinstance(arg, bytes):
            res += struct.pack(f"{len(arg)}s", arg)
        else:
            raise RuntimeError(f"Unsupported type: {type(arg)}")
    return res


def is_torch_tensor(tensor: Any) -> bool:
    return _use_torch and isinstance(tensor, torchTensor)
