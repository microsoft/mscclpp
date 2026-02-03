# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import os
import struct
import subprocess
import tempfile
from typing import Any, Type, Union

import cupy as cp
import numpy as np

from mscclpp._mscclpp import CppDataType as DataType

try:
    import torch

    _use_torch = True
    torchTensor = torch.Tensor
except ImportError:
    _use_torch = False
    torchTensor = Type[Any]


__all__ = [
    "Kernel",
    "KernelBuilder",
    "pack",
    "get_device_arch",
    "torch_dtype_to_mscclpp_dtype",
]


def get_device_arch() -> str:
    if cp.cuda.runtime.is_hip:
        return cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["gcnArchName"].decode("utf-8")
    else:
        return f"sm_{cp.cuda.Device().compute_capability}"


class Kernel:
    CU_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    CU_LAUNCH_PARAM_BUFFER_SIZE = 0x02
    CU_LAUNCH_PARAM_END = 0x00 if not cp.cuda.runtime.is_hip else 0x03

    def __init__(self, cubin: bytes, kernel_name: str):
        self._module = cp.cuda.driver.moduleLoadData(cubin)
        self._kernel = cp.cuda.driver.moduleGetFunction(self._module, kernel_name)

    def launch_kernel(
        self,
        params: bytes,
        nblocks: int,
        nthreads: int,
        shared: int,
        stream: Union[cp.cuda.Stream, None],
    ):
        buffer = (ctypes.c_byte * len(params)).from_buffer_copy(params)
        buffer_size = ctypes.c_size_t(len(params))
        config = np.array(
            [
                Kernel.CU_LAUNCH_PARAM_BUFFER_POINTER,
                ctypes.addressof(buffer),
                Kernel.CU_LAUNCH_PARAM_BUFFER_SIZE,
                ctypes.addressof(buffer_size),
                Kernel.CU_LAUNCH_PARAM_END,
            ],
            dtype=np.uint64,
        )
        cuda_stream = 0
        if stream:
            cuda_stream = stream.ptr if isinstance(stream, cp.cuda.Stream) else stream.cuda_stream
        cp.cuda.driver.launchKernel(
            self._kernel, nblocks, 1, 1, nthreads, 1, 1, shared, cuda_stream, 0, config.ctypes.data
        )

    def __del__(self):
        cp.cuda.driver.moduleUnload(self._module)


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
        cubin = self._compile_cuda(os.path.join(self._current_file_dir, file), f"{kernel_name}.cubin")
        self._kernel = Kernel(cubin, kernel_name)
        self.kernel_map[kernel_key] = self._kernel

    def _compile_cuda(self, source_file, output_file, std_version="c++17"):
        mscclpp_home = os.environ.get("MSCCLPP_HOME", "/usr/local/mscclpp")
        include_dir = os.path.join(mscclpp_home, "include")
        if not cp.cuda.runtime.is_hip:
            arch = get_device_arch()
            compute_capability = arch.replace("sm_", "")
            cuda_home = os.environ.get("CUDA_HOME")
            nvcc = os.path.join(cuda_home, "bin/nvcc") if cuda_home else "nvcc"
            command = [
                nvcc,
                f"-std={std_version}",
                "-cubin",
                "-Xcompiler",
                "-Wall,-Wextra",
                f"-I{include_dir}",
                f"{source_file}",
                f"--gpu-architecture=compute_{compute_capability}",
                f"--gpu-code=sm_{compute_capability}",
                "-o",
                f"{self._tempdir.name}/{output_file}",
            ]
        else:
            # the gcn arch name is like "gfx942:sramecc+:xnack-"
            gcn_arch = get_device_arch()
            rocm_home = os.environ.get("ROCM_HOME")
            hipcc = os.path.join(rocm_home, "bin/hipcc") if rocm_home else "hipcc"
            command = [
                hipcc,
                f"-std={std_version}",
                "--genco",
                "-D__HIP_PLATFORM_AMD__",
                f"--offload-arch={gcn_arch}",
                f"-I{include_dir}",
                f"{source_file}",
                "-o",
                f"{self._tempdir.name}/{output_file}",
            ]
        if self.macros:
            command += self.macros
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, bufsize=1, stdin=subprocess.DEVNULL)
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


def torch_dtype_to_mscclpp_dtype(dtype: "torch.dtype") -> DataType:
    if not _use_torch:
        raise RuntimeError("PyTorch is not available.")
    if dtype == torch.float16:
        return DataType.float16
    elif dtype == torch.float32:
        return DataType.float32
    elif dtype == torch.int32:
        return DataType.int32
    elif dtype == torch.bfloat16:
        return DataType.bfloat16
    elif dtype == torch.float8_e5m2:
        return DataType.float8_e5m2
    else:
        raise ValueError(f"Unknown data type: {dtype}")
