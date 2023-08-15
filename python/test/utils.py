import ctypes
import os
import struct
import subprocess
import tempfile

from cuda import cuda, nvrtc
import numpy as np
import torch


class Kernel:
    def __init__(self, ptx: bytes, kernel_name: str):
        device_index = torch.cuda.current_device()
        cu_device = self._cucall_and_check(cuda.cuDeviceGet, device_index)
        self._context = self._cucall_and_check(cuda.cuCtxCreate, 0, cu_device)
        self._module = self._cucall_and_check(cuda.cuModuleLoadData, ptx)
        self._kernel = self._cucall_and_check(cuda.cuModuleGetFunction, self._module, kernel_name.encode())

    def launch_kernel(self, params: bytes, nblocks: int, nthreads: int, shared: int, stream):
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
        self._cucall_and_check(
            cuda.cuLaunchKernel, self._kernel, nblocks, 1, 1, nthreads, 1, 1, shared, stream, 0, config.ctypes.data
        )

    def _cucall_and_check(self, cuda_func, *args):
        results = cuda_func(*args)
        if len(results) == 1:
            err, result = results[0], None
        elif len(results) == 2:
            err, result = results
        else:
            raise RuntimeError("Unknown result type: {}".format(results))

        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        elif isinstance(err, nvrtc.nvrtcResult):
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError("Nvrtc Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))
        return result

    def __del__(self):
        cuda.cuModuleUnload(self._module)
        cuda.cuCtxDestroy(self._context)


class KernelBase:
    def __init__(self, file: str, args: dict):
        self._tempdir = tempfile.TemporaryDirectory()
        self._current_file_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_name = args["KERNEL"]
        ptx = self._compile_cuda(os.path.join(self._current_file_dir, file), f"{args['KERNEL']}.ptx", args)
        self._kernel = Kernel(ptx, kernel_name)

    def _compile_cuda(self, source_file, output_file, defines, std_version="c++17"):
        header_dir = os.path.join(self._current_file_dir, "../../include")
        defines = " ".join([f"-D{k}={v}" for k, v in defines.items()])
        command = (
            f"nvcc -std={std_version} -ptx -Xcompiler -Wall,-Wextra -I{header_dir} -DPARAMETRIZE {source_file} {defines} "
            f"-o {self._tempdir.name}/{output_file}"
        )
        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(f"{self._tempdir.name}/{output_file}", "rb") as f:
                return f.read()
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Compilation failed:", e.stderr.decode(), command)

    def __del__(self):
        self._tempdir.cleanup()


def pack(*args):
    res = b""
    for arg in list(args):
        if isinstance(arg, int):
            res += struct.pack("i", arg)
        elif isinstance(arg, torch.Tensor):
            res += struct.pack("P", arg.data_ptr())
        else:
            raise RuntimeError(f"Unsupported type: {type(arg)}")
    return res
