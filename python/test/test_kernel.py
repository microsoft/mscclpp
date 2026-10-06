# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of the CUDA/HIP kernel-driver boundary."""

import ctypes
import importlib.util
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CudaKernelTests(unittest.TestCase):
    is_hip = False

    def setUp(self):
        prefix = "hip" if self.is_hip else "cu"
        self.module = object()
        self.function = object()
        self.load = Mock(__name__=f"{prefix}ModuleLoadData", return_value=(0, self.module))
        self.lookup = Mock(__name__=f"{prefix}ModuleGetFunction", return_value=(0, self.function))
        self.launch = Mock(__name__="hipModuleLaunchKernel" if self.is_hip else "cuLaunchKernel", return_value=(0,))
        self.unload = Mock(__name__=f"{prefix}ModuleUnload", return_value=(0,))
        self.initialize = Mock(__name__="cudaFree", return_value=(0,))
        driver = SimpleNamespace(
            **{func.__name__: func for func in (self.load, self.lookup, self.launch, self.unload)}
        )
        native = ModuleType("mscclpp._mscclpp")
        native.CppDataType = object
        native.is_hip = self.is_hip
        # Load the real utility module without requiring GPU libraries or the native extension.
        modules = {
            "mscclpp._mscclpp": native,
            "cupy": None,
            "numpy": ModuleType("numpy"),
            "torch": None,
            "hip": SimpleNamespace(hip=driver),
            "cuda": ModuleType("cuda"),
            "cuda.bindings": SimpleNamespace(driver=driver, runtime=SimpleNamespace(cudaFree=self.initialize)),
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        path = Path(__file__).resolve().parents[1] / "mscclpp" / "utils.py"
        spec = importlib.util.spec_from_file_location("_mscclpp_kernel_utils_test", path)
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        self.Kernel = utils.Kernel

    def create_kernel(self):
        kernel = self.Kernel(b"\x7fELF\x00image", "test_kernel")
        self.addCleanup(kernel.__del__)
        return kernel

    def test_loads_module_and_encodes_function_name(self):
        self.create_kernel()
        self.load.assert_called_once_with(b"\x7fELF\x00image")
        self.lookup.assert_called_once_with(self.module, b"test_kernel")
        if self.is_hip:
            self.initialize.assert_not_called()
        else:
            self.initialize.assert_called_once_with(0)

    def test_launch_preserves_packed_arguments_and_streams(self):
        kernel = self.create_kernel()
        for stream, expected_stream in ((None, 0), (0, 0), (123, 123), (SimpleNamespace(cuda_stream=456), 456)):
            for params in (b"", struct.pack("i", 7) + b"\x00\xff\x00\x01"):
                with self.subTest(stream=stream, params=params):

                    def launch(*args):
                        self.assertEqual(
                            args[:-1], (self.function, 3, 1, 1, 64, 1, 1, 128, expected_stream, 0)
                        )
                        config = (ctypes.c_void_p * 5).from_address(args[-1])
                        self.assertEqual(config[0], 1)
                        self.assertEqual(config[2], 2)
                        self.assertEqual(config[4], 3 if self.is_hip else None)
                        size = ctypes.c_size_t.from_address(config[3]).value
                        self.assertEqual(size, len(params))
                        self.assertEqual(ctypes.string_at(config[1], size), params)
                        return (0,)

                    self.launch.side_effect = launch
                    kernel.launch_kernel(params, 3, 64, 128, stream)
        self.assertEqual(self.launch.call_count, 8)

    def test_unloads_module_once(self):
        kernel = self.create_kernel()
        kernel.__del__()
        kernel.__del__()
        self.unload.assert_called_once_with(self.module)

    def test_load_failure_does_not_unload(self):
        self.load.return_value = (17, None)
        with self.assertRaisesRegex(RuntimeError, f"{self.load.__name__} failed with error 17"):
            self.create_kernel()
        self.lookup.assert_not_called()
        self.unload.assert_not_called()

    def test_lookup_failure_unloads_module(self):
        self.lookup.return_value = (18, None)
        with self.assertRaisesRegex(RuntimeError, f"{self.lookup.__name__} failed with error 18"):
            self.create_kernel()
        self.unload.assert_called_once_with(self.module)

    def test_launch_failure_is_reported(self):
        kernel = self.create_kernel()
        self.launch.return_value = (19,)
        with self.assertRaisesRegex(RuntimeError, f"{self.launch.__name__} failed with error 19"):
            kernel.launch_kernel(b"", 1, 32, 0, None)

    def test_unload_failure_is_reported_without_double_free(self):
        kernel = self.create_kernel()
        self.unload.return_value = (20,)
        with self.assertRaisesRegex(RuntimeError, f"{self.unload.__name__} failed with error 20"):
            kernel.__del__()
        kernel.__del__()
        self.unload.assert_called_once_with(self.module)

    def test_runtime_initialization_failure_does_not_load_module(self):
        if self.is_hip:
            self.skipTest("CUDA runtime context initialization")
        self.initialize.return_value = (21,)
        with self.assertRaisesRegex(RuntimeError, "cudaFree failed with error 21"):
            self.create_kernel()
        self.load.assert_not_called()
        self.unload.assert_not_called()


class HipKernelTests(CudaKernelTests):
    is_hip = True


if __name__ == "__main__":
    unittest.main()
