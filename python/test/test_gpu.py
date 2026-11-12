# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of shared CUDA/HIP runtime helpers."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CudaGpuTests(unittest.TestCase):
    is_hip = False

    def load_module(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.python_dir = Path(__file__).resolve().parents[1]
        package = ModuleType("mscclpp")
        package.__path__ = [str(self.python_dir / "mscclpp")]
        self.cuda = SimpleNamespace(
            cudaError_t=SimpleNamespace(cudaSuccess=0),
            cudaStreamCaptureMode=SimpleNamespace(cudaStreamCaptureModeRelaxed=2),
            cudaStreamNonBlocking=1,
            cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2),
        )
        self.hip = SimpleNamespace(
            hipError_t=SimpleNamespace(hipSuccess=0),
            hipStreamCaptureMode=SimpleNamespace(hipStreamCaptureModeRelaxed=2),
            hipStreamNonBlocking=1,
            hipMemcpyKind=SimpleNamespace(hipMemcpyHostToDevice=11, hipMemcpyDeviceToHost=12),
        )
        modules = {
            "mscclpp": package,
            "mscclpp._mscclpp": SimpleNamespace(CppDataType=object, is_hip=self.is_hip),
            "cuda": ModuleType("cuda"),
            "cuda.bindings": SimpleNamespace(runtime=self.cuda),
            "hip": SimpleNamespace(hip=self.hip),
            "numpy": ModuleType("numpy"),
            "torch": None,
            "cupy": None,
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.gpu = self.load_module("mscclpp.gpu", self.python_dir / "mscclpp" / "gpu.py")
        package.gpu = self.gpu
        for hip_name, cuda_name in self.gpu._API_NAMES.values():
            setattr(self.hip, hip_name, Mock(return_value=(0,)))
            setattr(self.cuda, cuda_name, Mock(return_value=(0,)))
        self.runtime = self.hip if self.is_hip else self.cuda
        self.prefix = "hip" if self.is_hip else "cuda"
        self.api("GetDeviceCount").return_value = (0, 2)
        self.api("GetDevice").return_value = (0, 1)
        self.props = SimpleNamespace(name=b"Test GPU", major=9, minor=0, gcnArchName=b"gfx942:sramecc+:xnack-")
        self.cuda.cudaGetDeviceProperties.return_value = (0, self.props)
        self.api("GetErrorString").return_value = (0, b"test failure")
        self.utils = self.load_module("_gpu_utils_test", self.python_dir / "mscclpp" / "utils.py")

    def api(self, name):
        return getattr(self.runtime, self.prefix + name)

    def test_imports_and_api_lookup_do_not_initialize_runtime(self):
        self.assertIsNone(self.gpu._RUNTIME)
        for name in self.gpu.__all__:
            self.assertTrue(callable(getattr(self.gpu, name)))
        self.assertIsNone(self.gpu._RUNTIME)
        self.api("GetDeviceCount").assert_not_called()
        with self.assertRaises(AttributeError):
            getattr(self.gpu, "unknown_api")

    def test_uses_native_backend_when_both_bindings_are_installed(self):
        self.assertEqual(self.gpu.runtime_name(), self.prefix)
        self.gpu.init_runtime()
        self.assertEqual(self.gpu.current_device(), 1)
        self.api("GetDeviceCount").assert_called_once_with()
        other = self.cuda if self.is_hip else self.hip
        for func in vars(other).values():
            if isinstance(func, Mock):
                func.assert_not_called()

    def test_get_device_arch_uses_shared_current_device_and_properties(self):
        expected = "gfx942:sramecc+:xnack-" if self.is_hip else "sm_90"
        self.assertEqual(self.utils.get_device_arch(), expected)
        self.api("GetDevice").assert_called_once_with()
        self.assert_properties_query(1)

    def test_get_device_arch_preserves_string_arch_and_multi_digit_major(self):
        self.props.gcnArchName = "gfx950:xnack-"
        self.props.major = 10
        self.assertEqual(self.utils.get_device_arch(), "gfx950:xnack-" if self.is_hip else "sm_100")

    def test_current_device_error_uses_shared_error_handling(self):
        self.api("GetDevice").return_value = (17, None)
        with self.assertRaisesRegex(RuntimeError, r"get_device failed: test failure \(17\)"):
            self.utils.get_device_arch()

    def test_device_properties_error_uses_shared_error_handling(self):
        self.api("GetDeviceProperties").side_effect = lambda *args: (17,)
        with self.assertRaisesRegex(RuntimeError, r"get_device_properties failed: test failure \(17\)"):
            self.utils.get_device_arch()

    def test_error_string_failure_preserves_error_code(self):
        self.api("GetDevice").return_value = (18, None)
        self.api("GetErrorString").return_value = (19, None)
        with self.assertRaisesRegex(RuntimeError, "get_device failed with error 18"):
            self.utils.get_device_arch()

    def test_missing_selected_binding_does_not_fall_back(self):
        binding = "hip" if self.is_hip else "cuda.bindings"
        with patch.dict(sys.modules, {binding: None}):
            with self.assertRaises(ImportError):
                self.gpu.init_runtime()
        self.assertIsNone(self.gpu._RUNTIME)

    def test_no_devices_is_reported_and_initialization_can_be_retried(self):
        self.api("GetDeviceCount").return_value = (0, 0)
        with self.assertRaisesRegex(RuntimeError, f"No usable {self.prefix} devices found"):
            self.gpu.init_runtime()
        self.assertIsNone(self.gpu._RUNTIME)
        self.api("GetDeviceCount").return_value = (0, 2)
        self.gpu.init_runtime()
        self.assertEqual(self.api("GetDeviceCount").call_count, 2)

    def test_device_count_failure_is_reported(self):
        self.api("GetDeviceCount").return_value = (20, None)
        with self.assertRaisesRegex(RuntimeError, r"get_device_count failed: test failure \(20\)"):
            self.gpu.init_runtime()

    def test_direct_imports_share_helpers_and_runtime(self):
        from mscclpp import gpu
        from mscclpp.gpu import current_device, init_runtime

        self.assertIs(gpu, self.gpu)
        self.assertIs(current_device, gpu.current_device)
        self.assertIs(init_runtime, gpu.init_runtime)
        self.assertIsNone(gpu._RUNTIME)
        init_runtime()
        self.assertEqual(current_device(), gpu.current_device())
        self.api("GetDeviceCount").assert_called_once_with()

    def test_device_names_and_stream_flags(self):
        self.assertEqual(self.gpu.device_name(), "Test GPU")
        self.assert_properties_query(1)
        self.api("StreamCreateWithFlags").return_value = (0, 123)
        for non_blocking in (False, True):
            self.assertEqual(self.gpu.create_stream(non_blocking), 123)
            self.api("StreamCreateWithFlags").assert_called_with(int(non_blocking))

    def assert_properties_query(self, device_id):
        self.api("GetDeviceProperties").assert_called_once_with(device_id)

    def test_device_properties_return_single_properties_object_for_explicit_device(self):
        (props,) = self.gpu.get_device_properties(0)
        self.assertEqual(vars(props), vars(self.props))
        self.assert_properties_query(0)
        self.api("GetDevice").assert_not_called()

    def test_version_decoding(self):
        self.api("RuntimeGetVersion").return_value = (0, 60_400_000 if self.is_hip else 12040)
        self.assertEqual(self.gpu.version(), (6, 4, 0) if self.is_hip else (12, 4, 0))

    def test_pointer_attributes_and_capture_status_use_shared_runtime(self):
        attributes = SimpleNamespace(device=1)
        self.api("PointerGetAttributes").return_value = (0, attributes)
        self.api("StreamIsCapturing").return_value = (0, 1)
        self.assertEqual(self.gpu.pointer_get_attributes(1234), (attributes,))
        self.assertEqual(self.gpu.stream_is_capturing(123), (1,))
        self.api("PointerGetAttributes").assert_called_once_with(1234)
        self.api("StreamIsCapturing").assert_called_once_with(123)

    def test_memcpy_async_resolves_backend_direction_and_stream(self):
        for direction, kind in (("HostToDevice", 11 if self.is_hip else 1), ("DeviceToHost", 12 if self.is_hip else 2)):
            with self.subTest(direction=direction):
                self.assertEqual(self.gpu.memcpy_async(4096, 8192, 64, direction), ())
                self.api("MemcpyAsync").assert_called_with(4096, 8192, 64, kind, 0)
                self.gpu.memcpy_async(4096, 8192, 64, direction, 123)
                self.api("MemcpyAsync").assert_called_with(4096, 8192, 64, kind, 123)

    def test_memcpy_async_rejects_unknown_direction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported host transfer direction"):
            self.gpu.memcpy_async(4096, 8192, 64, "invalid")
        self.api("MemcpyAsync").assert_not_called()

    def test_graph_capture_uses_backend_specific_instantiate_signature(self):
        self.api("StreamEndCapture").return_value = (0, 42)
        self.api("GraphInstantiate").return_value = (0, 43)
        capture = Mock()
        graph = self.gpu.capture_graph(123, capture)
        capture.assert_called_once_with()
        self.api("StreamBeginCapture").assert_called_once_with(123, 2)
        instantiate_args = (42, None, 0) if self.is_hip else (42, 0)
        self.api("GraphInstantiate").assert_called_once_with(*instantiate_args)
        self.api("GraphDestroy").assert_called_once_with(42)
        graph.launch(123)
        self.api("GraphLaunch").assert_called_once_with(43, 123)
        graph.close()
        graph.close()
        self.api("GraphExecDestroy").assert_called_once_with(43)


class HipGpuTests(CudaGpuTests):
    is_hip = True

    def setUp(self):
        super().setUp()
        self.hip.hipDeviceProp_t = Mock(side_effect=SimpleNamespace)

        def get_device_properties(props, device_id):
            vars(props).update(vars(self.props))
            return (0,)

        self.hip.hipGetDeviceProperties.side_effect = get_device_properties

    def assert_properties_query(self, device_id):
        self.hip.hipDeviceProp_t.assert_called_once_with()
        props, actual_device_id = self.api("GetDeviceProperties").call_args.args
        self.assertEqual(vars(props), vars(self.props))
        self.assertEqual(actual_device_id, device_id)
        self.api("GetDeviceProperties").assert_called_once_with(props, device_id)

    def test_fp8_module_import_uses_shared_properties_query(self):
        self.props.gcnArchName = b"gfx950:sramecc+:xnack-"
        decorator = lambda *args, **kwargs: lambda func: func
        skipif = Mock(return_value=object())
        with (
            patch.multiple(
                sys.modules["mscclpp"],
                CommGroup=object,
                GpuBuffer=object,
                DataType=SimpleNamespace(float8_e4m3fn=object(), float8_e4m3fnuz=object()),
                ReduceOp=object,
                is_nvls_supported=object,
                create=True,
            ),
            patch.dict(
                sys.modules,
                {
                    "mscclpp.ext": SimpleNamespace(AlgorithmCollectionBuilder=object),
                    "mscclpp.utils": self.utils,
                    "mscclpp_benchmark.arrays": SimpleNamespace(array_module=Mock(), to_array=Mock(), copy_to=Mock()),
                    "pytest": SimpleNamespace(mark=SimpleNamespace(parametrize=decorator, skipif=skipif)),
                    "python.test.mscclpp_mpi": SimpleNamespace(
                        MpiGroup=object, parametrize_mpi_groups=decorator, mpi_group=object
                    ),
                },
            ),
        ):
            module = self.load_module("python.test._fp8_gpu_test", self.python_dir / "test" / "test_fp8_accum.py")
        self.assertEqual(module._gcn_arch_name, "gfx950")
        self.assertTrue(module._is_cdna4)
        self.assertFalse(module._skip_fp8)
        skipif.assert_called_once_with(False, reason="FP8 accum tests require SM >= 89 on CUDA")
        self.assert_properties_query(1)

    def test_properties_queries_allocate_independent_output_objects(self):
        (first,) = self.gpu.get_device_properties(0)
        self.props.gcnArchName = b"gfx950"
        (second,) = self.gpu.get_device_properties(1)
        self.assertIsNot(first, second)
        self.assertEqual(first.gcnArchName, b"gfx942:sramecc+:xnack-")
        self.assertEqual(second.gcnArchName, b"gfx950")
        self.assertEqual(self.hip.hipDeviceProp_t.call_count, 2)

    def test_properties_scalar_status_is_checked(self):
        self.hip.hipGetDeviceProperties.side_effect = lambda props, device_id: 17
        with self.assertRaisesRegex(RuntimeError, r"get_device_properties failed: test failure \(17\)"):
            self.gpu.get_device_properties(0)
        self.hip.hipGetDeviceProperties.side_effect = lambda props, device_id: 0
        (props,) = self.gpu.get_device_properties(0)
        self.assertIsInstance(props, SimpleNamespace)


if __name__ == "__main__":
    unittest.main()
