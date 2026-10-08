# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only checks of native-buffer ownership, transfers, and consumer boundaries."""

import ctypes
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CudaBufferTests(unittest.TestCase):
    is_hip = False

    def setUp(self):
        self.device = 2
        self.prefix = "hip" if self.is_hip else "cuda"
        self.python_dir = Path(__file__).resolve().parents[1]
        current_device = lambda: self.device

        class Allocation:
            def __init__(self, size, granularity):
                self.memory = ctypes.create_string_buffer((size + 63) // 64 * 64)
                self.device = current_device()

            def data(self):
                return ctypes.addressof(self.memory)

            def bytes(self):
                return ctypes.sizeof(self.memory)

            def device_id(self):
                return self.device

        self.allocate = Mock(side_effect=Allocation)

        def set_device(device):
            self.device = device
            return (0,)

        def memcpy(dst, src, size, direction):
            ctypes.memmove(dst, src, size)
            return (0,)

        def memset(dst, value, size):
            ctypes.memset(dst, value, size)
            return (0,)

        self.runtime = SimpleNamespace()
        for name, callback in (
            ("GetDevice", lambda: (0, self.device)),
            ("SetDevice", set_device),
            ("DeviceSynchronize", lambda: (0,)),
            ("Memcpy", memcpy),
            ("Memset", memset),
        ):
            setattr(self.runtime, self.prefix + name, Mock(__name__=self.prefix + name, side_effect=callback))
        setattr(
            self.runtime,
            self.prefix + "MemcpyKind",
            SimpleNamespace(**{self.prefix + "MemcpyHostToDevice": 1, self.prefix + "MemcpyDeviceToHost": 2}),
        )
        modules = {
            "mscclpp._mscclpp": SimpleNamespace(
                CppRawGpuBuffer=self.allocate,
                CppGpuBufferGranularity=SimpleNamespace(MultiCastMinimum=0),
                is_hip=self.is_hip,
            ),
            "cupy": None,
            "cuda": SimpleNamespace(),
            "cuda.bindings": SimpleNamespace(runtime=self.runtime),
            "hip": SimpleNamespace(hip=self.runtime),
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.buffer_module = load_module("_native_buffer_test", self.python_dir / "mscclpp" / "_core" / "buffer.py")
        self.GpuBuffer = self.buffer_module.GpuBuffer

    def api(self, name):
        return getattr(self.runtime, self.prefix + name)

    def test_allocation_metadata_and_granularity(self):
        buffer = self.GpuBuffer((2, 3), dtype=np.int32, granularity=7)
        self.allocate.assert_called_once_with(24, 7)
        self.assertEqual((buffer.shape, buffer.size, buffer.ndim), ((2, 3), 6, 2))
        self.assertEqual((buffer.itemsize, buffer.nbytes, buffer.strides), (4, 24, (12, 4)))
        self.assertEqual(buffer.allocation_size, 64)
        self.assertEqual(buffer.device_id, 2)
        self.assertEqual(buffer.data.ptr, buffer.ptr)
        with self.assertRaises(AttributeError):
            buffer.shape = (1,)

    def test_round_trip_c_and_fortran_and_noncontiguous_host(self):
        for order in ("C", "F"):
            with self.subTest(order=order):
                host = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
                buffer = self.GpuBuffer(host.shape, dtype=host.dtype, order=order)
                buffer.copy_from_numpy(host)
                upload = self.api("Memcpy").call_args.args
                self.assertEqual((upload[0], upload[2], upload[3]), (buffer.ptr, host.nbytes, 1))
                result = buffer.to_numpy()
                self.assertEqual(self.api("Memcpy").call_args.args, (result.ctypes.data, buffer.ptr, host.nbytes, 2))
                np.testing.assert_array_equal(result, host)
                self.assertEqual(buffer.strides, result.strides)
                self.assertEqual(result.flags.f_contiguous, order == "F")
                self.assertGreaterEqual(self.api("DeviceSynchronize").call_count, 4)

    def test_from_numpy_casts_and_uploads_handle_bytes(self):
        host = memoryview(b"\x00\xff\x10\x00")
        buffer = self.GpuBuffer.from_numpy(host, dtype=np.uint8)
        np.testing.assert_array_equal(buffer.to_numpy(), np.asarray(host))
        matrix = np.asfortranarray(np.arange(6, dtype=np.int16).reshape(2, 3))
        buffer = self.GpuBuffer.from_numpy(matrix)
        self.assertEqual(buffer.strides, matrix.strides)
        np.testing.assert_array_equal(buffer.to_numpy(), matrix)

    def test_views_alias_and_retain_allocation(self):
        host = np.arange(8, dtype=np.int32)
        buffer = self.GpuBuffer.from_numpy(host)
        view = buffer[2:7][1:3]
        self.assertEqual(view.ptr, buffer.ptr + 3 * host.itemsize)
        self.assertEqual(view.nbytes, 2 * host.itemsize)
        view.fill(19)
        host[3:5] = 19
        np.testing.assert_array_equal(buffer.to_numpy(), host)
        owner = weakref.ref(buffer._allocation)
        del buffer
        gc.collect()
        self.assertIsNotNone(owner())
        np.testing.assert_array_equal(view.to_numpy(), host[3:5])
        del view
        gc.collect()
        self.assertIsNone(owner())
        self.assertEqual(self.allocate.call_count, 1)

    def test_fill_preserves_nonzero_and_negative_zero_bits(self):
        buffer = self.GpuBuffer(5, dtype=np.float32)
        for value in (2.5, -0.0, 0.0):
            buffer.fill(value)
            expected = np.full(5, value, dtype=np.float32)
            np.testing.assert_array_equal(buffer.to_numpy().view(np.uint32), expected.view(np.uint32))
        self.api("Memset").assert_called_once_with(buffer.ptr, 0, buffer.nbytes)
        with self.assertRaisesRegex(ValueError, "scalar"):
            buffer.fill([1, 2])

    def test_transfers_restore_device_even_on_failure(self):
        buffer = self.GpuBuffer(4, dtype=np.int32)
        self.device = 6
        buffer.fill(0)
        self.assertEqual(self.device, 6)
        self.assertEqual([call.args for call in self.api("SetDevice").call_args_list], [(2,), (6,)])
        self.api("Memcpy").side_effect = lambda *args: (17,)
        with self.assertRaisesRegex(RuntimeError, self.prefix + "Memcpy failed with error 17"):
            buffer.to_numpy()
        self.assertEqual(self.device, 6)
        self.api("SetDevice").side_effect = lambda device: (18,)
        self.api("Memcpy").reset_mock()
        with self.assertRaisesRegex(RuntimeError, "SetDevice failed"):
            buffer.to_numpy()
        self.api("Memcpy").assert_not_called()

    def test_invalid_layout_and_dtype_fail_before_allocation(self):
        for kwargs in (
            {"shape": (-1,)},
            {"shape": (1.5,)},
            {"shape": "x"},
            {"shape": 4, "dtype": object},
            {"shape": 4, "dtype": "S4"},
            {"shape": 4, "dtype": np.dtype(np.int32).newbyteorder("S")},
            {"shape": 4, "order": "A"},
            {"shape": (2, 3), "dtype": np.int32, "strides": (4, 8)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.GpuBuffer(**kwargs)
        with self.assertRaises(OverflowError):
            self.GpuBuffer((sys.maxsize, 2))
        self.allocate.assert_not_called()

    def test_contiguous_strides_and_scalar_and_empty_shapes(self):
        buffer = self.GpuBuffer((2, 3), np.int32, strides=(4, 8), order="F")
        self.assertEqual(buffer.strides, (4, 8))
        scalar = self.GpuBuffer.from_numpy(np.array(7, dtype=np.int32))
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.to_numpy().item(), 7)
        self.api("Memcpy").reset_mock()
        empty = self.GpuBuffer((0, 3), np.int32)
        self.assertEqual(empty.nbytes, 0)
        self.assertEqual(empty.to_numpy().shape, (0, 3))
        self.assertEqual(self.GpuBuffer(4)[3:1].size, 0)
        self.api("Memcpy").assert_not_called()

    def test_unsupported_array_operations_and_uploads_are_explicit(self):
        buffer = self.GpuBuffer(4, dtype=np.int32)
        with self.assertRaisesRegex(TypeError, "to_numpy"):
            np.asarray(buffer)
        with self.assertRaises(TypeError):
            buffer[0]
        with self.assertRaises(ValueError):
            buffer[::2]
        with self.assertRaises(TypeError):
            self.GpuBuffer((2, 2))[:1]
        for host in (np.ones(3, dtype=np.int32), np.ones(4, dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "shape and dtype"):
                buffer.copy_from_numpy(host)
        self.api("Memcpy").assert_not_called()

    def test_executor_buffers_remain_on_device_and_inplace_views_alias(self):
        mscclpp = SimpleNamespace(
            GpuBuffer=self.GpuBuffer,
            DataType=object,
            Executor=object,
            ExecutionPlan=object,
            PacketType=SimpleNamespace(LL16=1),
            npkit=object,
            env=object,
            CommGroup=object,
        )
        with patch.dict(
            sys.modules,
            {
                "mscclpp": mscclpp,
                "mscclpp.utils": SimpleNamespace(KernelBuilder=object, pack=object),
                "mpi4py": SimpleNamespace(MPI=object),
            },
        ):
            executor = load_module("_buffer_executor_test", self.python_dir / "test" / "executor_test.py")
        for collective in ("allreduce", "allgather", "reducescatter", "sendrecv"):
            for in_place in (False, True):
                with self.subTest(collective=collective, in_place=in_place):
                    inputs, outputs, tests, _ = executor.build_bufs(collective, 128, in_place, np.int32, 1, 4)
                    for buffer in inputs + outputs + tests:
                        self.assertIsInstance(buffer, self.GpuBuffer)
                    for buffer in tests:
                        np.testing.assert_array_equal(buffer.to_numpy(), 0)
                    if in_place and collective == "allgather":
                        self.assertEqual(inputs[0].ptr, outputs[0].ptr + 32)
                    if in_place and collective == "reducescatter":
                        self.assertEqual(outputs[0].ptr, inputs[0].ptr + 32)

    def test_benchmark_reference_downloads_once_and_uploads_input(self):
        mpi = SimpleNamespace(
            COMM_WORLD=SimpleNamespace(size=2, allreduce=lambda value, op: value), LAND=1, MAX=2, SUM=3
        )
        with patch.dict(
            sys.modules,
            {
                "mpi4py": SimpleNamespace(MPI=mpi),
                "mscclpp_benchmark.gpu": SimpleNamespace(device_synchronize=lambda: None),
            },
        ):
            correctness = load_module(
                "_buffer_correctness_test", self.python_dir / "mscclpp_benchmark" / "correctness.py"
            )
        for collective in ("allreduce", "allgather"):
            output = self.GpuBuffer(8 if collective == "allreduce" else 16, np.float32)
            case = SimpleNamespace(
                collective=collective,
                input=output if collective == "allreduce" else output[:8],
                output=output,
                dtype_spec=SimpleNamespace(name="float32", storage_dtype=np.float32, fp8_format=None),
            )
            correctness.fill_case_for_benchmark(case, 0)
            np.testing.assert_array_equal(case.input.to_numpy(), correctness._benchmark_input_values(case, 0))
            inputs = [correctness._correctness_input_values(case, rank, 0) for rank in range(2)]
            expected = sum(inputs) if collective == "allreduce" else np.concatenate(inputs)

            def run(case, config):
                np.testing.assert_array_equal(case.input.to_numpy(), inputs[0])
                case.output.copy_from_numpy(expected)
                return 0

            comm = SimpleNamespace(rank=0, nranks=2, run=run, comm_group=SimpleNamespace(barrier=lambda: None))
            with patch.object(output, "to_numpy", wraps=output.to_numpy) as download:
                stats = correctness.check_correctness(comm, case, None)
                self.assertTrue(stats.ok)
                self.assertEqual(stats.mismatches, 0)
                self.assertEqual(stats.total, expected.size)
                self.assertEqual(download.call_count, 2 if case.input is output else 1)


class HipBufferTests(CudaBufferTests):
    is_hip = True


if __name__ == "__main__":
    unittest.main()
