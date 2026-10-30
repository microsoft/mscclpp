# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of the benchmark's NCCL/RCCL C API boundary."""

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class NcclTests(unittest.TestCase):
    is_hip = False

    def setUp(self):
        with patch.dict(
            sys.modules,
            {
                "mscclpp": SimpleNamespace(GpuBuffer=object),
                "mscclpp._mscclpp": SimpleNamespace(is_hip=self.is_hip),
                "cupy": None,
            },
        ):
            path = Path(__file__).resolve().parents[1] / "mscclpp_benchmark" / "nccl_op.py"
            spec = importlib.util.spec_from_file_location("_nccl_op_test", path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.uid = bytes(range(128))
        self.handle = 0x123456781234

        def unique_id(target):
            ctypes.memmove(target, self.uid, 128)
            return 0

        def initialize(target, nranks, uid, rank):
            self.assertEqual(bytes(uid), self.uid)
            ctypes.cast(target, ctypes.POINTER(ctypes.c_void_p))[0] = self.handle
            return 0

        self.lib = SimpleNamespace()
        for name in ("ncclGetUniqueId", "ncclCommInitRank", "ncclCommDestroy", "ncclCommAbort", "ncclAllReduce"):
            setattr(self.lib, name, Mock(__name__=name, return_value=0))
        self.lib.ncclGetUniqueId.side_effect = unique_id
        self.lib.ncclCommInitRank.side_effect = initialize
        self.lib.ncclGetErrorString = Mock(return_value=b"native error")
        loader_patch = patch.object(self.module.ctypes, "CDLL", return_value=self.lib)
        self.loader = loader_patch.start()
        self.addCleanup(loader_patch.stop)
        environment_patch = patch.dict(os.environ)
        environment_patch.start()
        os.environ.pop("MSCCLPP_BENCH_NCCL_LIBRARY", None)
        self.addCleanup(environment_patch.stop)

    def test_library_selection_and_unique_id_preserve_embedded_nulls(self):
        self.assertEqual(self.module.get_unique_id(), self.uid)
        self.assertEqual(self.module.get_unique_id(), self.uid)
        self.loader.assert_called_once_with("librccl.so.1" if self.is_hip else "libnccl.so.2")
        self.assertEqual(ctypes.sizeof(self.module._UniqueId), 128)

    def test_library_override_and_load_error(self):
        os.environ["MSCCLPP_BENCH_NCCL_LIBRARY"] = "/vendor/lib.so"
        self.module.get_unique_id()
        self.loader.assert_called_once_with("/vendor/lib.so")
        self.module._library.cache_clear()
        self.loader.side_effect = OSError("missing")
        with self.assertRaisesRegex(RuntimeError, "Install NCCL/RCCL"):
            self.module.get_unique_id()

    def test_checked_unique_id_and_initialization_errors(self):
        self.lib.ncclGetUniqueId.side_effect = lambda target: 3
        with self.assertRaisesRegex(RuntimeError, "ncclGetUniqueId failed \\(3\\): native error"):
            self.module.get_unique_id()
        self.lib.ncclCommInitRank.side_effect = lambda *args: 4
        with self.assertRaisesRegex(RuntimeError, "ncclCommInitRank failed"):
            self.module.NcclCommunicator(2, self.uid, 0)
        self.lib.ncclCommDestroy.assert_not_called()

    def test_c_api_signatures_and_64_bit_arguments(self):
        received = []
        signature = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.lib.ncclAllReduce = signature(lambda *args: received.append(args) or 0)
        self.lib.ncclAllReduce.__name__ = "ncclAllReduce"
        with self.module.NcclCommunicator(2, self.uid, 1) as comm:
            self.assertEqual(
                self.lib.ncclCommInitRank.argtypes,
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, self.module._UniqueId, ctypes.c_int],
            )
            self.assertIs(self.lib.ncclGetErrorString.restype, ctypes.c_char_p)
            memory = SimpleNamespace(dtype=np.dtype("float32"), data=SimpleNamespace(ptr=0x234567892345), size=2**33)
            call = self.module.NcclAllReduce(comm, memory)
            self.assertIs(call(0x3456789A3456), memory)
            self.assertEqual(
                received[0],
                (memory.data.ptr, memory.data.ptr, 2**33, 7, 0, self.handle, 0x3456789A3456),
            )
        self.lib.ncclCommDestroy.assert_called_once()
        comm.close()
        self.lib.ncclCommDestroy.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            call(None)

    def test_dtype_mapping_and_default_stream(self):
        with self.module.NcclCommunicator(2, self.uid, 0) as comm:
            for dtype, enum in ((np.float16, 6), (np.float32, 7), (np.int32, 2)):
                memory = SimpleNamespace(dtype=np.dtype(dtype), data=SimpleNamespace(ptr=1234), size=8)
                self.module.NcclAllReduce(comm, memory)(None)
                args = self.lib.ncclAllReduce.call_args.args
                self.assertEqual(args[:5], (1234, 1234, 8, enum, 0))
                self.assertEqual(args[-1], 0)
            with self.assertRaisesRegex(RuntimeError, "data type"):
                self.module.NcclAllReduce(comm, SimpleNamespace(dtype=np.uint8))

    def test_context_aborts_on_collective_failure(self):
        self.lib.ncclAllReduce.return_value = 5
        memory = SimpleNamespace(dtype=np.float32, data=SimpleNamespace(ptr=1234), size=8)
        with self.assertRaisesRegex(RuntimeError, "ncclAllReduce failed"):
            with self.module.NcclCommunicator(2, self.uid, 0) as comm:
                self.module.NcclAllReduce(comm, memory)(None)
        self.lib.ncclCommAbort.assert_called_once()
        self.lib.ncclCommDestroy.assert_not_called()

    def test_destroy_failure_is_reported_and_can_be_retried(self):
        comm = self.module.NcclCommunicator(2, self.uid, 0)
        self.lib.ncclCommDestroy.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "ncclCommDestroy failed"):
            comm.close()
        self.lib.ncclCommDestroy.return_value = 0
        comm.close()
        self.assertEqual(self.lib.ncclCommDestroy.call_count, 2)

    def test_invalid_rank_and_id_do_not_call_native_code(self):
        for nranks, uid, rank in ((0, self.uid, 0), (2, self.uid, 2), (2, b"short", 0)):
            with self.assertRaises(ValueError):
                self.module.NcclCommunicator(nranks, uid, rank)
        self.loader.assert_not_called()


class RcclTests(NcclTests):
    is_hip = True


if __name__ == "__main__":
    unittest.main()
