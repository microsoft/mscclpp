# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of native flag-buffer caching and ownership without CuPy."""

import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import weakref


class _Owner:
    pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlagBufferTests(unittest.TestCase):
    def setUp(self):
        self.ptr = (1 << 40) + 123
        self.nbytes = 512
        self.owners = []

        def allocate():
            owner = _Owner()
            self.owners.append(weakref.ref(owner))
            return self.ptr, self.nbytes, owner

        self.get_native_buffer = Mock(side_effect=allocate)
        self.native_collection = Mock()
        self.native_collection.to_list.return_value = []
        self.native_builder = Mock()
        self.native_builder.build_default_algorithms.return_value = self.native_collection
        self.builder_type = Mock()
        self.builder_type.get_instance.return_value = self.native_builder
        native = SimpleNamespace(
            **{
                name: object
                for name in (
                    "CppAlgorithm",
                    "CppDslAlgorithm",
                    "CppAlgorithmType",
                    "CppCommunicator",
                    "CppCollectiveBufferMode",
                    "CppDataType",
                    "CppExecutor",
                    "CppExecutionPlan",
                    "CppAlgorithmBuilder",
                    "CppAlgorithmCollection",
                )
            },
            CppReduceOp=SimpleNamespace(NOP=0),
            CppAlgorithmCollectionBuilder=self.builder_type,
            cpp_get_flag_buffer=self.get_native_buffer,
        )
        root = Path(__file__).resolve().parents[1] / "mscclpp"
        with patch.dict(sys.modules, {"mscclpp._mscclpp": native, "cupy": None}):
            self.algorithm = _load_module("_mscclpp_flag_buffer_test", root / "_core" / "algorithm.py")
            with patch.dict(sys.modules, {"mscclpp._core.algorithm": self.algorithm}), patch("atexit.register"):
                self.builder_module = _load_module(
                    "_mscclpp_flag_builder_test", root / "ext" / "algorithm_collection_builder.py"
                )

    def test_native_tuple_is_cached_without_cupy(self):
        buffer = self.algorithm.get_flag_buffer()
        self.assertEqual(buffer[:2], (self.ptr, self.nbytes))
        self.assertIs(buffer[2], self.owners[0]())
        self.assertIs(self.algorithm.get_flag_buffer(), buffer)
        self.get_native_buffer.assert_called_once_with()

    def test_cache_retains_native_owner(self):
        self.algorithm.get_flag_buffer()
        gc.collect()
        self.assertIsNotNone(self.owners[0]())
        self.algorithm._flag_buffer_cache = None
        gc.collect()
        self.assertIsNone(self.owners[0]())

    def test_builder_forwards_pointer_and_byte_size_and_reuses_buffer(self):
        builder = self.builder_module.AlgorithmCollectionBuilder()
        collection = builder.build_default_algorithms("4096", 8192, 3)
        builder.build_default_algorithms(16384, 32768, 1)
        self.assertIs(collection._native_collection, self.native_collection)
        self.assertIs(builder._flag_buffer, self.algorithm.get_flag_buffer())
        self.get_native_buffer.assert_called_once_with()
        self.assertEqual(
            self.native_builder.build_default_algorithms.call_args_list,
            [
                call(4096, 8192, self.ptr, self.nbytes, 3),
                call(16384, 32768, self.ptr, self.nbytes, 1),
            ],
        )

    def test_builder_retains_owner_if_module_cache_is_cleared(self):
        builder = self.builder_module.AlgorithmCollectionBuilder()
        builder.build_default_algorithms(4096, 8192, 0)
        self.algorithm._flag_buffer_cache = None
        gc.collect()
        self.assertIsNotNone(self.owners[0]())
        builder._flag_buffer = None
        gc.collect()
        self.assertIsNone(self.owners[0]())

    def test_builder_reset_preserves_shared_cache(self):
        builder_type = self.builder_module.AlgorithmCollectionBuilder
        builder_type().build_default_algorithms(4096, 8192, 0)
        builder_type.reset()
        self.builder_type.reset.assert_called_once_with()
        builder_type().build_default_algorithms(4096, 8192, 0)
        self.get_native_buffer.assert_called_once_with()
        self.assertIsNotNone(self.owners[0]())

    def test_native_failure_leaves_cache_empty_and_can_be_retried(self):
        allocate = self.get_native_buffer.side_effect
        self.get_native_buffer.side_effect = RuntimeError("Flag allocation failed")
        with self.assertRaisesRegex(RuntimeError, "Flag allocation failed"):
            self.algorithm.get_flag_buffer()
        self.assertIsNone(self.algorithm._flag_buffer_cache)
        self.get_native_buffer.side_effect = allocate
        self.assertEqual(self.algorithm.get_flag_buffer()[:2], (self.ptr, self.nbytes))
        self.assertEqual(self.get_native_buffer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
