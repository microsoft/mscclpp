# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of kernel argument packing without CuPy."""

import ctypes
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class _NumpyArray:
    def __init__(self, ptr):
        self.ctypes = SimpleNamespace(data=ptr)


class _TorchTensor:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class PackTests(unittest.TestCase):
    def load_utils(self, torch_available=True):
        modules = {
            "mscclpp._mscclpp": SimpleNamespace(CppDataType=object, is_hip=False),
            "numpy": SimpleNamespace(ndarray=_NumpyArray),
            "torch": SimpleNamespace(Tensor=_TorchTensor) if torch_available else None,
            "cupy": None,
        }
        path = Path(__file__).resolve().parents[1] / "mscclpp" / "utils.py"
        spec = importlib.util.spec_from_file_location("_mscclpp_pack_utils_test", path)
        utils = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(utils)
        return utils

    def setUp(self):
        self.utils = self.load_utils()

    def test_empty_arguments(self):
        self.assertEqual(self.utils.pack(), b"")

    def test_scalar_and_byte_layout_is_unchanged(self):
        size = (1 << (8 * ctypes.sizeof(ctypes.c_size_t) - 1)) + 7
        self.assertEqual(
            self.utils.pack(-3, True, False, ctypes.c_size_t(size), b"\x00\xff"),
            struct.pack("i", -3)
            + struct.pack("i", 1)
            + struct.pack("i", 0)
            + struct.pack("N", size)
            + b"\x00\xff",
        )

    def test_numpy_and_torch_pointers(self):
        storage = (ctypes.c_int * 4)()
        ptr = ctypes.addressof(storage)
        for array in (_NumpyArray(ptr), _TorchTensor(ptr)):
            with self.subTest(array=type(array)):
                self.assertEqual(self.utils.pack(array), struct.pack("P", ptr))

    def test_gpu_pointer_protocol_without_cuda_array_interface(self):
        for ptr in (0, (1 << (8 * struct.calcsize("P") - 1)) + 123):
            with self.subTest(ptr=ptr):
                array = SimpleNamespace(data=SimpleNamespace(ptr=ptr))
                self.assertEqual(self.utils.pack(array), struct.pack("P", ptr))

    def test_numpy_and_torch_take_priority_over_generic_pointer(self):
        for array in (_NumpyArray(123), _TorchTensor(123)):
            with self.subTest(array=type(array)):
                array.data = SimpleNamespace(ptr=456)
                self.assertEqual(self.utils.pack(array), struct.pack("P", 123))

    def test_mixed_arguments_preserve_order_and_offsets(self):
        gpu_array = SimpleNamespace(data=SimpleNamespace(ptr=789))
        self.assertEqual(
            self.utils.pack(1, _NumpyArray(123), b"x", _TorchTensor(456), gpu_array, ctypes.c_size_t(8)),
            struct.pack("i", 1)
            + struct.pack("P", 123)
            + b"x"
            + struct.pack("P", 456)
            + struct.pack("P", 789)
            + struct.pack("N", 8),
        )

    def test_packing_without_torch(self):
        utils = self.load_utils(torch_available=False)
        gpu_array = SimpleNamespace(data=SimpleNamespace(ptr=456))
        self.assertEqual(
            utils.pack(_NumpyArray(123), gpu_array),
            struct.pack("P", 123) + struct.pack("P", 456),
        )
        with self.assertRaisesRegex(RuntimeError, "Unsupported type"):
            utils.pack(_TorchTensor(123))

    def test_unsupported_objects_raise(self):
        for value in (None, object(), 1.5, "text", [], SimpleNamespace(data=object())):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "Unsupported type"):
                    self.utils.pack(value)

    def test_invalid_gpu_pointers_raise(self):
        for ptr in ("123", 1.5, True, -1, 1 << (8 * struct.calcsize("P"))):
            with self.subTest(ptr=ptr):
                with self.assertRaisesRegex(RuntimeError, "Invalid data pointer"):
                    self.utils.pack(SimpleNamespace(data=SimpleNamespace(ptr=ptr)))

    def test_pointer_property_errors_are_not_hidden(self):
        class BrokenArray:
            @property
            def data(self):
                raise RuntimeError("Buffer is no longer valid")

        with self.assertRaisesRegex(RuntimeError, "Buffer is no longer valid"):
            self.utils.pack(BrokenArray())


if __name__ == "__main__":
    unittest.main()
