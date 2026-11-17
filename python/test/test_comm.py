# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of memory registration without CuPy."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch, sentinel


class _NumpyArray:
    def __init__(self):
        self.ctypes = SimpleNamespace(data=123)
        self.size = 7
        self.itemsize = 4


class _TorchTensor:
    def data_ptr(self):
        return 123

    def numel(self):
        return 7

    def element_size(self):
        return 4


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommGroupTests(unittest.TestCase):
    def setUp(self):
        native = SimpleNamespace(
            **{
                name: object
                for name in (
                    "CppDataType",
                    "CppCommunicator",
                    "CppConnection",
                    "connect_nvls_collective",
                    "CppEndpointConfig",
                    "CppSemaphore",
                    "CppProxyService",
                    "CppRegisteredMemory",
                    "CppPortChannel",
                    "CppTcpBootstrap",
                    "CppTransport",
                )
            }
        )
        native.is_hip = False
        self.memory_channel = Mock(side_effect=lambda *args: args)
        native.CppMemoryChannel = self.memory_channel
        native.CppTransportFlags = Mock(return_value=0)
        root = Path(__file__).resolve().parents[1] / "mscclpp"
        modules = {
            "mscclpp._mscclpp": native,
            "numpy": SimpleNamespace(ndarray=_NumpyArray),
            "torch": SimpleNamespace(Tensor=_TorchTensor),
            "cupy": None,
        }
        with patch.dict(sys.modules, modules):
            utils = _load_module("_mscclpp_comm_utils_test", root / "utils.py")
            with patch.dict(sys.modules, {"mscclpp.utils": utils}):
                self.comm = _load_module("_mscclpp_comm_test", root / "_core" / "comm.py")
        self.connections = {1: Mock(), 2: Mock()}
        self.connections[1].transport.return_value = 1
        self.connections[2].transport.return_value = 4
        self.scratch = Mock()
        self.scratch.data.return_value = 456

    def buffers(self):
        return (
            _NumpyArray(),
            _TorchTensor(),
            SimpleNamespace(data=SimpleNamespace(ptr=123), size=7, itemsize=4),
        )

    def make_group(self):
        group = self.comm.CommGroup.__new__(self.comm.CommGroup)
        group.my_rank = 0
        group.communicator = Mock()
        group.communicator.register_memory.return_value = sentinel.local_memory
        group.make_semaphores = Mock(return_value={1: sentinel.sem1, 2: sentinel.sem2})
        group._register_memory_with_connections = Mock(
            return_value={0: self.scratch, 1: sentinel.remote1, 2: sentinel.remote2}
        )
        self.memory_channel.reset_mock()
        return group

    def test_register_local_memory_preserves_pointer_size_and_transports(self):
        for buffer in self.buffers():
            with self.subTest(buffer=type(buffer)):
                group = self.make_group()
                result = group.register_local_memory(buffer, self.connections)
                self.assertIs(result, sentinel.local_memory)
                group.communicator.register_memory.assert_called_once_with(123, 28, 5)

    def test_registration_without_connections_uses_empty_transport_flags(self):
        group = self.make_group()
        group.register_local_memory(_NumpyArray(), {})
        group.communicator.register_memory.assert_called_once_with(123, 28, 0)

    def test_memory_channels_preserve_scratch_and_empty_transport_flags(self):
        for buffer in self.buffers():
            with self.subTest(buffer=type(buffer)):
                group = self.make_group()
                channels = group.make_memory_channels_with_scratch(buffer, self.scratch, self.connections)
                group.communicator.register_memory.assert_called_once_with(123, 28, 0)
                group.make_semaphores.assert_called_once_with(self.connections)
                group._register_memory_with_connections.assert_called_once_with(self.scratch, self.connections)
                self.assertEqual(
                    channels,
                    {
                        1: (sentinel.sem1, sentinel.remote1, sentinel.local_memory, 456),
                        2: (sentinel.sem2, sentinel.remote2, sentinel.local_memory, 456),
                    },
                )

    def test_port_channels_preserve_registration_and_proxy_mapping(self):
        for buffer in self.buffers():
            with self.subTest(buffer=type(buffer)):
                group = self.make_group()
                proxy = Mock()
                proxy.add_memory.side_effect = [10, 11, 12]
                proxy.add_semaphore.side_effect = [21, 22]
                proxy.port_channel.side_effect = [sentinel.channel1, sentinel.channel2]
                channels = group.make_port_channels_with_scratch(proxy, buffer, self.scratch, self.connections)
                group.communicator.register_memory.assert_called_once_with(123, 28, 5)
                group.make_semaphores.assert_called_once_with(self.connections)
                group._register_memory_with_connections.assert_called_once_with(self.scratch, self.connections)
                self.assertEqual(
                    proxy.add_memory.call_args_list,
                    [call(sentinel.local_memory), call(sentinel.remote1), call(sentinel.remote2)],
                )
                self.assertEqual(proxy.add_semaphore.call_args_list, [call(sentinel.sem1), call(sentinel.sem2)])
                self.assertEqual(proxy.port_channel.call_args_list, [call(21, 11, 10), call(22, 12, 10)])
                self.assertEqual(channels, {1: sentinel.channel1, 2: sentinel.channel2})

    def test_invalid_pointers_never_reach_native_registration(self):
        for method in ("register_local_memory", "make_memory_channels_with_scratch", "make_port_channels_with_scratch"):
            with self.subTest(method=method):
                group = self.make_group()
                buffer = SimpleNamespace(data=SimpleNamespace(ptr=-1), size=7, itemsize=4)
                with self.assertRaisesRegex(RuntimeError, "Invalid data pointer"):
                    if method == "register_local_memory":
                        group.register_local_memory(buffer, self.connections)
                    elif method == "make_memory_channels_with_scratch":
                        group.make_memory_channels_with_scratch(buffer, self.scratch, self.connections)
                    else:
                        group.make_port_channels_with_scratch(Mock(), buffer, self.scratch, self.connections)
                group.communicator.register_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
