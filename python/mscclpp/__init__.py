# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import warnings

from ._mscclpp import (
    Communicator,
    Connection,
    connect_nvls_collective,
    EndpointConfig,
    Fifo,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    numa,
    ProxyService,
    RegisteredMemory,
    ProxyChannel,
    MemoryChannel,
    MemoryDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    TransportFlags,
    DataType,
    Executor,
    ExecutionPlan,
    PacketType,
    version,
    PyGpuBuffer,
    is_nvls_supported,
    npkit,
)


__all__ = [
    "Communicator",
    "Connection",
    "connect_nvls_collective",
    "EndpointConfig",
    "Fifo",
    "Host2DeviceSemaphore",
    "Host2HostSemaphore",
    "numa",
    "ProxyService",
    "RegisteredMemory",
    "ProxyChannel",
    "MemoryChannel",
    "MemoryDevice2DeviceSemaphore",
    "TcpBootstrap",
    "Transport",
    "TransportFlags",
    "DataType",
    "Executor",
    "ExecutionPlan",
    "PacketType",
    "version",
    "is_nvls_supported",
    "alloc_shared_physical_cuda",
    "npkit",
    "__version__",
    "get_include",
    "get_lib",
    ### Deprecated ###
    "SmChannel",
    "SmDevice2DeviceSemaphore",
]

__version__ = version()

if os.environ.get("MSCCLPP_HOME", None) is None:
    os.environ["MSCCLPP_HOME"] = os.path.abspath(os.path.dirname(__file__))


def get_include():
    """Return the directory that contains the MSCCL++ headers."""
    return os.path.join(os.path.dirname(__file__), "include")


def get_lib():
    """Return the directory that contains the MSCCL++ headers."""
    return os.path.join(os.path.dirname(__file__), "lib")


class MetaDeprecated(type):
    def __new__(cls, name, bases, class_dict):
        new_class = super().__new__(cls, name, bases, class_dict)

        # Override the __init__ method to add a deprecation warning
        original_init = new_class.__init__

        def new_init(self, *args, **kwargs):
            warnings.warn(f"{name} is deprecated, use {bases[0].__name__} instead.", DeprecationWarning, stacklevel=2)
            original_init(self, *args, **kwargs)

        new_class.__init__ = new_init
        return new_class


class SmChannel(MemoryChannel, metaclass=MetaDeprecated):
    pass


class SmDevice2DeviceSemaphore(MemoryDevice2DeviceSemaphore, metaclass=MetaDeprecated):
    pass
