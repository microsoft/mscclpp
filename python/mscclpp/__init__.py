# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

import os
import warnings
from functools import wraps

from ._mscclpp import (
    Env,
    ErrorCode,
    BaseError,
    Error,
    SysError,
    CudaError,
    CuError,
    IbError,
    Device,
    DeviceType,
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
    PortChannel,
    MemoryChannel,
    MemoryDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    TransportFlags,
    DataType,
    Executor,
    ExecutionPlan,
    ExecutionPlanHandle,
    ExecutionPlanRegistry,
    ExecutionRequest,
    PacketType,
    RawGpuBuffer,
    env,
    version,
    is_nvls_supported,
    npkit,
)


__all__ = [
    "Device",
    "DeviceType",
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
    "PortChannel",
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
    "ProxyChannel",
    "SmChannel",
    "SmDevice2DeviceSemaphore",
]

__version__: str = str(version())

if os.environ.get("MSCCLPP_HOME", None) is None:
    os.environ["MSCCLPP_HOME"] = os.path.abspath(os.path.dirname(__file__))


def get_include() -> str:
    """Return the directory that contains the MSCCL++ headers."""
    return os.path.join(os.path.dirname(__file__), "include")


def get_lib() -> str:
    """Return the directory that contains the MSCCL++ headers."""
    return os.path.join(os.path.dirname(__file__), "lib")


def deprecated(new_cls):
    def decorator(old_cls):
        @wraps(old_cls)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{old_cls.__name__} is deprecated, use {new_cls.__name__} instead.",
                DeprecationWarning,
            )
            return new_cls(*args, **kwargs)

        return wrapper

    return decorator


@deprecated(PortChannel)
class ProxyChannel(PortChannel):
    pass


@deprecated(MemoryChannel)
class SmChannel(MemoryChannel):
    pass


@deprecated(MemoryDevice2DeviceSemaphore)
class SmDevice2DeviceSemaphore(MemoryDevice2DeviceSemaphore):
    pass
