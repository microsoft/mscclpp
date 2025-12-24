# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

from functools import wraps
import os
import warnings


from functools import wraps
from mscclpp._version import __version__, __commit_id__
from mscclpp._algorithm import Algorithm, AlgorithmCollectionBuilder, AlgorithmCollection
from mscclpp.language.utils import AlgoSpec
from mscclpp._compiler import DslCompiler, NativeCodeCompiler

if os.environ.get("MSCCLPP_HOME", None) is None:
    os.environ["MSCCLPP_HOME"] = os.path.abspath(os.path.dirname(__file__))


# Parse the version
version = {
    "version": __version__,
    "git_commit": __commit_id__,
}


from ._mscclpp import (
    Device,
    DeviceType,
    Communicator,
    Connection,
    connect_nvls_collective,
    EndpointConfig,
    Fifo,
    Semaphore,
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
    ErrorCode,
    Executor,
    ExecutionPlan,
    PacketType,
    RawGpuBuffer,
    ReduceOp,
    env,
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
    "Semaphore",
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
    "RawGpuBuffer",
    "env",
    "version",
    "is_nvls_supported",
    "alloc_shared_physical_cuda",
    "npkit",
    # Version information
    "__version__",
    "version",
    "get_include",
    "get_lib",
    # Python API
    "Algorithm",
    "AlgorithmCollectionBuilder",
    "AlgorithmCollection",
    "AlgoSpec",
    "ReduceOp",
]


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


compile: DslCompiler = DslCompiler()
compile_native: NativeCodeCompiler = NativeCodeCompiler()
