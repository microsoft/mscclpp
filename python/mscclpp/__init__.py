# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

import os
import warnings
from functools import wraps

# Import version information first
try:
    from ._version import (
        __version__,
        __base_version__,
        __git_commit__,
        __git_branch__,
        __git_remote__,
        __git_dirty__,
        __git_distance__,  # Add this
        __scm_version__,
        get_version_info,
        show_version
    )

except ImportError:
    # Fallback for development or if _version.py doesn't exist
    from ._mscclpp import version
    __version__ = str(version())
    __base_version__ = __version__
    __git_commit__ = "unknown"
    __git_branch__ = "unknown"
    __git_remote__ = "unknown"
    __git_dirty__ = False
    __git_distance__ = 0  # Add this
    __scm_version__ = None

    def get_version_info():
        return {
            "version": __version__,
            "base_version": __base_version__,
            "commit": __git_commit__,
            "branch": __git_branch__,
            "remote": __git_remote__,
            "dirty": __git_dirty__,
            "distance": __git_distance__,  # Add this
            "scm_version": __scm_version__
        }

    def show_version(verbose=True):
        info = get_version_info()
        if verbose:
            print(f"MSCCLPP Version: {info['version']}")
        return info

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
    # Version information
    "__version__",
    "__base_version__",
    "__git_commit__",
    "__git_branch__",
    "__git_remote__",
    "__git_dirty__",
    "__git_distance__",
    "__scm_version__",
    "get_version_info",
    "show_version",
    "get_include",
    "get_lib",
    ### Deprecated ###
    "ProxyChannel",
    "SmChannel",
    "SmDevice2DeviceSemaphore",
]

# Remove the old __version__ assignment since it's now imported from _version.py

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
