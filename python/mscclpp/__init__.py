# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

import os
import warnings
from functools import wraps

# Get version
def _get_version():
    """Get version from the best available source"""

    # Try setuptools-scm generated _version.py (most reliable)
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        raise RuntimeError("Could not determine MSCCL++ version from setuptools-scm generated _version.py.")

__version__ = _get_version()

# Parse version components
import re

def _parse_version(version_string):
    """Parse version components from setuptools-scm generated version"""
    # Pattern for versions like "0.7.1.dev36+g6e2360d69" (without .dYYYYMMDD)
    pattern = r"^v?(?P<base>[\d\.]+)(?:\.dev(?P<distance>\d+))?(?:\+g(?P<commit>[a-f0-9]+))?(?P<dirty>\.dirty)?$"
    match = re.match(pattern, version_string)

    if match:
        return {
            "base_version": match.group("base"),
            "git_commit": match.group("commit") or "unknown",
            "git_distance": int(match.group("distance") or 0),
            "git_dirty": bool(match.group("dirty"))
        }
    else:
        # Fallback parsing - try to extract what we can
        base = version_string.split("+")[0].lstrip("v").split(".dev")[0]

        # Try to extract commit hash
        commit = "unknown"
        if "+g" in version_string:
            commit_match = re.search(r"\+g([a-f0-9]+)", version_string)
            if commit_match:
                commit = commit_match.group(1)

        # Try to extract dev distance
        distance = 0
        if ".dev" in version_string:
            dev_match = re.search(r"\.dev(\d+)", version_string)
            if dev_match:
                distance = int(dev_match.group(1))

        # Check if dirty
        dirty = ".dirty" in version_string

        return {
            "base_version": base,
            "git_commit": commit,
            "git_distance": distance,
            "git_dirty": dirty
        }

# Parse the version
_version_info = _parse_version(__version__)
__base_version__ = _version_info["base_version"]
__git_commit__ = _version_info["git_commit"]
__git_distance__ = _version_info["git_distance"]
__git_dirty__ = _version_info["git_dirty"]

def get_version_info():
    """Get complete version information as a dictionary"""
    return {
        "version": __version__,
        "base_version": __base_version__,
        "commit": __git_commit__,
        "dirty": __git_dirty__,
        "distance": __git_distance__,
    }

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
