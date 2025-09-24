# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python bindings with version tracking"""

import os
import warnings
from functools import wraps

# Import version information first (should always work)
try:
    from ._version import (
        __version__,
        __base_version__,
        __git_commit__,
        __git_branch__,
        __git_remote__,
        get_version_info,
        show_version
    )
except ImportError:
    # Fallback if version file doesn't exist
    __version__ = "0.7.0"
    __base_version__ = "0.7.0"
    __git_commit__ = "unknown"
    __git_branch__ = "unknown"
    __git_remote__ = "unknown"
    
    def get_version_info():
        """Fallback version info"""
        return {
            "version": __version__,
            "base_version": __base_version__,
            "commit": __git_commit__,
            "branch": __git_branch__,
            "remote": __git_remote__
        }
    
    def show_version(verbose=True):
        """Fallback version display"""
        info = get_version_info()
        if verbose:
            print("MSCCL++ Version Information:")
            print(f"  Package Version: {info['version']}")
            print(f"  Base Version: {info['base_version']}")
            print(f"  Git Commit: {info['commit']}")
            print(f"  Git Branch: {info['branch']}")
            print(f"  Git Remote: {info['remote']}")
        return info

# Try to import the C++ extension
_cpp_extension_available = False
_cpp_import_error = None

try:
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
    _cpp_extension_available = True
except ImportError as e:
    _cpp_import_error = str(e)
    
    # Define stub version function if C++ extension is not available
    def version():
        """Return version from Python if C++ extension is not available"""
        return __version__
    
    def _warn_cpp_not_available():
        warnings.warn(
            f"MSCCL++ C++ extension is not available. "
            f"Please build the package first using 'pip install .' "
            f"Error: {_cpp_import_error}",
            ImportWarning,
            stacklevel=2
        )

# Also try the alternate import name
if not _cpp_extension_available:
    try:
        from .mscclpp_py import *
        _cpp_extension_available = True
        from .mscclpp_py import version
    except ImportError:
        pass  # Keep the error from _mscclpp import

# Set MSCCLPP_HOME environment variable
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


# Define deprecated classes only if C++ extension is available
if _cpp_extension_available:
    @deprecated(PortChannel)
    class ProxyChannel(PortChannel):
        pass

    @deprecated(MemoryChannel)
    class SmChannel(MemoryChannel):
        pass

    @deprecated(MemoryDevice2DeviceSemaphore)
    class SmDevice2DeviceSemaphore(MemoryDevice2DeviceSemaphore):
        pass
    
    # Also set the C++ version as a fallback
    if __version__ == "unknown" and callable(version):
        try:
            __version__ = str(version())
        except:
            pass

# Define __all__ for exports
__all__ = [
    # Version tracking
    "__version__",
    "__git_commit__", 
    "__git_branch__",
    "__git_remote__",
    "show_version", 
    "get_version_info",
    # Utility functions
    "get_include",
    "get_lib",
]

# Add C++ exports if available
if _cpp_extension_available:
    __all__.extend([
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
        "npkit",
        # Deprecated classes
        "ProxyChannel",
        "SmChannel",
        "SmDevice2DeviceSemaphore",
    ])
