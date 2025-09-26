# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

import os
import warnings
from functools import wraps

# Get version - try multiple sources in order of preference
def _get_version():
    """Get version from the best available source"""

    # Method 1: Try setuptools-scm generated _version.py (most reliable)
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        pass

    # Method 2: Try package metadata
    try:
        from importlib.metadata import version
        return version("mscclpp")
    except Exception:
        pass

    # Method 3: Generate from git (fallback)
    try:
        import subprocess

        # Get git describe output
        result = subprocess.check_output(
            ["git", "describe", "--tags", "--long", "--dirty"],
            text=True, stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).strip()

        # Convert to setuptools-scm format
        if "-" in result:
            parts = result.split("-")
            if len(parts) >= 3:
                tag = parts[0].lstrip('v')
                distance = parts[1]
                commit = parts[2]

                if distance != "0":
                    # Increment patch for development versions
                    tag_parts = tag.split('.')
                    if len(tag_parts) >= 3:
                        patch = int(tag_parts[2]) + 1
                        base = f"{tag_parts[0]}.{tag_parts[1]}.{patch}"
                    else:
                        base = tag
                    version_str = f"{base}.dev{distance}+{commit}"
                else:
                    version_str = f"{tag}+{commit}"

                if "dirty" in result:
                    version_str += ".dirty"

                return version_str

        return result.lstrip('v')

    except Exception:
        pass

    # Final fallback
    return "0.7.0+unknown"

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

# Get additional git information at runtime (from the source repository if available)
def _get_git_info():
    """Get additional git information from source repository if available"""
    try:
        import subprocess

        # Try to find the source repository
        # First check if we're in a development installation
        potential_paths = [
            os.path.dirname(__file__),  # Current directory
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up two levels
            "/home/qinghuazhou/mscclpp_pip_version",  # Known development path
        ]

        for path in potential_paths:
            if os.path.exists(os.path.join(path, ".git")):
                def git_cmd(cmd):
                    try:
                        return subprocess.check_output(
                            cmd, text=True, stderr=subprocess.DEVNULL,
                            cwd=path
                        ).strip()
                    except:
                        return None

                branch = git_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                remote = git_cmd(["git", "config", "--get", "remote.origin.url"])
                commit_full = git_cmd(["git", "rev-parse", "HEAD"])

                if branch or remote or commit_full:
                    return {
                        "branch": branch or "unknown",
                        "remote": remote or "unknown",
                        "commit_full": commit_full or "unknown"
                    }

        # If we can't find a git repository but have a commit in the version,
        # try to construct the full commit hash
        if __git_commit__ != "unknown":
            # The short commit is usually the first 7-9 characters
            # We can't reconstruct the full hash, but we can indicate we have partial info
            return {
                "branch": "unknown",
                "remote": "unknown",
                "commit_full": f"{__git_commit__}... (partial)"
            }

        return {
            "branch": "unknown",
            "remote": "unknown",
            "commit_full": "unknown"
        }
    except Exception:
        return {
            "branch": "unknown",
            "remote": "unknown",
            "commit_full": "unknown"
        }

_git_info = _get_git_info()
__git_branch__ = _git_info["branch"]
__git_remote__ = _git_info["remote"]
__git_commit_full__ = _git_info["commit_full"]
__scm_version__ = __version__

def get_version_info():
    """Get complete version information as a dictionary"""
    return {
        "version": __version__,
        "base_version": __base_version__,
        "commit": __git_commit__,
        "commit_full": __git_commit_full__,
        "branch": __git_branch__,
        "remote": __git_remote__,
        "dirty": __git_dirty__,
        "distance": __git_distance__,
        "scm_version": __scm_version__
    }

def show_version(verbose=True):
    """Display version information"""
    info = get_version_info()
    if verbose:
        print("MSCCLPP Version Information:")
        print(f"  Package Version: {info['version']}")
        print(f"  Base Version: {info['base_version']}")
        print(f"  Git Commit (short): {info['commit']}")
        print(f"  Git Commit (full): {info['commit_full']}")
        print(f"  Git Branch: {info['branch']}")
        print(f"  Git Remote: {info['remote']}")
        print(f"  Working Tree Dirty: {info['dirty']}")
        print(f"  Distance from Tag: {info['distance']}")
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
