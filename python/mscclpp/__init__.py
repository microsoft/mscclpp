# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MSCCL++ Python API."""

import atexit
from dataclasses import dataclass
from functools import cached_property, wraps
import inspect
import json
import os
from pathlib import Path
from typing import Any
import warnings

from blake3 import blake3

from mscclpp.language.program import CollectiveProgram
from mscclpp.language.utils import AlgoSpec
from functools import wraps
from mscclpp._version import __version__, __commit_id__

if os.environ.get("MSCCLPP_HOME", None) is None:
    os.environ["MSCCLPP_HOME"] = os.path.abspath(os.path.dirname(__file__))


# Parse the version
version = {
    "version": __version__,
    "git_commit": __commit_id__,
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
    Executor,
    ExecutionPlan,
    ExecutionPlanConstraint,
    PacketType,
    RawGpuBuffer,
    env,
    is_nvls_supported,
    npkit,
    ExecutionPlanHandle as _ExecutionPlanHandle,
    ExecutionPlanRegistry as _ExecutionPlanRegistry,
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


class ExecutionPlanHandle:

    def __init__(self, handle: _ExecutionPlanHandle):
        self._handle = handle

    @cached_property
    def id(self) -> int:
        return self._handle.id

    @cached_property
    def tags(self) -> set:
        return frozenset(self._handle.tags)

    @cached_property
    def plan(self) -> ExecutionPlan:
        return self._handle.plan

    @cached_property
    def constraints(self) -> ExecutionPlanConstraint:
        return self._handle.constraints


@dataclass(frozen=True)
class ExecutionRequest:
    collective: str
    world_size: int
    n_ranks_per_node: int
    send_buffer: int
    recv_buffer: int
    message_size: int
    hints: dict


class ExecutionPlanRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionPlanRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._registry = _ExecutionPlanRegistry.get_instance()
            self._id_map = {}
            self._collective_map = {}
            self._selector = None
            self._initialized = True

    def register_plan(self, plan: ExecutionPlanHandle):
        self._id_map[plan.id] = plan
        if plan.plan.collective not in self._collective_map:
            self._collective_map[plan.plan.collective] = []
        self._collective_map[plan.plan.collective].append(plan)
        return self._instance._registry.register_plan(plan._handle)

    def set_selector(self, selector):
        self._selector = selector
        self._instance._registry.set_selector(selector)

    def set_default_selector(self, selector):
        self._selector = selector
        self._instance._registry.set_default_selector(selector)

    def get(self, id: str) -> ExecutionPlanHandle:
        return self._id_map.get(id, None)

    def select(
        self,
        collective: str,
        world_size: int,
        n_ranks_per_node: int,
        send_buffer: int,
        recv_buffer: int,
        message_size: int,
        hints: dict = {},
    ) -> ExecutionPlanHandle:
        if self._selector is None or collective not in self._collective_map:
            return None
        req = ExecutionRequest(
            collective=collective,
            world_size=world_size,
            n_ranks_per_node=n_ranks_per_node,
            send_buffer=send_buffer,
            recv_buffer=recv_buffer,
            message_size=message_size,
            hints=hints,
        )
        return self._selector(self._collective_map[collective], req)

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._registry.clear()
            cls._instance._id_map = {}
            cls._instance._collective_map = {}
            cls._instance._selector = None
            cls._instance = None


atexit.register(ExecutionPlanRegistry.reset_instance)

_execution_plan_registry = ExecutionPlanRegistry()


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def compile(
    algo,
    algo_spec: AlgoSpec,
    rank: int,
    **kwargs,
) -> ExecutionPlanHandle:
    """Compile a MSCCL++ program from a high-level algorithm description.
    Args:
        algo: The high-level algorithm description (e.g., a function or class).
        algo_spec (AlgoSpec): Algorithm specification containing collective type,
            world size, ranks per node, instances, protocol, and other configuration.
        rank (int): The rank of the current process.
        **kwargs: Additional keyword arguments passed to the algorithm function.
    Returns:
        ExecutionPlanHandle: The compiled execution plan handle.
    Raises:
        ValueError: If the 'algo' argument is not callable.
    """
    if not callable(algo):
        raise ValueError("The 'algo' argument must be a callable (e.g., a function or class).")
    prog: CollectiveProgram = algo(
        algo_spec,
        **kwargs,
    )
    source = inspect.getsource(algo)

    source_hash = blake3(source.encode("utf-8")).hexdigest()
    plan_id = blake3(
        _stable_json_bytes(
            {
                "version": __version__,
                "algo_name": algo_spec.name,
                "collective": algo_spec.collective.name,
                "tags": sorted(algo_spec.tags.items()),
                "source_hash": source_hash,
                "envs": {
                    "nranks_per_node": algo_spec.nranks_per_node,
                    "world_size": algo_spec.world_size,
                    "instances": algo_spec.instances,
                    "protocol": algo_spec.protocol,
                },
            }
        )
    ).hexdigest()
    plan_handle = _execution_plan_registry.get(plan_id)
    if plan_handle is not None:
        return plan_handle

    plan_dir = os.environ.get("MSCCLPP_EXECUTION_PLAN_DIR", Path.home() / ".cache/mscclpp")
    os.makedirs(plan_dir, exist_ok=True)
    filename = f"{plan_id}.json"
    plan_path = os.path.join(plan_dir, filename)
    tmp_path = plan_path + f".tmp.{os.getpid()}"
    if not os.path.exists(plan_path):
        try:
            # TODO (binyli): Each rank could generate its own execution plan separately. Doesn't need to generate whole plan.
            with open(tmp_path, "w") as f:
                prog.post_process_operations()
                f.write(prog.to_json(indent=None, separators=(",", ":"), ensure_ascii=False))
                f.flush()
                os.fsync(f.fileno())
            if not os.path.exists(plan_path):
                os.rename(tmp_path, plan_path)
            else:
                os.remove(tmp_path)
        except Exception:
            Path(plan_path).unlink(missing_ok=True)
    execution_plan = ExecutionPlan(plan_path, rank)
    handle = _ExecutionPlanHandle.create(
        id=plan_id,
        world_size=algo_spec.world_size,
        nranks_per_node=algo_spec.nranks_per_node,
        plan=execution_plan,
        tags=algo_spec.tags,
    )
    return ExecutionPlanHandle(handle)
