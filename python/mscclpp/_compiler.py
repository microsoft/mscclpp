# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import inspect
import json
import os
from typing import Any, Callable
from pathlib import Path

from blake3 import blake3

from mscclpp._version import __version__
from mscclpp._executor import ExecutionPlanHandle, ExecutionPlanRegistry
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.utils import AlgoSpec

from ._mscclpp import (
    ExecutionPlan,
    ExecutionPlanHandle as _ExecutionPlanHandle,
)


_execution_plan_registry = ExecutionPlanRegistry()

def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

def compile(
    algo: Callable[..., CollectiveProgram],
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


def compile_native(file: str):
    """Compile a MSCCL++ native program from a CUDA source file.
    Args:
        file (str): The path to the CUDA source file.
        **kwargs: Additional keyword arguments for future extensions.
    Returns:
        Any: The compiled native program object.
    Raises:
        FileNotFoundError: If the specified CUDA source file does not exist.
    """
    compiler = None
    if "nvidia":
        compiler = "nvcc"
    elif "amd":
        compiler = "hipcc"
    else:
        raise ValueError("Unsupported platform for native compilation.")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"The specified CUDA source file does not exist: {file}")
    compile_options = ["-std=c++17", "-O3"]
    # run the compiler
    output_file = file + ".out"
    compile_command = f"{compiler} {' '.join(compile_options)} -o {output_file} {file}"
    ret = os.system(compile_command)
    if ret != 0:
        raise RuntimeError(f"Compilation failed with return code {ret}. Command: {compile_command}")