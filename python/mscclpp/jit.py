# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from blake3 import blake3
import inspect
import os

from mscclpp.language.program import CollectiveProgram
from mscclpp.plan import PlanHandle, Registry

from ._mscclpp import ExecutionPlan, version 


_version = version()
def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    collective: str
    nranks_per_node: int
    world_size: int
    instances: int
    protocol: str
    num_threads_per_block: int
    min_message_size: int
    max_message_size: int
    tags: set

def compile(
    algo,
    name: str,
    collective: str,
    rank: int,
    nranks_per_node: int,
    world_size: int,
    instances: int,
    protocol: str,
    num_threads_per_block: int = 1024,
    min_message_size: int = 0,
    max_message_size: int = 2**64 - 1,
    tags: set = {},
    **kwargs,
) -> PlanHandle:
    """Compile a MSCCL++ program from a high-level algorithm description.
    Args:
        algo: The high-level algorithm description (e.g., a function or class).
        name (str): The name of the program.
        collective (str): The collective operation type (e.g., "allreduce").
        rank (int): The rank of the current process.
        nranks_per_node (int): Number of ranks per node.
        world_size (int): Total number of ranks in the program.
        instances (int): Number of instances to replicate.
        protocol (str): Communication protocol ("Simple" or "LL").
        num_threads_per_block (int): Number of threads per GPU thread block.
        min_msg_size (int): Minimum message size for this program.
        max_msg_size (int): Maximum message size for this program.
        tags (set): Additional tags or metadata for the program.
        **kwargs: Additional keyword arguments for future extensions.
    Returns:
        The compiled program object.
    Raises:
        NotImplementedError: If the compilation logic is not implemented.
    """
    if not callable(algo):
        raise ValueError("The 'algo' argument must be a callable (e.g., a function or class).")
    prog: CollectiveProgram = algo(
        AlgoSpec(
            name=name,
            collective=collective,
            nranks_per_node=nranks_per_node,
            world_size=world_size,
            instances=instances,
            protocol=protocol,
            num_threads_per_block=num_threads_per_block,
            min_message_size=min_message_size,
            max_message_size=max_message_size,
            tags=tags,
        ),
        **kwargs,
    )
    source = inspect.getsource(algo)

    source_hash = blake3(source.encode("utf-8")).hexdigest()
    plan_id = blake3(
        _stable_json_bytes(
            {
                "version": _version,
                "algo_name": name,
                "collective": collective,
                "tags": sorted(tags),
                "source_hash": source_hash,
                "envs": {
                    "nranks_per_node": nranks_per_node,
                    "world_size": world_size,
                    "instances": instances,
                    "protocol": protocol,
                },
            }
        )
    ).hexdigest()
    plan_handel = Registry.get(plan_id)
    if plan_handel is not None:
        return plan_handel

    plan_dir = os.environ.get("MSCCLPP_EXECUTION_PLAN_DIR", Path.home() / ".cache/mscclpp")
    os.makedirs(plan_dir, exist_ok=True)
    filename = f"{plan_id}.json"
    plan_path = os.path.join(plan_dir, filename)
    if not os.path.exists(plan_path):
        try:
            with open(f"/{plan_dir}/{filename}", "w") as f:
                f.write(prog.to_json(indent=None, separators=(",", ":"), ensure_ascii=False))
        except Exception:
            Path(plan_path).unlink(missing_ok=True)
    execution_plan = ExecutionPlan(plan_path, rank)
    return PlanHandle(
        id=plan_id,
        name=name,
        collective=collective,
        tags=tags,
        constraints={
            "min_message_size": min_message_size,
            "max_message_size": max_message_size,
            "nranks_per_node": nranks_per_node,
            "world_size": world_size,
        },
        executionPlan=execution_plan,
    )
