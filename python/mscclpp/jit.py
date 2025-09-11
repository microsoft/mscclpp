# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import json
from pathlib import Path
from typing import Any
from blake3 import blake3
import inspect
import os

from python.mscclpp.language.program import CollectiveProgram

from ._mscclpp import __version__ as mscclpp_version


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

def compile(
    algo,
    name: str,
    collective: str,
    nranks_per_node: int,
    world_size: int,
    instances: int,
    protocol: str,
    num_threads_per_block: int = 1024,
    min_msg_size: int = 0,
    max_msg_size: int = 2**64 - 1,
    tags: dict = {},
    **kwargs,
):
    """Compile a MSCCL++ program from a high-level algorithm description.
    Args:
        algo: The high-level algorithm description (e.g., a function or class).
        name (str): The name of the program.
        collective (str): The collective operation type (e.g., "allreduce").
        nranks_per_node (int): Number of ranks per node.
        world_size (int): Total number of ranks in the program.
        instances (int): Number of instances to replicate.
        protocol (str): Communication protocol ("Simple" or "LL").
        num_threads_per_block (int): Number of threads per GPU thread block.
        min_msg_size (int): Minimum message size for this program.
        max_msg_size (int): Maximum message size for this program.
        tags (dict): Additional tags or metadata for the program.
        **kwargs: Additional keyword arguments for future extensions.
    Returns:
        The compiled program object.
    Raises:
        NotImplementedError: If the compilation logic is not implemented.
    """
    prog: CollectiveProgram = algo(
        name,
        collective,
        nranks_per_node,
        world_size,
        instances,
        protocol,
        num_threads_per_block,
        min_msg_size,
        max_msg_size,
        **kwargs,
    )
    source = inspect.getsource(algo)

    source_hash = blake3(source.encode("utf-8")).hexdigest()
    plan_id = blake3(
        _stable_json_bytes(
            {
                "version": mscclpp_version,
                "algo_name": name,
                "collective": collective,
                "tags": tags,
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
    plan_dir = os.environ.get("MSCCLPP_EXECUTION_PLAN_DIR", Path.home() / ".cache/mscclpp")
    os.makedirs(plan_dir, exist_ok=True)
    filename = f"{plan_id}".json
    full_path = os.path.join(plan_dir, filename)
    if not os.path.exists(full_path):
        with open(f"/{plan_dir}/{filename}", "w") as f:
            json.dump(prog.to_json(), f, separators=(",", ":"), ensure_ascii=False)
