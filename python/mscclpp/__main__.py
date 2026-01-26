# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import argparse
from pathlib import Path

from mscclpp import default_algos as def_algo
from mscclpp.language.collectives import *
from mscclpp.language.utils import AlgoSpec

default_algo_configs = [
    {
        "filename": "allreduce_2nodes_1K_64K.json",
        "function": def_algo.allreduce_2nodes,
        "spec": AlgoSpec(
            name="allreduce_2nodes_1K_64K",
            collective=AllReduce(16, 1, True),
            nranks_per_node=8,
            world_size=16,
            in_place=True,
            instances=1,
            protocol="LL",
            auto_sync=False,
            num_threads_per_block=1024,
            reuse_resources=True,
            use_double_scratch_buffer=True,
            min_message_size=1 << 10,
            max_message_size=64 << 10,
            tags={"default": 1},
        ),
        "additional_kwargs": {"thread_block_group_size": 1},
    },
    {
        "filename": "allreduce_2nodes_128K_2M.json",
        "function": def_algo.allreduce_2nodes,
        "spec": AlgoSpec(
            name="allreduce_2nodes_128K_2M",
            collective=AllReduce(16, 1, True),
            nranks_per_node=8,
            world_size=16,
            in_place=True,
            instances=1,
            protocol="LL",
            auto_sync=False,
            num_threads_per_block=1024,
            reuse_resources=True,
            use_double_scratch_buffer=True,
            min_message_size=128 << 10,
            max_message_size=2 << 20,
            tags={"default": 1},
        ),
        "additional_kwargs": {"thread_block_group_size": 4},
    },
]


def create_default_plans():
    plan_dir = os.environ.get("MSCCLPP_CACHE_DIR", Path.home() / ".cache/mscclpp/default")
    plan_path = Path(plan_dir)
    if plan_path.exists():
        shutil.rmtree(plan_path)
    plan_path.mkdir(parents=True)

    for config in default_algo_configs:
        filename = config["filename"]
        func = config["function"]
        spec = config["spec"]
        additional_kwargs = config.get("additional_kwargs", {})
        plan_path = os.path.join(plan_dir, filename)

        try:
            if additional_kwargs:
                prog = func(spec, **additional_kwargs)
            else:
                prog = func(spec)

            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(prog.to_json())
                f.flush()

        except Exception as e:
            print(f"Error creating plan for {spec.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="flag to install default plans")
    args = parser.parse_args()

    if args.install:
        create_default_plans()


if __name__ == "__main__":
    main()
