import os
import shutil
from pathlib import Path

from .language import default_algos as def_algo
from .language.internal.types import AlgoSpec

default_algo_configs = [
    {
        "filename": "allreduce_2nodes.json",
        "function": def_algo.allreduce_2nodes,
        "spec": AlgoSpec(
            name="allreduce_2nodes",
            collective="allreduce",
            nranks_per_node=8,
            world_size=16,
            instances=1,
            protocol="LL",
            num_threads_per_block=1024,
            min_message_size=0,
            max_message_size=2 << 20,
            tags={"default": 1},
        ),
        "additional_args": [4]  
    },
    {
        "filename": "allreduce_naivy.json",
        "function": def_algo.allreduce_naivy,
        "spec": AlgoSpec(
            name="allreduce_naivy",
            collective="allreduce",
            nranks_per_node=8,
            world_size=8,
            instances=1,
            protocol="LL",
            num_threads_per_block=1024,
            min_message_size=0,
            max_message_size=2 << 20,
            tags={"default": 1},
        )
    }
]

def create_default_plans():
    plan_dir = os.environ.get("MSCCLPP_EXECUTION_PLAN_DIR", Path.home() / ".cache/mscclpp_default")
    plan_path = Path(plan_dir)
    if plan_path.exists():
        shutil.rmtree(plan_path)
    plan_path.mkdir(parents=True)

    for config in default_algo_configs:
        filename = config["filename"]
        func = config["function"]
        spec = config["spec"]
        additional_args = config.get("additional_args", [])
        plan_path = os.path.join(plan_dir, filename)
        
        try:
            if additional_args:
                prog = func(spec, *additional_args)
            else:
                prog = func(spec)

            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(prog.to_json())
                f.flush()
                    
        except Exception as e:
            print(f"Error creating plan for {spec.name}: {e}")
            continue

def main():
    create_default_plans()

if __name__ == "__main__":
    main()