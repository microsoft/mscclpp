# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import importlib.util
import inspect
import logging
import json
import os
import subprocess
from typing import Any, Callable
from pathlib import Path

import pybind11
from shutil import which
import sys
import sysconfig

from blake3 import blake3

from mscclpp._version import __version__
from mscclpp._executor import ExecutionPlanHandle, ExecutionPlanRegistry
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.utils import AlgoSpec

from ._mscclpp import (
    ExecutionPlan,
    ExecutionPlanHandle as _ExecutionPlanHandle,
)


logging.basicConfig(level=logging.INFO)

_execution_plan_registry = ExecutionPlanRegistry()

def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

class DslCompiler:
    def __init__(self):
        pass

    def __call__(self, algo: Callable[..., CollectiveProgram], algo_spec: AlgoSpec, rank: int, **kwds):
        return self.compile(algo, algo_spec, rank, **kwds)

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

class NativeCodeCompiler:
    def __init__(self):
        self._hip_available = which("hipcc") is not None
        self._cuda_available = which("nvcc") is not None
        if not (self._hip_available or self._cuda_available):
            raise RuntimeError("No suitable compiler found (nvcc or hipcc).")
        self._compiler = "nvcc" if self._cuda_available else "hipcc"
        self._default_options = ["-std=c++17", "-O3", "--shared"]
        self._device_arch = "sm_90" # need to detect dynamically in the future
        python_include = sysconfig.get_path('include')
        pybind11_include = pybind11.get_include()
        self._default_options += [
            f"-I{python_include}",
            f"-I{pybind11_include}"
        ]

        python_lib = f"-lpython{sys.version_info.major}.{sys.version_info.minor}"
        self._default_options.append(python_lib)

        if self._cuda_available:
            # Format: -gencode=arch=compute_90,code=sm_90
            compute_arch = self._device_arch.replace("sm_", "compute_")
            arch_flag = f"-gencode=arch={compute_arch},code={self._device_arch}"
            self._default_options.append(arch_flag)
            self._default_options += ["--compiler-options", "-fPIC"]
        else:
            # Format for HIP: --offload-arch=gfx90a
            arch_flag = f"--offload-arch={self._device_arch}"
            self._default_options.append(arch_flag)
            self._default_options += ["-fPIC"]

        self._lib_home = os.path.abspath(os.path.dirname(__file__))
        self._default_options = self._default_options + ["-I" + os.path.join(self._lib_home, "include"),
                                                         "-L" + os.path.join(self._lib_home, "lib"),
                                                         "-lmscclpp",]


    def is_hip(self) -> bool:
        return self._hip_available

    def is_cuda(self) -> bool:
        return self._cuda_available
    
    def get_arch(self):
        return self._device_arch

    def __call__(self, name: str, file: str, **kwds):
        return self.compile(name, file, **kwds)

    def compile(self, name: str, file: str):
        """Compile a MSCCL++ native program from a CUDA/HIP source file.
        Args:
            name (str): The name of the python module to be created.
            file (str): The path to the CUDA/HIP source file.
        Returns:
            str: The path to the compiled shared library.
        Raises:
            FileNotFoundError: If the specified source file does not exist.
            RuntimeError: If compilation fails.
        """
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The specified source file does not exist: {file}")
        
        output_file = os.path.splitext(file)[0] + ".so"
        compile_command = [self._compiler] + self._default_options + ["-o", output_file, file]
        
        try:
            subprocess.run(compile_command, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"Compiler '{self._compiler}' not found. Make sure it's installed and in PATH.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Compilation failed with return code {e.returncode}.\n"
                f"Command: {' '.join(compile_command)}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            ) from e
        spec = importlib.util.spec_from_file_location(name, output_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module '{name}' from '{output_file}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        logging.info(f"Successfully compiled and loaded module '{name}' from '{output_file}'")
        return module
