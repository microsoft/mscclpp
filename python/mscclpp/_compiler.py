# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import importlib.util
import inspect
import logging
import json
import os
import subprocess
import fcntl
from typing import Any, Callable
from pathlib import Path

import pybind11
import sys
import sysconfig

from blake3 import blake3
import cupy as cp

from mscclpp._version import __version__
from mscclpp._algorithm import Algorithm
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.utils import AlgoSpec
from mscclpp.utils import get_device_arch

from ._mscclpp import (
    ExecutionPlan,
)


logging.basicConfig(level=logging.INFO)


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

    def __call__(self, algo: Callable[..., CollectiveProgram], algo_spec: AlgoSpec, rank: int, **kwds) -> Algorithm:
        return self.compile(algo, algo_spec, rank, **kwds)

    def compile(
        self,
        algo: Callable[..., CollectiveProgram],
        algo_spec: AlgoSpec,
        rank: int,
        **kwargs,
    ) -> Algorithm:
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
        return Algorithm(
            id=plan_id,
            execution_plan=execution_plan,
            constraint=Algorithm.Constraint(
                world_size=algo_spec.world_size, n_ranks_per_node=algo_spec.nranks_per_node
            ),
            tags=algo_spec.tags,
        )


class NativeCodeCompiler:
    def __init__(self):
        self._is_hip = cp.cuda.runtime.is_hip
        self._device_arch = get_device_arch()
        self._compiler = self._get_compiler()
        self._default_options = ["-std=c++17", "-O3", "--shared"]
        python_include = sysconfig.get_path("include")
        pybind11_include = pybind11.get_include()
        self._default_options += [f"-I{python_include}", f"-I{pybind11_include}"]

        python_lib = f"-lpython{sys.version_info.major}.{sys.version_info.minor}"
        self._default_options.append(python_lib)
        self._lib_home = os.path.abspath(os.path.dirname(__file__))

        if not self._is_hip:
            # Format: -gencode=arch=compute_90,code=sm_90
            compute_arch = self._device_arch.replace("sm_", "compute_")
            arch_flag = f"-gencode=arch={compute_arch},code={self._device_arch}"
            self._default_options.append(arch_flag)
            self._default_options += ["--compiler-options", "-fPIC"]
            self._default_options += ["--linker-options", f"-rpath,{self._lib_home}/lib"]
        else:
            # Format for HIP: --offload-arch=gfx90a
            arch_flag = f"--offload-arch={self._device_arch}"
            self._default_options.append(arch_flag)
            self._default_options += ["-fPIC"]

        self._default_options = self._default_options + [
            "-I" + os.path.join(self._lib_home, "include"),
            "-L" + os.path.join(self._lib_home, "lib"),
            "-lmscclpp",
        ]
        cache_root = os.environ.get("MSCCLPP_NATIVE_CACHE_DIR", Path.home() / ".cache/mscclpp/native")
        self._cache_dir = Path(cache_root)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_compiler(self) -> str:
        if self._is_hip:
            rocm_home = os.environ.get("ROCM_HOME")
            return os.path.join(rocm_home, "bin/hipcc") if rocm_home else "hipcc"
        else:
            cuda_home = os.environ.get("CUDA_HOME")
            return os.path.join(cuda_home, "bin/nvcc") if cuda_home else "nvcc"

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

        with open(file, "rb") as source_file:
            source_bytes = source_file.read()
        source_hash = blake3(source_bytes).hexdigest()
        cache_key = blake3(
            _stable_json_bytes(
                {
                    "version": __version__,
                    "source_hash": source_hash,
                    "compiler": self._compiler,
                    "options": self._default_options,
                    "arch": self._device_arch,
                }
            )
        ).hexdigest()
        output_file = self._cache_dir / f"{name}-{cache_key}.so"
        lock_file = output_file.with_suffix(output_file.suffix + ".lock")

        with open(lock_file, "w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if not output_file.exists():
                tmp_file = output_file.with_suffix(output_file.suffix + f".tmp.{os.getpid()}")
                compile_command = [self._compiler] + self._default_options + ["-o", str(tmp_file), file]
                try:
                    subprocess.run(compile_command, check=True, capture_output=True, text=True)
                    os.replace(tmp_file, output_file)
                except FileNotFoundError as e:
                    Path(tmp_file).unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Compiler '{self._compiler}' not found. Make sure it's installed and in PATH."
                    ) from e
                except subprocess.CalledProcessError as e:
                    Path(tmp_file).unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Compilation failed with return code {e.returncode}.\n"
                        f"Command: {' '.join(compile_command)}\n"
                        f"Stdout: {e.stdout}\n"
                        f"Stderr: {e.stderr}"
                    ) from e
        module_name = name
        existing_module = sys.modules.get(module_name)
        if existing_module and getattr(existing_module, "__mscclpp_cache_key__", None) == cache_key:
            return existing_module

        spec = importlib.util.spec_from_file_location(module_name, output_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module '{name}' from '{output_file}'")
        module = importlib.util.module_from_spec(spec)
        module.__mscclpp_cache_key__ = cache_key
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        logging.debug(f"Successfully compiled and loaded module '{name}' from '{output_file}'")
        return module
