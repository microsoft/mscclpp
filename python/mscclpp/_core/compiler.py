# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import annotations
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
from .algorithm import Algorithm
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.utils import AlgoSpec
from mscclpp.utils import get_device_arch

from mscclpp._mscclpp import CppExecutionPlan

logging.basicConfig(level=logging.INFO)

__all__ = ["DslCompiler", "NativeCodeCompiler"]


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class DslCompiler:
    """Compiler for MSCCL++ DSL (Domain-Specific Language) algorithms.

    This compiler transforms high-level algorithm descriptions written in Python
    into execution plans that can be run on GPUs. The compiled plans are cached
    to disk for reuse.

    The cache location can be configured via the `MSCCLPP_CACHE_DIR`
    environment variable (defaults to `~/.cache/mscclpp`).

    Example:
        >>> compiler = DslCompiler()
        >>> algo = compiler.compile(my_allreduce_algo, algo_spec, rank=0)
    """

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
        """Compile a MSCCL++ DSL program from a high-level algorithm description.

        This method takes a Python function that defines a collective communication
        algorithm and compiles it into an executable Algorithm. The compilation
        result is cached based on a hash of the source code and algorithm specification.

        Args:
            algo: A callable (function or class) that takes an AlgoSpec and returns
                a CollectiveProgram. This defines the communication pattern.
            algo_spec: Algorithm specification containing:
                - collective: The collective operation type (e.g., allreduce, allgather)
                - world_size: Total number of ranks
                - nranks_per_node: Number of ranks per node
                - instances: Number of algorithm instances
                - protocol: Communication protocol to use
                - name: Human-readable algorithm name
                - tags: Dictionary of tags for algorithm selection
            rank: The rank of the current process (0 to world_size-1).
            **kwargs: Additional keyword arguments passed to the algorithm function.

        Returns:
            Algorithm: The compiled algorithm ready for execution.

        Raises:
            ValueError: If the 'algo' argument is not callable.

        Note:
            Compiled execution plans are cached to disk. The cache key is computed
            from the algorithm source code, specification, and MSCCL++ version.
            Subsequent calls with the same inputs will reuse the cached plan.

        Example:
            >>> def my_ring_allreduce(spec: AlgoSpec) -> CollectiveProgram:
            ...     # Define algorithm using MSCCL++ DSL
            ...     ...
            >>> compiler = DslCompiler()
            >>> spec = AlgoSpec(collective=Collective.allreduce, world_size=8, ...)
            >>> algo = compiler.compile(my_ring_allreduce, spec, rank=0)
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

        plan_dir = os.environ.get("MSCCLPP_CACHE_DIR", Path.home() / ".cache/mscclpp")
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
        execution_plan = CppExecutionPlan(plan_path, rank)
        return Algorithm(
            id=plan_id,
            execution_plan=execution_plan,
            constraint=Algorithm.Constraint(
                world_size=algo_spec.world_size, n_ranks_per_node=algo_spec.nranks_per_node
            ),
            tags=algo_spec.tags,
        )


class NativeCodeCompiler:
    """Compiler for native CUDA/HIP algorithm implementations.

    This compiler takes CUDA or HIP source files containing custom collective
    algorithm kernels and compiles them into loadable Python modules using
    pybind11 bindings.

    The compiler automatically detects whether to use NVCC (CUDA) or HIPCC (ROCm)
    based on the runtime environment. Compiled modules are cached to avoid
    recompilation.

    The cache location can be configured via the `MSCCLPP_CACHE_DIR`
    environment variable (defaults to `~/.cache/mscclpp`).

    Attributes:
        _is_hip: True if running on AMD/ROCm, False for NVIDIA/CUDA.
        _device_arch: The GPU architecture string (e.g., "sm_90" or "gfx90a").
        _compiler: Path to the compiler executable (nvcc or hipcc).

    Example:
        >>> compiler = NativeCodeCompiler()
        >>> module = compiler.compile("my_kernel", "path/to/kernel.cu")
        >>> algo = module.create_algorithm()
    """

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
            self._default_options += ["-D__HIP_PLATFORM_AMD__"]
            self._default_options += [f"-Wl,-rpath,{self._lib_home}/lib"]

        self._default_options = self._default_options + [
            "-I" + os.path.join(self._lib_home, "include"),
            "-L" + os.path.join(self._lib_home, "lib"),
            "-lmscclpp",
        ]
        cache_root = os.environ.get("MSCCLPP_CACHE_DIR", Path.home() / ".cache/mscclpp" / "native")
        self._cache_dir = Path(cache_root)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_compiler(self) -> str:
        """Get the path to the appropriate compiler.

        Returns:
            Path to nvcc (CUDA) or hipcc (ROCm) compiler.
        """
        if self._is_hip:
            rocm_home = os.environ.get("ROCM_HOME")
            return os.path.join(rocm_home, "bin/hipcc") if rocm_home else "hipcc"
        else:
            cuda_home = os.environ.get("CUDA_HOME")
            return os.path.join(cuda_home, "bin/nvcc") if cuda_home else "nvcc"

    def get_arch(self):
        """Get the target GPU architecture.

        Returns:
            str: The GPU architecture string (e.g., "sm_90" for NVIDIA or "gfx90a" for AMD).
        """
        return self._device_arch

    def __call__(self, name: str, file: str, **kwds):
        return self.compile(name, file, **kwds)

    def compile(self, name: str, file: str):
        """Compile a native CUDA/HIP source file into a Python module.

        This method compiles a CUDA (.cu) or HIP source file containing custom
        collective algorithm kernels into a dynamically loadable Python module.
        The module is expected to use pybind11 bindings to expose algorithm
        creation functions.

        Compilation results are cached based on a hash of the source code,
        compiler options, and GPU architecture. Subsequent calls with unchanged
        inputs will return the cached module.

        Args:
            name: The name of the Python module to create. This will be the
                module name used for importing (e.g., `import name`).
            file: Path to the CUDA/HIP source file to compile.

        Returns:
            module: The compiled and loaded Python module containing the
                algorithm implementation.

        Raises:
            FileNotFoundError: If the specified source file does not exist.
            RuntimeError: If compilation fails (compiler not found, syntax errors, etc.).
            ImportError: If the compiled module cannot be loaded.

        Note:
            - The source file should include pybind11 bindings to expose functions.
            - MSCCLPP headers are automatically included in the compilation.
            - The module is cached in `MSCCLPP_CACHE_DIR` (default: ~/.cache/mscclpp).
            - File locking is used to prevent race conditions during parallel compilation.

        Example:
            >>> compiler = NativeCodeCompiler()
            >>> # Compile a custom allreduce kernel
            >>> module = compiler.compile("my_allreduce", "kernels/allreduce.cu")
            >>> # Use the module to create an algorithm
            >>> algo = module.create_allreduce_algorithm(comm, buffer, size)
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
