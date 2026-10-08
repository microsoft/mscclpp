# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline compilation and content-addressed caching of native MegaMoE variants."""

from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

_ARCHITECTURES = {(10, 0): "sm_100f", (10, 3): "sm_100f", (10, 7): "sm_100f"}
_CACHE_FORMAT_VERSION = 2
_BUILD_FORMAT_VERSION = 2


@dataclass(frozen=True)
class KernelConfig:
    """Routed-kernel specialization; local shared and precision stay fixed.

    ``load_stages`` controls raw weights, scales, and activations together.
    ``transform_stages`` controls the converted BF16 weights in TMEM.
    """

    tile_n: int = 32
    load_stages: int = 8
    transform_stages: int = 7
    tile_k: int = 128
    tile_m: int = 256

    def __post_init__(self):
        for name, allowed in (
            ("tile_m", (128, 256)),
            ("tile_n", (32, 64, 128)),
            ("load_stages", (4, 6, 8)),
            ("transform_stages", (2, 3, 4, 5, 6, 7)),
            ("tile_k", (32, 64, 128)),
        ):
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if (self.tile_n, self.load_stages, self.transform_stages) not in (
            (32, 8, 7),
            (32, 6, 7),
            (64, 6, 6),
            (128, 4, 4),
        ):
            raise ValueError("unsupported tile_n/load_stages/transform_stages combination")
        if self.tile_m == 128 and self.tile_k == 32:
            raise ValueError("tile_m=128 requires tile_k=64 or 128")
        if 2 * self.tile_n + self.tile_k // 2 * self.transform_stages > 512:
            raise ValueError("kernel configuration exceeds the 512-column TMEM budget")


@dataclass(frozen=True)
class W4A8KernelConfig:
    """Compile-time W4A8 policy for one external native JIT module."""

    tile_n: int = 64
    tile_k: int = 128
    load_stages: int = 9
    num_warps: int = 16
    transfer_registers: int = 128
    load_warps: int = 2
    split_pipelines: bool = False
    epilogue_tokens: int = 32
    epilogue_warps: int = 4
    epilogue_registers: int = 224
    dispatch_chunk: int = 3072
    dispatch_warps: int = 4
    dispatch_stages: int = 1

    def __post_init__(self):
        allowed = (
            ("tile_n", (32, 64, 128)),
            ("tile_k", (128, 256, 512)),
            ("num_warps", (12, 16)),
            ("transfer_registers", (32, 64, 96, 128)),
            ("load_warps", (1, 2)),
            ("epilogue_tokens", (16, 32)),
            ("epilogue_warps", (4, 8)),
            ("dispatch_chunk", (512, 1024, 2048, 3072, 4096)),
            ("dispatch_warps", (2, 3, 4)),
            ("dispatch_stages", (1, 2)),
        )
        for name, values in allowed:
            value = getattr(self, name)
            if type(value) is not int or value not in values:
                raise ValueError(f"{name} must be one of {values}")
        if type(self.load_stages) is not int or not 2 <= self.load_stages <= 10:
            raise ValueError("load_stages must be an integer in [2, 10]")
        if type(self.epilogue_registers) is not int or not 128 <= self.epilogue_registers <= 224:
            raise ValueError("epilogue_registers must be an integer in [128, 224]")
        if self.epilogue_registers % 8:
            raise ValueError("epilogue_registers must be divisible by 8")
        if type(self.split_pipelines) is not bool:
            raise ValueError("split_pipelines must be bool")
        if self.split_pipelines and self.load_warps != 2:
            raise ValueError("split_pipelines requires load_warps=2")
        if self.tile_n % self.epilogue_tokens:
            raise ValueError("epilogue_tokens must divide tile_n")
        if self.dispatch_chunk % self.tile_k:
            raise ValueError("dispatch_chunk must be divisible by tile_k")
        if self.epilogue_warps + 4 + self.dispatch_warps > self.num_warps:
            raise ValueError("warp roles exceed num_warps")
        threads = self.num_warps * 32
        epilogue_threads = self.epilogue_warps * 32
        low_register_threads = threads - epilogue_threads - 128
        registers = (
            epilogue_threads * self.epilogue_registers + 128 * self.transfer_registers + low_register_threads * 32
        )
        if low_register_threads < 0 or registers > threads * 128:
            raise ValueError("kernel configuration exceeds the CTA register budget")


@dataclass(frozen=True)
class CompiledKernel:
    """Prepared native module; retained by a context, never rebuilt by forward."""

    config: KernelConfig | W4A8KernelConfig
    path: str
    key: str
    cache_hit: bool

    def __post_init__(self):
        if not isinstance(self.config, (KernelConfig, W4A8KernelConfig)):
            raise TypeError("config must be KernelConfig or W4A8KernelConfig")
        if not isinstance(self.path, str) or type(self.cache_hit) is not bool:
            raise TypeError("path must be a string and cache_hit must be bool")
        if self.key == "builtin":
            if self.path or not isinstance(self.config, KernelConfig) or self.config != KernelConfig():
                raise ValueError("builtin kernel must use the default configuration and an empty path")
        elif not isinstance(self.key, str) or not re.fullmatch(r"[0-9a-f]{64}", self.key):
            raise ValueError("kernel key must be 'builtin' or a SHA256 hexadecimal digest")
        elif not isinstance(self.path, str) or not Path(self.path).is_absolute():
            raise ValueError("compiled kernel path must be absolute")


def _config_kind(config):
    if isinstance(config, KernelConfig):
        return "w8a16"
    if isinstance(config, W4A8KernelConfig):
        return "w4a8"
    raise TypeError("config must be KernelConfig or W4A8KernelConfig")


def _config_from_build(build):
    kind = build.get("kernel_kind")
    values = build.get("config")
    if not isinstance(values, dict):
        raise ValueError("MegaMoE JIT manifest has an invalid configuration")
    if kind == "w8a16":
        return KernelConfig(**values)
    if kind == "w4a8":
        return W4A8KernelConfig(**values)
    raise ValueError("MegaMoE JIT manifest has an unsupported kernel kind")


def _compile_definitions(config):
    if isinstance(config, KernelConfig):
        return [
            "-DMSCCLPP_MEGAMOE_JIT_W4A8=0",
            f"-DMSCCLPP_MEGAMOE_TILE_N={config.tile_n}",
            f"-DMSCCLPP_MEGAMOE_TILE_K={config.tile_k}",
            f"-DMSCCLPP_MEGAMOE_TILE_M={config.tile_m}",
            f"-DMSCCLPP_MEGAMOE_LOAD_STAGES={config.load_stages}",
            f"-DMSCCLPP_MEGAMOE_TRANSFORM_STAGES={config.transform_stages}",
        ]
    if isinstance(config, W4A8KernelConfig):
        values = asdict(config)
        names = {
            "tile_n": "TILE_N",
            "tile_k": "TILE_K",
            "load_stages": "LOAD_STAGES",
            "num_warps": "NUM_WARPS",
            "transfer_registers": "TRANSFER_REGISTERS",
            "load_warps": "LOAD_WARPS",
            "split_pipelines": "SPLIT_PIPELINES",
            "epilogue_tokens": "EPILOGUE_TOKENS",
            "epilogue_warps": "EPILOGUE_WARPS",
            "epilogue_registers": "EPILOGUE_REGISTERS",
            "dispatch_chunk": "DISPATCH_CHUNK",
            "dispatch_warps": "DISPATCH_WARPS",
            "dispatch_stages": "DISPATCH_STAGES",
        }
        return [
            "-DMSCCLPP_MEGAMOE_JIT_W4A8=1",
            *[
                f"-DMSCCLPP_MEGAMOE_W4_{names[name]}={int(value) if isinstance(value, bool) else value}"
                for name, value in values.items()
            ],
        ]
    raise TypeError("config must be KernelConfig or W4A8KernelConfig")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root, suffixes):
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    if not files:
        raise FileNotFoundError(f"No required source/header files found under {root}")
    return _digest({str(path.relative_to(root)): _file_hash(path) for path in files})


def _package_root():
    return Path(__file__).resolve().parents[2]


def _source_root():
    installed = _package_root() / "share/mscclpp/megamoe"
    checkout = Path(__file__).resolve().parents[4] / "src/ext/megamoe"
    for root in (installed, checkout):
        if (root / "megamoe_w8a16.cu").is_file():
            return root
    raise FileNotFoundError("MegaMoE JIT sources are missing; reinstall a build with MSCCLPP_BUILD_EXT_MEGAMOE=ON")


def _library(name):
    installed = _package_root() / "lib" / f"lib{name}.so.0"
    if installed.is_file():
        return installed.resolve()
    prefix = f"lib{name}.so"
    loaded = sorted(
        {
            Path(line.split()[-1]).resolve()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line and Path(line.split()[-1]).name.startswith(prefix)
        }
    )
    if len(loaded) == 1:
        return loaded[0]
    raise FileNotFoundError(f"Required native library not found: {installed}")


def _check_device(device=None):
    import torch

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("MegaMoE JIT requires NVIDIA CUDA")
    device = torch.cuda.current_device() if device is None else device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare MegaMoE kernels outside CUDA Graph capture")
        capability = torch.cuda.get_device_capability(device)
        if capability not in _ARCHITECTURES:
            raise RuntimeError("MegaMoE JIT requires an SM100-family GPU")
    return torch.cuda.get_device_properties(device), _ARCHITECTURES[capability]


def runtime_fingerprint(device=None):
    """Return an exact cache/profile environment key without invoking a compiler."""
    import torch

    properties, architecture = _check_device(device)
    driver = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    status = driver.cuDriverGetVersion(ctypes.byref(version))
    if status:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA driver status {status}")
    return {
        "gpu_name": properties.name,
        "sm_count": properties.multi_processor_count,
        "arch": architecture,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_driver_version": version.value,
        "native_library_sha256": _file_hash(_library("mscclpp_megamoe")),
        "core_library_sha256": _file_hash(_library("mscclpp")),
        "jit_source_sha256": _tree_hash(_source_root(), {".cu", ".cuh", ".hpp"}),
        "jit_builder_sha256": _file_hash(Path(__file__)),
        "cache_tag": os.environ.get("MSCCLPP_MEGAMOE_CACHE_TAG", ""),
        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def _cache_root(cache_dir=None):
    value = cache_dir or os.environ.get("MSCCLPP_MEGAMOE_CACHE_DIR")
    if value is None:
        value = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mscclpp/megamoe"
    return Path(value).expanduser().resolve()


@contextmanager
def _cache_lock(path, timeout):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for JIT cache lock: {path}") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_cached(key, cache_dir, environment):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("invalid MegaMoE JIT cache key")
    folder = _cache_root(cache_dir) / key
    with (folder / "manifest.json").open() as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("build"), dict):
        raise ValueError(f"Invalid MegaMoE JIT manifest: {folder}")
    build = manifest["build"]
    if (
        manifest.get("version") != _CACHE_FORMAT_VERSION
        or manifest.get("key") != key
        or _digest(build) != key
        or build.get("environment") != environment
        or build.get("version") != _BUILD_FORMAT_VERSION
        or build.get("kernel_kind") not in ("w8a16", "w4a8")
        or not isinstance(build.get("config"), dict)
        or not isinstance(build.get("definitions"), list)
        or not isinstance(manifest.get("module_sha256"), str)
    ):
        raise ValueError(f"Stale or incompatible MegaMoE JIT manifest: {folder}")
    config = _config_from_build(build)
    if build["definitions"] != _compile_definitions(config):
        raise ValueError(f"Stale or incompatible MegaMoE JIT manifest: {folder}")
    module = folder / "kernel.so"
    if _file_hash(module) != manifest["module_sha256"]:
        raise ValueError(f"MegaMoE JIT module checksum mismatch: {module}")
    return CompiledKernel(config, str(module), key, True)


def load_cached_kernel(key, *, cache_dir=None):
    """Load a prepared module without nvcc or CUTLASS; reject stale/corrupt entries."""
    if key == "builtin":
        return CompiledKernel(KernelConfig(), "", "builtin", True)
    return _load_cached(key, cache_dir, runtime_fingerprint())


def _tool(value, name):
    found = shutil.which(str(value))
    if found is None:
        raise FileNotFoundError(f"{name} not found: {value}")
    return str(Path(found).resolve())


def _version(tool):
    return subprocess.run(
        [tool, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _run(command, log, timeout):
    # The compiler launches ptxas/linker children; a timeout must stop the group.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
            log.write(output)
            log.flush()
            raise RuntimeError(f"MegaMoE JIT command timed out after {timeout}s; see {log.name}") from error
        log.write(output)
        log.flush()
        if process.returncode:
            raise RuntimeError(f"MegaMoE JIT command failed ({process.returncode}); see {log.name}\n{output[-4000:]}")


def compile_kernel(config, *, cache_dir=None, nvcc=None, cutlass_root=None, timeout=600):
    """Compile one specialization once per host/cache, outside collective construction.

    Set ``MSCCLPP_MEGAMOE_NVCC`` (nvcc >=13.0),
    ``MSCCLPP_MEGAMOE_CUTLASS_ROOT``, and optionally ``CUDA_HOME``.
    ``MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS`` supplies path-separated matching runtime
    include directories when compiler and runtime packages are installed separately.
    """
    if not isinstance(config, (KernelConfig, W4A8KernelConfig)):
        raise TypeError("config must be KernelConfig or W4A8KernelConfig")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    if isinstance(config, KernelConfig) and config == KernelConfig():
        return CompiledKernel(config, "", "builtin", True)
    environment = runtime_fingerprint()
    source_root = _source_root()
    sources = [source_root / name for name in ("megamoe_w8a16.cu", "megamoe_launch.cu", "megamoe_jit.cu")]
    if isinstance(config, W4A8KernelConfig):
        sources.insert(1, source_root / "megamoe_w4a8.cu")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Required MegaMoE JIT source not found: {source}")
    installed_include = _package_root() / "include"
    checkout_include = source_root.parents[2] / "include"
    build_include = _library("mscclpp").parent.parent / "include"
    include_root = next(
        (path for path in (installed_include, build_include) if (path / "mscclpp/version.hpp").is_file()),
        installed_include,
    )
    if not (include_root / "mscclpp/version.hpp").is_file():
        raise FileNotFoundError(f"MSCCL++ development headers are required: {include_root}")
    cutlass_value = cutlass_root or os.environ.get("MSCCLPP_MEGAMOE_CUTLASS_ROOT")
    if cutlass_value is None:
        raise ValueError("Set MSCCLPP_MEGAMOE_CUTLASS_ROOT to the compatible CUTLASS checkout")
    cutlass = Path(cutlass_value).expanduser().resolve() / "include"
    if not (cutlass / "cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp").is_file():
        raise FileNotFoundError(f"SM100-family mixed-input CUTLASS headers not found: {cutlass}")
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).expanduser().resolve()
    compiler = _tool(nvcc or os.environ.get("MSCCLPP_MEGAMOE_NVCC", str(cuda_root / "bin/nvcc")), "nvcc")
    cxx = _tool(os.environ.get("CXX", "c++"), "C++ compiler")
    compiler_version = _version(compiler)
    match = re.search(r"release (\d+)\.(\d+)", compiler_version)
    if match is None or tuple(map(int, match.groups())) < (13, 0):
        raise RuntimeError("MegaMoE JIT requires CUDA nvcc/ptxas >=13.0")
    cuda_lib = cuda_root / "lib64"
    if not (cuda_lib / "libcudart.so").is_file():
        raise FileNotFoundError(f"CUDA runtime development library not found: {cuda_lib / 'libcudart.so'}")
    extra_includes = [
        Path(value).expanduser().resolve()
        for value in os.environ.get("MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS", "").split(os.pathsep)
        if value
    ]
    ambient_includes = [
        Path(value).expanduser().resolve()
        for name in ("CPATH", "CPLUS_INCLUDE_PATH")
        for value in os.environ.get(name, "").split(os.pathsep)
        if value
    ]
    definitions = _compile_definitions(config)
    build = {
        "version": _BUILD_FORMAT_VERSION,
        "kernel_kind": _config_kind(config),
        "config": asdict(config),
        "definitions": definitions,
        "environment": environment,
        "compiler": compiler_version,
        "cxx": _version(cxx),
        "cutlass_headers_sha256": _tree_hash(cutlass, {".h", ".hpp", ".cuh", ".inl"}),
        "mscclpp_headers_sha256": _tree_hash(include_root, {".h", ".hpp", ".cuh", ".inl"}),
        "extra_cuda_headers_sha256": [_tree_hash(path, {".h", ".hpp", ".cuh", ".inl"}) for path in extra_includes],
        "ambient_headers_sha256": [_tree_hash(path, {".h", ".hpp", ".cuh", ".inl"}) for path in ambient_includes],
        "cuda_runtime_sha256": _file_hash(cuda_lib / "libcudart.so"),
        "compiler_environment": {
            name: os.environ.get(name, "")
            for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")
        },
    }
    key = _digest(build)
    root = _cache_root(cache_dir)
    destination = root / key
    with _cache_lock(root / f"{key}.lock", timeout):
        if destination.exists():
            return _load_cached(key, root, environment)
        with tempfile.TemporaryDirectory(prefix=f".{key}.", dir=root) as temporary:
            stage = Path(temporary)
            objects = [stage / f"{source.stem}.o" for source in sources]
            module = stage / "kernel.so"
            architecture = environment["arch"]
            compute_architecture = architecture.replace("sm_", "compute_", 1)
            command = [
                compiler,
                "-ccbin",
                cxx,
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                f"--generate-code=arch={compute_architecture},code={architecture}",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcompiler=-fPIC,-fvisibility=hidden",
                "-Xptxas=-v",
                "-DMSCCLPP_USE_CUDA",
                "-DMSCCLPP_MEGAMOE_JIT_MODULE=1",
                f'-DMSCCLPP_MEGAMOE_JIT_ID="{key}"',
                *definitions,
            ]
            for include in (*extra_includes, checkout_include, include_root, source_root / "include", cutlass):
                command.extend(["-I", str(include)])
            commands = [
                [*command, "-c", str(source), "-o", str(obj)] for source, obj in zip(sources, objects, strict=True)
            ]
            library_root = _library("mscclpp").parent
            link = [
                cxx,
                "-shared",
                *(str(obj) for obj in objects),
                "-o",
                str(module),
                f"-L{library_root}",
                "-lmscclpp",
                f"-L{cuda_lib}",
                "-lcudart",
                "-l:libcuda.so.1",
                f"-Wl,-rpath,{library_root}:{cuda_lib}",
            ]
            with (root / f"{key}.build.log").open("w") as log:
                log.write(json.dumps({"compile": commands, "link": link}) + "\n")
                for command in commands:
                    _run(command, log, timeout)
                _run(link, log, timeout)
            for obj in objects:
                obj.unlink()
            manifest = {
                "version": _CACHE_FORMAT_VERSION,
                "key": key,
                "build": build,
                "module_sha256": _file_hash(module),
            }
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(stage, destination)
    return CompiledKernel(config, str(destination / "kernel.so"), key, False)
