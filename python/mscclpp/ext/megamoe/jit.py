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


@dataclass(frozen=True)
class KernelConfig:
    """Routed-kernel specialization; M256/K128, local shared, and precision stay fixed.

    ``load_stages`` controls raw weights, scales, and activations together.
    ``transform_stages`` controls the converted BF16 weights in TMEM.
    """

    tile_n: int = 32
    load_stages: int = 8
    transform_stages: int = 7

    def __post_init__(self):
        for name, allowed in (
            ("tile_n", (32, 64, 128)),
            ("load_stages", (4, 6, 8)),
            ("transform_stages", (2, 3, 4, 5, 6, 7)),
        ):
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if 2 * self.tile_n + 64 * self.transform_stages > 512:
            raise ValueError("kernel configuration exceeds the 512-column TMEM budget")


@dataclass(frozen=True)
class CompiledKernel:
    """Prepared native module; retained by a context, never rebuilt by forward."""

    config: KernelConfig
    path: str
    key: str
    cache_hit: bool

    def __post_init__(self):
        if not isinstance(self.config, KernelConfig):
            raise TypeError("config must be KernelConfig")
        if not isinstance(self.path, str) or type(self.cache_hit) is not bool:
            raise TypeError("path must be a string and cache_hit must be bool")
        if self.key == "builtin":
            if self.path or self.config != KernelConfig():
                raise ValueError("builtin kernel must use the default configuration and an empty path")
        elif not isinstance(self.key, str) or not re.fullmatch(r"[0-9a-f]{64}", self.key):
            raise ValueError("kernel key must be 'builtin' or a SHA256 hexadecimal digest")
        elif not isinstance(self.path, str) or not Path(self.path).is_absolute():
            raise ValueError("compiled kernel path must be absolute")


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
        if (root / "megamoe.cu").is_file():
            return root
    raise FileNotFoundError("MegaMoE JIT sources are missing; reinstall a build with MSCCLPP_BUILD_EXT_MEGAMOE=ON")


def _library(name):
    path = _package_root() / "lib" / f"lib{name}.so.0"
    if not path.is_file():
        raise FileNotFoundError(f"Required native library not found: {path}")
    return path.resolve()


def _check_device(device=None):
    import torch

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("MegaMoE JIT requires NVIDIA CUDA")
    device = torch.cuda.current_device() if device is None else device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare MegaMoE kernels outside CUDA Graph capture")
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise RuntimeError("MegaMoE JIT requires an SM100 GPU")
    return torch.cuda.get_device_properties(device)


def runtime_fingerprint(device=None):
    """Return an exact cache/profile environment key without invoking a compiler."""
    import torch

    properties = _check_device(device)
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
        "arch": "sm_100a",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_driver_version": version.value,
        "native_library_sha256": _file_hash(_library("mscclpp_megamoe")),
        "core_library_sha256": _file_hash(_library("mscclpp")),
        "jit_source_sha256": _tree_hash(_source_root(), {".cu", ".cuh", ".hpp"}),
        "jit_builder_sha256": _file_hash(Path(__file__)),
        "frontend_code_sha256": _file_hash(Path(__file__).with_name("benchmark_shared.py")),
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
        manifest.get("version") != 1
        or manifest.get("key") != key
        or _digest(build) != key
        or build.get("environment") != environment
        or not isinstance(build.get("config"), dict)
        or not isinstance(manifest.get("module_sha256"), str)
    ):
        raise ValueError(f"Stale or incompatible MegaMoE JIT manifest: {folder}")
    config = KernelConfig(**build["config"])
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

    Set ``MSCCLPP_MEGAMOE_NVCC`` (nvcc >=13.3),
    ``MSCCLPP_MEGAMOE_CUTLASS_ROOT``, and optionally ``CUDA_HOME``.
    ``MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS`` supplies path-separated matching runtime
    include directories when compiler and runtime packages are installed separately.
    """
    if not isinstance(config, KernelConfig):
        raise TypeError("config must be KernelConfig")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    if config == KernelConfig():
        return CompiledKernel(config, "", "builtin", True)
    environment = runtime_fingerprint()
    source_root = _source_root()
    include_root = _package_root() / "include"
    if not (include_root / "mscclpp/version.hpp").is_file():
        raise FileNotFoundError(f"Installed MSCCL++ development headers are required: {include_root}")
    cutlass_value = cutlass_root or os.environ.get("MSCCLPP_MEGAMOE_CUTLASS_ROOT")
    if cutlass_value is None:
        raise ValueError("Set MSCCLPP_MEGAMOE_CUTLASS_ROOT to the compatible CUTLASS checkout")
    cutlass = Path(cutlass_value).expanduser().resolve() / "include"
    if not (cutlass / "cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp").is_file():
        raise FileNotFoundError(f"SM100 mixed-input CUTLASS headers not found: {cutlass}")
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).expanduser().resolve()
    compiler = _tool(nvcc or os.environ.get("MSCCLPP_MEGAMOE_NVCC", str(cuda_root / "bin/nvcc")), "nvcc")
    cxx = _tool(os.environ.get("CXX", "c++"), "C++ compiler")
    compiler_version = _version(compiler)
    match = re.search(r"release (\d+)\.(\d+)", compiler_version)
    if match is None or tuple(map(int, match.groups())) < (13, 3):
        raise RuntimeError("MegaMoE JIT requires CUDA nvcc/ptxas >=13.3")
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
    build = {
        "version": 1,
        "config": asdict(config),
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
            obj, module = stage / "kernel.o", stage / "kernel.so"
            command = [
                compiler,
                "-ccbin",
                cxx,
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "--generate-code=arch=compute_100a,code=sm_100a",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcompiler=-fPIC,-fvisibility=hidden",
                "-Xptxas=-v",
                "-DMSCCLPP_USE_CUDA",
                "-DMSCCLPP_MEGAMOE_JIT_MODULE=1",
                f'-DMSCCLPP_MEGAMOE_JIT_ID="{key}"',
                f"-DMSCCLPP_MEGAMOE_TILE_N={config.tile_n}",
                f"-DMSCCLPP_MEGAMOE_LOAD_STAGES={config.load_stages}",
                f"-DMSCCLPP_MEGAMOE_TRANSFORM_STAGES={config.transform_stages}",
            ]
            for include in (*extra_includes, include_root, source_root / "include", cutlass):
                command.extend(["-I", str(include)])
            command.extend(["-c", str(source_root / "megamoe.cu"), "-o", str(obj)])
            library_root = _package_root() / "lib"
            link = [
                cxx,
                "-shared",
                str(obj),
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
                log.write(json.dumps({"compile": command, "link": link}) + "\n")
                _run(command, log, timeout)
                _run(link, log, timeout)
            obj.unlink()
            manifest = {"version": 1, "key": key, "build": build, "module_sha256": _file_hash(module)}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(stage, destination)
    return CompiledKernel(config, str(destination / "kernel.so"), key, False)
