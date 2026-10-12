# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""W4A8 cache and compile-contract tests; host cases need no GPU or compiler."""

import json
from pathlib import Path

import pytest

from mscclpp.ext.megamoe import jit


@pytest.mark.parametrize(
    "fields",
    [
        {"tile_n": 96},
        {"tile_k": 64},
        {"load_stages": 1},
        {"load_stages": True},
        {"num_warps": 8},
        {"transfer_registers": 48},
        {"load_warps": 3},
        {"split_pipelines": 1},
        {"load_warps": 1, "split_pipelines": True},
        {"epilogue_tokens": 64},
        {"tile_n": 32, "epilogue_tokens": 16, "epilogue_warps": 8},
        {"epilogue_warps": 8, "epilogue_registers": 224, "transfer_registers": 128},
        {"epilogue_registers": 201},
        {"dispatch_chunk": 768},
        {"dispatch_warps": 1},
        {"dispatch_stages": 3},
        {"fixed_token_count": 1},
        {"expert_major_tasks": 1},
        {"cache_weights": 0},
        {"adaptive_cache_weights": 1},
        {"adaptive_cache_weights": "true"},
        {"adaptive_cache_weights": True, "cache_weights": True},
        {"cluster_size": 0},
        {"cluster_size": 3},
        {"cluster_size": True},
    ],
)
def test_w4a8_kernel_config_rejects_invalid(fields):
    with pytest.raises(ValueError):
        jit.W4A8KernelConfig(**fields)


def test_default_and_tuned_policies_validate():
    assert jit.W4A8KernelConfig().tile_n == 64
    assert jit.W4A8KernelConfig().cluster_size == 2
    tuned = jit.W4A8KernelConfig(
        tile_n=128,
        load_stages=7,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
    )
    assert tuned.epilogue_warps == 8 and tuned.dispatch_chunk == 4096


def test_builtin_requires_neither_compiler_nor_gpu(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("builtin must not inspect the JIT toolchain")

    monkeypatch.setattr(jit, "runtime_fingerprint", fail)
    monkeypatch.setattr(jit, "_tool", fail)
    compiled = jit.compile_kernel(jit.W4A8KernelConfig())
    assert compiled.key == "builtin" and compiled.path == ""
    assert jit.load_cached_kernel("builtin") == compiled


@pytest.fixture
def compiler_fixture(tmp_path, monkeypatch):
    package, source, cutlass, cuda = [tmp_path / name for name in ("package", "source", "cutlass", "cuda")]
    for path in (
        package / "include/mscclpp/version.hpp",
        package / "lib/libmscclpp.so.0",
        source / "megamoe_w4a8.cu",
        source / "megamoe_launch.cu",
        source / "megamoe_jit.cu",
        source / "include/megamoe_kernel.hpp",
        cutlass / "include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp",
        cuda / "lib64/libcudart.so",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("MSCCLPP_MEGAMOE_CUTLASS_ROOT", str(cutlass))
    monkeypatch.setenv("MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS", "")
    monkeypatch.setattr(jit, "_package_root", lambda: package)
    monkeypatch.setattr(jit, "_source_root", lambda: source)
    monkeypatch.setattr(jit, "runtime_fingerprint", lambda: {"native": "fixture", "driver": 13000, "arch": "sm_100f"})
    monkeypatch.setattr(jit, "_tool", lambda value, name: name)
    monkeypatch.setattr(jit, "_version", lambda tool: "release 13.0, V13.0.88")
    commands = []

    def run(command, log, timeout):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled_fixture")

    monkeypatch.setattr(jit, "_run", run)
    return tmp_path / "cache", commands, cutlass


def _tuned(**overrides):
    values = {"load_stages": 8}
    values.update(overrides)
    return jit.W4A8KernelConfig(**values)


def test_compile_is_cached_and_uses_every_policy_flag(compiler_fixture):
    cache, commands, _ = compiler_fixture
    config = _tuned(
        tile_n=128,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
        expert_major_tasks=True,
        cache_weights=True,
    )
    compiled = jit.compile_kernel(config, cache_dir=cache)
    assert compiled.key != "builtin" and not compiled.cache_hit and len(commands) == 4
    assert [Path(command[command.index("-c") + 1]).name for command in commands[:-1]] == [
        "megamoe_w4a8.cu",
        "megamoe_launch.cu",
        "megamoe_jit.cu",
    ]
    expected = {
        "-DMSCCLPP_MEGAMOE_W4_TILE_N=128",
        "-DMSCCLPP_MEGAMOE_W4_TILE_K=128",
        "-DMSCCLPP_MEGAMOE_W4_LOAD_STAGES=8",
        "-DMSCCLPP_MEGAMOE_W4_NUM_WARPS=16",
        "-DMSCCLPP_MEGAMOE_W4_TRANSFER_REGISTERS=64",
        "-DMSCCLPP_MEGAMOE_W4_LOAD_WARPS=2",
        "-DMSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES=0",
        "-DMSCCLPP_MEGAMOE_W4_EPILOGUE_TOKENS=32",
        "-DMSCCLPP_MEGAMOE_W4_EPILOGUE_WARPS=8",
        "-DMSCCLPP_MEGAMOE_W4_EPILOGUE_REGISTERS=208",
        "-DMSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK=4096",
        "-DMSCCLPP_MEGAMOE_W4_DISPATCH_WARPS=4",
        "-DMSCCLPP_MEGAMOE_W4_DISPATCH_STAGES=1",
        "-DMSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT=0",
        "-DMSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS=1",
        "-DMSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS=1",
        "-DMSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS=0",
        "-DMSCCLPP_MEGAMOE_W4_CLUSTER_SIZE=2",
    }
    for command in commands[:-1]:
        assert expected <= set(command)
        assert "--generate-code=arch=compute_100f,code=sm_100f" in command
        assert command[command.index("-o") + 1] in commands[-1]
    manifest = json.loads(Path(compiled.path).with_name("manifest.json").read_text())
    assert manifest["version"] == 3
    assert manifest["build"]["kernel_kind"] == "w4a8"
    assert manifest["build"]["config"] == {name: getattr(config, name) for name in config.__dataclass_fields__}
    loaded = jit.load_cached_kernel(compiled.key, cache_dir=cache)
    assert loaded.config == config and loaded.cache_hit
    assert jit.compile_kernel(config, cache_dir=cache).cache_hit
    assert len(commands) == 4


def test_fixed_and_cluster_options_have_distinct_cache_keys(compiler_fixture):
    cache, commands, _ = compiler_fixture
    configs = [
        _tuned(cluster_size=1),
        _tuned(cluster_size=2),
        _tuned(cluster_size=2, fixed_token_count=True),
    ]
    kernels = [jit.compile_kernel(config, cache_dir=cache) for config in configs]
    assert len({kernel.key for kernel in kernels}) == 3
    assert len(commands) == 12
    assert jit.load_cached_kernel(kernels[-1].key, cache_dir=cache).config.fixed_token_count is True


def test_cache_policies_have_distinct_keys_and_preserve_task_order(compiler_fixture):
    cache, commands, _ = compiler_fixture
    policies = [
        _tuned(),
        _tuned(cache_weights=True),
        _tuned(adaptive_cache_weights=True),
        _tuned(adaptive_cache_weights=True, expert_major_tasks=True),
    ]
    keys = set()
    for policy in policies:
        before = len(commands)
        kernel = jit.compile_kernel(policy, cache_dir=cache)
        keys.add(kernel.key)
        assert len(commands) == before + 4
        for command in commands[before : before + 3]:
            assert f"-DMSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS={int(policy.adaptive_cache_weights)}" in command
            assert f"-DMSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS={int(policy.cache_weights)}" in command
            assert f"-DMSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS={int(policy.expert_major_tasks)}" in command
        assert jit.load_cached_kernel(kernel.key, cache_dir=cache).config == policy
        assert jit.compile_kernel(policy, cache_dir=cache).cache_hit
        assert len(commands) == before + 4
    assert len(keys) == len(policies)
    assert jit.W4A8KernelConfig().adaptive_cache_weights is False


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_sm100_family_capabilities_use_one_jit_target(capability):
    assert jit._ARCHITECTURES[capability] == "sm_100f"


@pytest.mark.parametrize("name", ["megamoe_w4a8.cu", "megamoe_launch.cu", "megamoe_jit.cu"])
def test_missing_translation_unit_is_reported(compiler_fixture, name):
    cache, commands, _ = compiler_fixture
    (jit._source_root() / name).unlink()
    with pytest.raises(FileNotFoundError, match="Required MegaMoE JIT source"):
        jit.compile_kernel(_tuned(), cache_dir=cache)
    assert not commands


def test_installed_source_bundle_is_discovered(tmp_path, monkeypatch):
    source = tmp_path / "share/mscclpp/megamoe"
    source.mkdir(parents=True)
    (source / "megamoe_w4a8.cu").write_text("fixture")
    monkeypatch.setattr(jit, "_package_root", lambda: tmp_path)
    assert jit._source_root() == source


def test_concurrent_requests_publish_only_one_build(compiler_fixture):
    from concurrent.futures import ThreadPoolExecutor

    cache, commands, _ = compiler_fixture
    config = _tuned()
    with ThreadPoolExecutor(max_workers=2) as pool:
        kernels = [
            future.result() for future in [pool.submit(jit.compile_kernel, config, cache_dir=cache) for _ in range(2)]
        ]
    assert kernels[0].key == kernels[1].key
    assert sorted(kernel.cache_hit for kernel in kernels) == [False, True]
    assert len(commands) == 4


def test_cache_lock_timeout_is_reported(tmp_path):
    with jit._cache_lock(tmp_path / "lock", 1):
        with pytest.raises(TimeoutError, match="cache lock"):
            with jit._cache_lock(tmp_path / "lock", 0.01):
                pytest.fail("a second lock must not be acquired")


def test_cache_only_loading_does_not_invoke_compiler(compiler_fixture, monkeypatch):
    cache, _, _ = compiler_fixture
    compiled = jit.compile_kernel(_tuned(), cache_dir=cache)

    def missing(*args, **kwargs):
        raise FileNotFoundError("toolchain removed")

    monkeypatch.setattr(jit, "_tool", missing)
    monkeypatch.setattr(jit, "_source_root", missing)
    assert jit.load_cached_kernel(compiled.key, cache_dir=cache).cache_hit


def test_changed_headers_or_config_get_new_cache_key(compiler_fixture):
    cache, commands, cutlass = compiler_fixture
    first = jit.compile_kernel(_tuned(), cache_dir=cache)
    header = cutlass / "include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp"
    header.write_text("changed header")
    second = jit.compile_kernel(_tuned(), cache_dir=cache)
    third = jit.compile_kernel(_tuned(tile_n=128), cache_dir=cache)
    assert len({first.key, second.key, third.key}) == 3
    assert len(commands) == 12


def test_corrupt_or_stale_cache_is_rejected(compiler_fixture, monkeypatch):
    cache, commands, _ = compiler_fixture
    module = jit.compile_kernel(_tuned(), cache_dir=cache)
    Path(module.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        jit.load_cached_kernel(module.key, cache_dir=cache)
    assert len(commands) == 4
    monkeypatch.setattr(jit, "runtime_fingerprint", lambda: {"native": "changed"})
    with pytest.raises(ValueError, match="Stale"):
        jit.load_cached_kernel(module.key, cache_dir=cache)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_compile_timeout_must_be_bounded(timeout):
    with pytest.raises(ValueError, match="timeout"):
        jit.compile_kernel(jit.W4A8KernelConfig(), timeout=timeout)


def test_failed_compilation_is_not_published(compiler_fixture, monkeypatch):
    cache, _, _ = compiler_fixture

    def fail(command, log, timeout):
        raise RuntimeError("compile error")

    monkeypatch.setattr(jit, "_run", fail)
    with pytest.raises(RuntimeError, match="compile error"):
        jit.compile_kernel(_tuned(), cache_dir=cache)
    assert not list(cache.glob("*/manifest.json"))
