# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Cache and compile-contract tests; no compiler or GPU needed for host cases."""

import json
import os
from pathlib import Path

import pytest

from mscclpp.ext.megamoe import jit


@pytest.mark.parametrize(
    "fields",
    [
        {"tile_n": 0},
        {"tile_n": True},
        {"tile_n": 96},
        {"load_stages": 0},
        {"load_stages": 3},
        {"load_stages": 4.0},
        {"transform_stages": 1},
        {"transform_stages": 8},
        {"tile_n": 128, "transform_stages": 7},
        {"tile_n": 64, "transform_stages": 7},
    ],
)
def test_kernel_config_rejects_invalid(fields):
    with pytest.raises(ValueError):
        jit.KernelConfig(**fields)


@pytest.mark.parametrize("values", [(32, 8, 7), (32, 6, 7), (64, 6, 6), (128, 4, 4)])
def test_initial_specializations_fit_tmem(values):
    config = jit.KernelConfig(*values)
    assert 2 * config.tile_n + 64 * config.transform_stages <= 512


def test_builtin_requires_neither_compiler_nor_gpu(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("builtin must not inspect the JIT toolchain")

    monkeypatch.setattr(jit, "runtime_fingerprint", fail)
    monkeypatch.setattr(jit, "_tool", fail)
    compiled = jit.compile_kernel(jit.KernelConfig())
    assert compiled.key == "builtin" and compiled.path == ""
    assert jit.load_cached_kernel("builtin") == compiled


@pytest.fixture
def compiler_fixture(tmp_path, monkeypatch):
    package, source, cutlass, cuda = [tmp_path / name for name in ("package", "source", "cutlass", "cuda")]
    for path in (
        package / "include/mscclpp/version.hpp",
        package / "lib/libmscclpp.so",
        source / "megamoe.cu",
        source / "megamoe_launch.cu",
        source / "megamoe_jit.cu",
        source / "include/megamoe_kernel.hpp",
        cutlass / "include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp",
        cuda / "lib64/libcudart.so",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("MSCCLPP_MEGAMOE_CUTLASS_ROOT", str(cutlass))
    monkeypatch.setenv("MSCCLPP_MEGAMOE_CUDA_INCLUDE_DIRS", "")
    monkeypatch.setattr(jit, "_package_root", lambda: package)
    monkeypatch.setattr(jit, "_source_root", lambda: source)
    monkeypatch.setattr(jit, "runtime_fingerprint", lambda: {"native": "fixture", "driver": 13000})
    monkeypatch.setattr(jit, "_tool", lambda value, name: name)
    monkeypatch.setattr(jit, "_version", lambda tool: "release 13.3, V13.3.73")
    commands = []

    def run(command, log, timeout):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled_fixture")

    monkeypatch.setattr(jit, "_run", run)
    return tmp_path / "cache", commands, cutlass


def test_compile_is_cached_and_uses_native_sm100a(compiler_fixture):
    cache, commands, _ = compiler_fixture
    config = jit.KernelConfig(64, 6, 6)
    first = jit.compile_kernel(config, cache_dir=cache)
    assert not first.cache_hit and len(commands) == 4
    assert [Path(command[command.index("-c") + 1]).name for command in commands[:-1]] == [
        "megamoe.cu",
        "megamoe_launch.cu",
        "megamoe_jit.cu",
    ]
    for command in commands[:-1]:
        assert "--generate-code=arch=compute_100a,code=sm_100a" in command
        assert "-DMSCCLPP_MEGAMOE_TILE_N=64" in command
        assert "-DMSCCLPP_MEGAMOE_LOAD_STAGES=6" in command
        assert "-DMSCCLPP_MEGAMOE_TRANSFORM_STAGES=6" in command
        assert "-Xcompiler=-fPIC,-fvisibility=hidden" in command
        assert command[command.index("-o") + 1] in commands[-1]
    assert not list(Path(first.path).parent.glob("*.o"))
    again = jit.compile_kernel(config, cache_dir=cache)
    assert again.cache_hit and again.key == first.key and len(commands) == 4
    assert not list(cache.glob(".*"))


@pytest.mark.parametrize("name", ["megamoe.cu", "megamoe_launch.cu", "megamoe_jit.cu"])
def test_missing_translation_unit_is_reported(compiler_fixture, name):
    _, commands, _ = compiler_fixture
    (jit._source_root() / name).unlink()
    with pytest.raises(FileNotFoundError, match="Required MegaMoE JIT source"):
        jit.compile_kernel(jit.KernelConfig(64, 6, 6))
    assert not commands


def test_concurrent_requests_publish_only_one_build(compiler_fixture):
    from concurrent.futures import ThreadPoolExecutor

    cache, commands, _ = compiler_fixture
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jit.compile_kernel, jit.KernelConfig(32, 6, 7), cache_dir=cache) for _ in range(2)]
        kernels = [future.result() for future in futures]
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
    compiled = jit.compile_kernel(jit.KernelConfig(32, 6, 7), cache_dir=cache)

    def missing(*args, **kwargs):
        raise FileNotFoundError("toolchain removed")

    monkeypatch.setattr(jit, "_tool", missing)
    monkeypatch.setattr(jit, "_source_root", missing)
    loaded = jit.load_cached_kernel(compiled.key, cache_dir=cache)
    assert loaded.key == compiled.key and loaded.cache_hit


def test_changed_headers_or_config_get_new_cache_key(compiler_fixture):
    cache, commands, cutlass = compiler_fixture
    config = jit.KernelConfig(32, 6, 7)
    first = jit.compile_kernel(config, cache_dir=cache)
    header = cutlass / "include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp"
    header.write_text("changed header")
    second = jit.compile_kernel(config, cache_dir=cache)
    third = jit.compile_kernel(jit.KernelConfig(128, 4, 4), cache_dir=cache)
    assert len({first.key, second.key, third.key}) == 3
    assert len(commands) == 12


def test_corrupt_module_is_not_silently_reused_or_rebuilt(compiler_fixture):
    cache, commands, _ = compiler_fixture
    config = jit.KernelConfig(32, 6, 7)
    module = jit.compile_kernel(config, cache_dir=cache)
    Path(module.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        jit.load_cached_kernel(module.key, cache_dir=cache)
    with pytest.raises(ValueError, match="checksum"):
        jit.compile_kernel(config, cache_dir=cache)
    assert len(commands) == 4


@pytest.mark.parametrize("invalid", [[], {}, {"version": 1, "build": {}}])
def test_invalid_manifest_is_an_explicit_error(compiler_fixture, invalid):
    cache, _, _ = compiler_fixture
    module = jit.compile_kernel(jit.KernelConfig(32, 6, 7), cache_dir=cache)
    Path(module.path).with_name("manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="manifest"):
        jit.load_cached_kernel(module.key, cache_dir=cache)


def test_stale_profile_and_missing_cache_are_errors(compiler_fixture, monkeypatch):
    cache, _, _ = compiler_fixture
    module = jit.compile_kernel(jit.KernelConfig(32, 6, 7), cache_dir=cache)
    monkeypatch.setattr(jit, "runtime_fingerprint", lambda: {"native": "changed"})
    with pytest.raises(ValueError, match="Stale"):
        jit.load_cached_kernel(module.key, cache_dir=cache)
    with pytest.raises(FileNotFoundError):
        jit.load_cached_kernel("f" * 64, cache_dir=cache)
    with pytest.raises(ValueError, match="key"):
        jit.load_cached_kernel("../outside", cache_dir=cache)


def test_failed_compilation_is_not_published(compiler_fixture, monkeypatch):
    cache, _, _ = compiler_fixture

    def fail(command, log, timeout):
        raise RuntimeError("compile error")

    monkeypatch.setattr(jit, "_run", fail)
    with pytest.raises(RuntimeError, match="compile error"):
        jit.compile_kernel(jit.KernelConfig(32, 6, 7), cache_dir=cache)
    assert not list(cache.glob("*/manifest.json"))
    assert not list(cache.glob(".*"))


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_compile_timeout_must_be_bounded(timeout):
    with pytest.raises(ValueError, match="timeout"):
        jit.compile_kernel(jit.KernelConfig(), timeout=timeout)


@pytest.mark.parametrize(
    "body,message",
    [
        ("print('compiler failed', flush=True); raise SystemExit(7)", r"failed \(7\)"),
        ("import time; print('compiler started', flush=True); time.sleep(10)", "timed out"),
    ],
)
def test_compiler_errors_keep_diagnostic_log(tmp_path, body, message):
    import sys

    path = tmp_path / "build.log"
    with path.open("w") as log:
        with pytest.raises(RuntimeError, match=message):
            jit._run([sys.executable, "-c", body], log, 0.5)
    assert "compiler" in path.read_text()


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_JIT") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_JIT=1 to compile and exercise the CUDA specializations",
)
@pytest.mark.parametrize("values", [(32, 6, 7), (64, 6, 6), (128, 4, 4)])
@pytest.mark.parametrize("e5m2", [False, True])
def test_jit_variants_match_builtin_and_replay_graphs(values, e5m2):
    import gc
    import torch
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig
    from mscclpp.ext.megamoe.benchmark import _weights

    torch.cuda.set_device(0)
    module = jit.compile_kernel(jit.KernelConfig(*values))
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    comm = Communicator(bootstrap)
    config = MegaMoEConfig(0, 1, 129, 128, 128, 2, 2, sm_margin=120, weight_e5m2=e5m2)
    torch.manual_seed(783)
    weights = _weights(config, torch.device("cuda", 0))
    baseline = MegaMoE(config, comm, *weights)
    variant = MegaMoE(config, comm, *weights, kernel=module, tag=17924)
    assert variant.kernel_id == module.key
    assert variant.kernel_config == module.config
    assert variant.shared_bytes > 0
    assert variant._native.kernel_tile_n == values[0]
    for tokens in (0, 1, 33, 65, 129):
        inputs = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([0, 1], dtype=torch.int32, device="cuda").expand(tokens, 2).contiguous()
        scores = torch.rand(tokens, 2, device="cuda")
        expected = baseline(inputs, ids, scores)
        output = variant(inputs, ids, scores)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            variant(inputs, ids, scores, output=output, stream=stream)
        inputs.neg_()
        expected = baseline(inputs, ids, scores)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        del graph
    view = variant.input_view(1)
    del variant, baseline
    gc.collect()
    view.fill_(2)
    torch.cuda.synchronize()
    assert torch.all(view == 2)
    del view
    gc.collect()


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_JIT") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_JIT=1 to exercise CUDA module lifetime",
)
def test_jit_context_recreation_after_failed_preflight():
    import gc
    import torch
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig
    from mscclpp.ext.megamoe.benchmark import _weights

    torch.cuda.set_device(0)
    module = jit.compile_kernel(jit.KernelConfig(64, 6, 6))
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    config = MegaMoEConfig(0, 1, 2, 128, 128, 2, 1, sm_margin=120)
    weights = _weights(config, torch.device("cuda", 0))
    invalid = jit.CompiledKernel(module.config, module.path, "f" * 64, True)
    inputs = torch.ones(2, 128, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    scores = torch.ones(2, 1, device="cuda")
    for _ in range(3):
        with pytest.raises((ValueError, RuntimeError), match="preflight"):
            MegaMoE(config, communicator, *weights, kernel=invalid)
        context = MegaMoE(config, communicator, *weights, kernel=module)
        output = context(inputs, ids, scores)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        del context
        gc.collect()
