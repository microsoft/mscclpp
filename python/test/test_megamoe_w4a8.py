# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only contracts for external W4A8 JIT policy profiles."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from mscclpp.ext.megamoe import w4a8

from mscclpp.ext.megamoe import (
    MegaMoEConfig,
    W4A8KernelConfig,
    W4A8KernelProfileMismatchError,
    load_w4a8_kernel_profiles,
    resolve_w4a8_kernel_config,
)


@pytest.fixture(autouse=True)
def gb300_device(monkeypatch):
    monkeypatch.setattr(w4a8, "_current_gpu_name", lambda: "NVIDIA GB300")


def _config(tokens=128, world_size=32, **overrides):
    fields = {
        "rank": 0,
        "world_size": world_size,
        "max_tokens": tokens,
        "hidden": 4096,
        "intermediate": 6656,
        "num_experts": 32 * world_size,
        "top_k": 8,
    }
    fields.update(overrides)
    return MegaMoEConfig(**fields)


@pytest.mark.parametrize(
    "tokens,tile_k,load_stages,dispatch_warps,dispatch_stages",
    [
        (32, 128, 8, 2, 2),
        (64, 128, 8, 2, 1),
        (128, 128, 8, 2, 2),
        (256, 128, 8, 2, 1),
        (512, 128, 8, 2, 1),
        (1024, 256, 4, 4, 1),
    ],
)
def test_packaged_gb300_ep32_dynamic_n128_policy(tokens, tile_k, load_stages, dispatch_warps, dispatch_stages):
    assert resolve_w4a8_kernel_config(_config(tokens)) == W4A8KernelConfig(
        tile_n=128,
        tile_k=tile_k,
        load_stages=load_stages,
        num_warps=16,
        transfer_registers=128,
        load_warps=2,
        split_pipelines=False,
        epilogue_tokens=32,
        epilogue_warps=4,
        epilogue_registers=224,
        dispatch_chunk=4096,
        dispatch_warps=dispatch_warps,
        dispatch_stages=dispatch_stages,
        fixed_token_count=True,
        expert_major_tasks=False,
        cache_weights=False,
        cluster_size=2,
        adaptive_cache_weights=False,
    )


def test_packaged_profiles_contain_only_gb300_ep32_token_buckets():
    profile = load_w4a8_kernel_profiles()
    assert set(profile) == {"version", "kind", "GB300"}
    assert profile["version"] == 1
    entries = profile["GB300"]["entries"]
    assert len(entries) == 6
    assert {entry["match"]["max_tokens"] for entry in entries} == {32, 64, 128, 256, 512, 1024}
    for entry in entries:
        assert entry["match"] == {
            "world_size": 32,
            "max_tokens": entry["match"]["max_tokens"],
            "hidden": 4096,
            "intermediate": 6656,
            "local_experts": 32,
            "top_k": 8,
            "gate_up_clamp": "negative",
        }
        assert entry["config"]["tile_n"] == 128


@pytest.mark.parametrize("device_name", ["GB300", "NVIDIA GB300"])
def test_explicit_device_name_selects_gpu_section(device_name):
    policy = resolve_w4a8_kernel_config(_config(), device_name=device_name)
    assert policy == resolve_w4a8_kernel_config(_config())


def test_offline_device_selection_does_not_query_cuda(monkeypatch):
    def unexpected_device_query():
        raise AssertionError("explicit device selection must not query CUDA")

    monkeypatch.setattr(w4a8, "_current_gpu_name", unexpected_device_query)
    assert resolve_w4a8_kernel_config(_config(), device_name="GB300").tile_n == 128


def test_detected_gpu_cannot_fall_back_to_gb300(monkeypatch):
    monkeypatch.setattr(w4a8, "_current_gpu_name", lambda: "NVIDIA GB200")
    with pytest.raises(W4A8KernelProfileMismatchError, match="GPU 'GB200'"):
        resolve_w4a8_kernel_config(_config())


def test_explicit_gpu_cannot_fall_back_to_gb300():
    with pytest.raises(W4A8KernelProfileMismatchError, match="GPU 'GB200'"):
        resolve_w4a8_kernel_config(_config(), device_name="GB200")


@pytest.mark.parametrize("device_name", ["version", "kind"])
def test_metadata_keys_are_not_gpu_profiles(device_name):
    with pytest.raises(W4A8KernelProfileMismatchError, match="no W4A8 kernel profile for GPU"):
        resolve_w4a8_kernel_config(_config(), device_name=device_name)


@pytest.mark.parametrize("device_name", ["", " ", 123])
def test_invalid_device_name_is_rejected(device_name):
    with pytest.raises(ValueError, match="GPU device name"):
        resolve_w4a8_kernel_config(_config(), device_name=device_name)


def test_same_workload_can_have_distinct_policies_per_gpu(tmp_path):
    profile = load_w4a8_kernel_profiles()
    profile["GB200"] = deepcopy(profile["GB300"])
    profile["GB200"]["entries"][0]["config"]["load_stages"] = 7
    path = tmp_path / "gpus.json"
    path.write_text(json.dumps(profile))
    assert resolve_w4a8_kernel_config(_config(32), profile_path=path, device_name="GB200").load_stages == 7
    assert resolve_w4a8_kernel_config(_config(32), profile_path=path, device_name="GB300").load_stages == 8


@pytest.mark.parametrize(
    "world_size,tokens,hidden,intermediate,local_experts",
    [(world_size, tokens, 8192, 4096, 16) for world_size in (4, 32) for tokens in (16, 32, 64, 128)]
    + [(4, tokens, 4096, 6656, 32) for tokens in (128, 190)],
)
def test_removed_packaged_shapes_require_external_profiles(world_size, tokens, hidden, intermediate, local_experts):
    config = _config(
        tokens,
        world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=local_experts * world_size,
    )
    with pytest.raises(W4A8KernelProfileMismatchError, match="no exact"):
        resolve_w4a8_kernel_config(config)


@pytest.mark.parametrize(
    "config",
    [
        _config(8),
        _config(128, 2),
        _config(128, hidden=9216),
        _config(128, intermediate=4608),
        _config(128, num_experts=32),
        _config(128, top_k=7),
        _config(128, gate_up_clamp=0.125),
        _config(48),
        _config(32, 4),
    ],
)
def test_w4a8_profile_requires_an_exact_match(config):
    with pytest.raises(W4A8KernelProfileMismatchError, match="no exact"):
        resolve_w4a8_kernel_config(config)


def test_w4a8_profile_schema_rejects_unknown_policy_fields(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_w4a8_profiles.json"
    value = json.loads(packaged.read_text())
    value["GB300"]["entries"][0]["config"]["unknown"] = 1
    path = tmp_path / "w4a8.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="expected fields"):
        load_w4a8_kernel_profiles(path)


@pytest.mark.parametrize("tile_n", [32, 64])
def test_external_w4a8_profiles_reject_removed_tile_sizes(tmp_path, tile_n):
    profile = load_w4a8_kernel_profiles()
    profile["GB300"]["entries"][0]["config"]["tile_n"] = tile_n
    path = tmp_path / "removed_tile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="tile_n"):
        load_w4a8_kernel_profiles(path)


def test_external_w4a8_profiles_accept_arbitrary_shapes(tmp_path):
    profile = load_w4a8_kernel_profiles()
    profile["GB300"]["entries"] = profile["GB300"]["entries"][:1]
    entry = profile["GB300"]["entries"][0]
    entry["match"].update(world_size=4, max_tokens=190, hidden=8192, intermediate=4608, local_experts=16, top_k=7)
    entry["config"]["tile_n"] = 128
    path = tmp_path / "external.json"
    path.write_text(json.dumps(profile))
    config = _config(190, 4, hidden=8192, intermediate=4608, num_experts=64, top_k=7)
    assert resolve_w4a8_kernel_config(config, profile_path=path) == W4A8KernelConfig(**entry["config"])


def test_w4a8_profiles_default_optional_optimization_flags_off(tmp_path):
    profile = load_w4a8_kernel_profiles()
    for entry in profile["GB300"]["entries"]:
        for flag in ("expert_major_tasks", "cache_weights", "adaptive_cache_weights"):
            entry["config"].pop(flag)
    path = tmp_path / "optional_flags.json"
    path.write_text(json.dumps(profile))
    assert all(
        entry["config"]["expert_major_tasks"] is False
        and entry["config"]["cache_weights"] is False
        and entry["config"]["adaptive_cache_weights"] is False
        for entry in load_w4a8_kernel_profiles(path)["GB300"]["entries"]
    )


@pytest.mark.parametrize("expert_major_tasks,cache_weights", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("cluster_size", [1, 2])
def test_w4a8_profiles_select_optimization_flags_per_token_bucket(
    tmp_path, expert_major_tasks, cache_weights, cluster_size
):
    profile = load_w4a8_kernel_profiles()
    entries = {entry["match"]["max_tokens"]: entry for entry in profile["GB300"]["entries"]}
    small, large = entries[32], entries[64]
    profile["GB300"]["entries"] = [small, large]
    small["config"]["expert_major_tasks"] = False
    small["config"]["cache_weights"] = False
    small["config"]["adaptive_cache_weights"] = False
    small["config"]["cluster_size"] = cluster_size
    large["config"]["expert_major_tasks"] = expert_major_tasks
    large["config"]["cache_weights"] = cache_weights
    large["config"]["adaptive_cache_weights"] = False
    large["config"]["cluster_size"] = 2
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(profile))
    small_policy = resolve_w4a8_kernel_config(_config(32), profile_path=path)
    large_policy = resolve_w4a8_kernel_config(_config(64), profile_path=path)
    assert small_policy.expert_major_tasks is False and small_policy.cache_weights is False
    assert small_policy.cluster_size == cluster_size
    assert large_policy.cluster_size == 2
    assert large_policy.expert_major_tasks is expert_major_tasks
    assert large_policy.cache_weights is cache_weights


def test_w4a8_profile_without_cluster_size_defaults_to_two_ctas(tmp_path):
    profile = load_w4a8_kernel_profiles()
    for entry in profile["GB300"]["entries"]:
        entry["config"].pop("cluster_size")
    path = tmp_path / "optional_cluster.json"
    path.write_text(json.dumps(profile))
    assert all(entry["config"]["cluster_size"] == 2 for entry in load_w4a8_kernel_profiles(path)["GB300"]["entries"])


def test_w4a8_profiles_roundtrip_adaptive_cache_policy(tmp_path):
    profile = load_w4a8_kernel_profiles()
    for entry in profile["GB300"]["entries"]:
        entry["config"]["cache_weights"] = False
        entry["config"]["adaptive_cache_weights"] = True
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps(profile))
    policy = resolve_w4a8_kernel_config(_config(), profile_path=path)
    assert policy.adaptive_cache_weights is True
    assert policy.cache_weights is False

    profile["GB300"]["entries"][0]["config"]["cache_weights"] = True
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_w4a8_kernel_profiles(path)


def test_w4a8_profile_schema_rejects_duplicates_and_versions(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_w4a8_profiles.json"
    value = json.loads(packaged.read_text())
    value["GB300"]["entries"].append(value["GB300"]["entries"][0])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate"):
        load_w4a8_kernel_profiles(duplicate)
    value["GB300"]["entries"].pop()
    value["version"] = 2
    version = tmp_path / "version.json"
    version.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="version"):
        load_w4a8_kernel_profiles(version)


def test_flat_legacy_profile_is_rejected(tmp_path):
    profile = load_w4a8_kernel_profiles()
    legacy = {"version": 1, "kind": profile["kind"], **profile["GB300"]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="GPU profile"):
        load_w4a8_kernel_profiles(path)


def test_profile_requires_gpu_sections(tmp_path):
    profile = load_w4a8_kernel_profiles()
    del profile["GB300"]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="GPU sections"):
        load_w4a8_kernel_profiles(path)
