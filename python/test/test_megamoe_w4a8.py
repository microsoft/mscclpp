# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only contracts for external W4A8 JIT policy profiles."""

import json
from pathlib import Path

import pytest

from mscclpp.ext.megamoe import (
    MegaMoEConfig,
    W4A8KernelConfig,
    W4A8KernelProfileMismatchError,
    load_w4a8_kernel_profiles,
    resolve_w4a8_kernel_config,
)


def _config(tokens=128, world_size=4, **overrides):
    fields = {
        "rank": 0,
        "world_size": world_size,
        "max_tokens": tokens,
        "hidden": 8192,
        "intermediate": 4096,
        "num_experts": 16 * world_size,
        "top_k": 8,
        "weight_mxfp4": True,
    }
    fields.update(overrides)
    return MegaMoEConfig(**fields)


@pytest.mark.parametrize("world_size", [4, 32])
@pytest.mark.parametrize("tokens", [16, 32, 64])
def test_packaged_w4a8_default_policy_exact_matches(world_size, tokens):
    assert resolve_w4a8_kernel_config(_config(tokens, world_size)) == W4A8KernelConfig()


@pytest.mark.parametrize("world_size", [4, 32])
def test_packaged_w4a8_t128_policy_exact_matches(world_size):
    policy = resolve_w4a8_kernel_config(_config(128, world_size))
    assert policy == W4A8KernelConfig(
        tile_n=128,
        load_stages=7,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
    )


def test_packaged_h4096_i6656_t128_policy_exact_matches():
    policy = resolve_w4a8_kernel_config(_config(128, hidden=4096, intermediate=6656, num_experts=128))
    assert policy == W4A8KernelConfig(
        transfer_registers=128,
        epilogue_warps=8,
        epilogue_registers=176,
        dispatch_chunk=4096,
        fixed_token_count=True,
    )


def test_packaged_h4096_i6656_case_g_policy_exact_matches():
    policy = resolve_w4a8_kernel_config(_config(190, hidden=4096, intermediate=6656, num_experts=128))
    assert policy == W4A8KernelConfig(
        tile_n=128,
        load_stages=7,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
    )


@pytest.mark.parametrize(
    "config",
    [
        _config(8),
        _config(128, 2),
        _config(128, hidden=9216),
        _config(128, intermediate=4608),
        _config(128, num_experts=32),
        _config(128, top_k=7),
        _config(128, weight_mxfp4=False),
        _config(128, gate_up_clamp=0.125),
    ],
)
def test_w4a8_profile_requires_an_exact_match(config):
    with pytest.raises(W4A8KernelProfileMismatchError, match="no exact"):
        resolve_w4a8_kernel_config(config)


def test_w4a8_profile_schema_rejects_unknown_policy_fields(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_w4a8_profiles.json"
    value = json.loads(packaged.read_text())
    value["entries"][0]["config"]["unknown"] = 1
    path = tmp_path / "w4a8.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="expected fields"):
        load_w4a8_kernel_profiles(path)


def test_w4a8_profile_schema_rejects_duplicates_and_versions(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_w4a8_profiles.json"
    value = json.loads(packaged.read_text())
    value["entries"].append(value["entries"][0])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate"):
        load_w4a8_kernel_profiles(duplicate)
    value["entries"].pop()
    value["version"] = 2
    version = tmp_path / "version.json"
    version.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="version"):
        load_w4a8_kernel_profiles(version)
