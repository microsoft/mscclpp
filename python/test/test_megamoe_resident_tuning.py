# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only contracts for packaged MegaMoE resident tuning results."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mscclpp.ext.megamoe import MegaMoEConfig, ResidentTuningMismatchError, resolve_resident_tuning

HARDWARE = {"device_name": "NVIDIA GB200", "compute_capability": [10, 0], "sm_count": 152}


def _config(intermediate=4096, tokens=32, **overrides):
    fields = {
        "rank": 0,
        "world_size": 32,
        "max_tokens": tokens,
        "hidden": 9216,
        "intermediate": intermediate,
        "num_experts": 512,
        "top_k": 8,
        "weight_mxfp4": True,
    }
    fields.update(overrides)
    return MegaMoEConfig(**fields)


@pytest.mark.parametrize(
    "intermediate,tokens,sm_margin,ctas",
    [
        (4096, 32, 4, 148),
        (4096, 64, 4, 148),
        (4096, 128, 2, 150),
        (4608, 32, 4, 148),
        (4608, 64, 6, 146),
        (4608, 128, 2, 150),
    ],
)
def test_packaged_resident_winners(intermediate, tokens, sm_margin, ctas):
    config = _config(intermediate, tokens)
    tuning = resolve_resident_tuning(config, hardware=HARDWARE)
    assert (tuning.sm_margin, tuning.ctas) == (sm_margin, ctas)
    assert tuning.apply(config) == replace(config, sm_margin=sm_margin)
    assert tuning.kernel_policy["tile_n"] == 64
    assert tuning.kernel_policy["load_stages"] == 9
    assert tuning.kernel_policy["tma_cache_hints"] == {
        "weights": "evict_first",
        "weight_scales": "evict_first",
        "activations": "evict_last",
        "activation_scales": "evict_last",
    }
    assert tuning.measurement["measured_kernel_commit"] == tuning.provenance["source_commit"]


@pytest.mark.parametrize(
    "config,kwargs,match",
    [
        (_config(tokens=32), {"tokens": 16}, "tokens == max_tokens"),
        (_config(tokens=32), {"graph_batch": 1}, "graph_batch"),
        (_config(tokens=32), {"graph_timing": "isolated"}, "graph_timing"),
        (_config(tokens=32, world_size=4, num_experts=64), {}, "world_size"),
        (_config(tokens=32, hidden=8704), {}, "hidden"),
        (_config(tokens=32, num_experts=1024), {}, "num_experts"),
        (_config(tokens=32, top_k=7), {}, "top_k"),
        (_config(tokens=32, weight_mxfp4=False), {}, "weight_mxfp4"),
        (_config(tokens=32, gate_up_clamp=0.125), {}, "gate_up_clamp"),
        (_config(intermediate=4352, tokens=32), {}, "no exact resident tuning entry"),
    ],
)
def test_resident_mismatches_are_explicit(config, kwargs, match):
    with pytest.raises(ResidentTuningMismatchError, match=match):
        resolve_resident_tuning(config, hardware=HARDWARE, **kwargs)


@pytest.mark.parametrize(
    "hardware,match",
    [
        ({"device_name": "NVIDIA H100", "compute_capability": [9, 0], "sm_count": 132}, "device_name"),
        ({"device_name": "NVIDIA GB200", "compute_capability": [10, 0], "sm_count": 144}, "sm_count"),
    ],
)
def test_resident_hardware_mismatch(hardware, match):
    with pytest.raises(ResidentTuningMismatchError, match=match):
        resolve_resident_tuning(_config(), hardware=hardware)


def test_resident_profile_schema_rejects_invalid_cta_count(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_resident_tuning.json"
    value = json.loads(packaged.read_text())
    value["entries"][0]["winner"]["ctas"] = 146
    path = tmp_path / "resident.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not match sm_margin"):
        resolve_resident_tuning(_config(), hardware=HARDWARE, profile_path=path)


def test_resident_profile_rejects_stale_measurements(tmp_path):
    packaged = Path(__file__).parents[1] / "mscclpp" / "ext" / "megamoe" / "megamoe_resident_tuning.json"
    value = json.loads(packaged.read_text())
    value["entries"][0]["measurement"]["measured_kernel_commit"] = "0" * 40
    path = tmp_path / "resident.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="do not match"):
        resolve_resident_tuning(_config(), hardware=HARDWARE, profile_path=path)
