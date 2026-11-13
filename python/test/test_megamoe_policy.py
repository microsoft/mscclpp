# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import asdict
import json

import pytest

from mscclpp.ext.megamoe import (
    MegaMoEConfig,
    MegaMoEPolicyMismatchError,
    SharedExpertConfig,
    W4A8KernelConfig,
    load_megamoe_policies,
    resolve_megamoe_policies,
    resolve_megamoe_policy,
)
from mscclpp.ext.megamoe.policy import bucket_limits, validate_megamoe_policies
from mscclpp.ext.megamoe.tune import build_policies


def _config(**overrides):
    fields = dict(
        rank=0,
        world_size=32,
        max_tokens=16,
        hidden=4096,
        intermediate=4352,
        num_experts=512,
        top_k=7,
        activation_dtype="fp8_static",
        activation_scale=1.0,
    )
    fields.update(overrides)
    return MegaMoEConfig(**fields)


SHARED = SharedExpertConfig(2048, hidden=8704)


def _record(tokens, pattern, latency, *, fused=False, stages=8, margin=0, delay=None):
    return {
        "gpu": "GB300",
        "match": {
            "world_size": 32,
            "hidden": 4096,
            "intermediate": 4352,
            "local_experts": 16,
            "top_k": 7,
            "weight_dtype": "mxfp4",
            "activation_dtype": "fp8_static",
            "gate_up_clamp": "negative",
            "shared": {"hidden": 8704, "intermediate": 2048} if fused else None,
        },
        "tokens": tokens,
        "pattern": pattern,
        "kernel": asdict(W4A8KernelConfig(load_stages=stages)),
        "runtime": {"sm_margin": margin, "fc1_delay_ns": delay},
        "latency_us": latency,
    }


def test_packaged_policies_resolve_by_token_range():
    load_megamoe_policies()
    for tokens, expected in ((1, (1, 8)), (8, (1, 8)), (9, (9, 16)), (64, (33, 64))):
        policy = resolve_megamoe_policy(_config(max_tokens=tokens), device_name="NVIDIA GB300")
        assert policy.tokens == expected and policy.fc1_delay_ns is None
        fused = resolve_megamoe_policy(_config(max_tokens=tokens), SHARED, device_name="GB300")
        assert fused.tokens == expected and fused.fc1_delay_ns is not None
    with pytest.raises(MegaMoEPolicyMismatchError):
        resolve_megamoe_policy(_config(max_tokens=65), device_name="GB300")
    with pytest.raises(MegaMoEPolicyMismatchError):
        resolve_megamoe_policy(_config(top_k=8), device_name="GB300")
    with pytest.raises(MegaMoEPolicyMismatchError):
        resolve_megamoe_policy(_config(), device_name="GB200")


def test_policy_apply_and_buckets():
    policy = resolve_megamoe_policy(_config(), SHARED, device_name="GB300")
    config, shared = policy.apply(_config(sm_margin=10), SHARED)
    assert config.sm_margin == policy.sm_margin and shared.fc1_delay_ns == policy.fc1_delay_ns
    policies = resolve_megamoe_policies(_config(), SHARED, device_name="GB300")
    assert bucket_limits(policies, 64) == [8, 16, 32, 64]
    assert bucket_limits(policies, 20) == [8, 16, 20]
    assert bucket_limits(policies, 100) == [8, 16, 32, 64, 100]


def test_build_selects_best_mean_across_patterns(tmp_path):
    records = [
        _record(8, "balanced", 80.0, stages=8),
        _record(8, "zipf", 90.0, stages=8),
        _record(8, "balanced", 82.0, stages=10),
        _record(8, "zipf", 85.0, stages=10),
        _record(8, "balanced", 70.0, stages=6),  # missing zipf: not eligible
        _record(32, "balanced", 95.0, margin=2),
        _record(32, "zipf", 96.0, margin=2),
        _record(8, "balanced", 88.0, fused=True, delay=2500),
        _record(8, "zipf", 89.0, fused=True, delay=2500),
    ]
    profile = build_policies(records, provenance={"source": "test"})
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(profile))
    entries = profile["GB300"]["entries"]
    routed = [e for e in entries if e["match"]["shared"] is None]
    assert [e["match"]["tokens"] for e in routed] == [[1, 8], [9, 32]]
    assert routed[0]["kernel"]["load_stages"] == 10 and routed[0]["measurement"]["mean_us"] == 83.5
    assert routed[1]["runtime"] == {"sm_margin": 2}
    fused = resolve_megamoe_policy(_config(max_tokens=4), SHARED, profile_path=path, device_name="GB300")
    assert fused.fc1_delay_ns == 2500


def test_policy_validation_rejects_bad_entries():
    profile = load_megamoe_policies()
    entry = profile["GB300"]["entries"][0]
    bad = json.loads(json.dumps(profile))
    bad["GB300"]["entries"].append(entry)
    with pytest.raises(ValueError, match="duplicate"):
        validate_megamoe_policies(bad)
    bad = json.loads(json.dumps(profile))
    bad["GB300"]["entries"][0]["match"]["tokens"] = [9, 8]
    with pytest.raises(ValueError):
        validate_megamoe_policies(bad)
    bad = json.loads(json.dumps(profile))
    routed = next(e for e in bad["GB300"]["entries"] if e["match"]["shared"] is None)
    routed["runtime"]["fc1_delay_ns"] = 100
    with pytest.raises(ValueError, match="fc1_delay_ns"):
        validate_megamoe_policies(bad)
