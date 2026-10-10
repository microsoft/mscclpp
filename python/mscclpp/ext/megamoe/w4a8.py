# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact-match loading of external MegaMoE W4A8 JIT policies."""

from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path

from .jit import W4A8KernelConfig

PROFILE_VERSION = 1
PROFILE_KIND = "mscclpp-native-megamoe-w4a8-kernel-profiles"
DEFAULT_PROFILE_PATH = Path(__file__).with_name("megamoe_w4a8_profiles.json")
_MATCH_FIELDS = (
    "world_size",
    "max_tokens",
    "hidden",
    "intermediate",
    "local_experts",
    "top_k",
    "weight_mxfp4",
    "gate_up_clamp",
)
_CONFIG_FIELDS = tuple(field.name for field in fields(W4A8KernelConfig))


class W4A8KernelProfileMismatchError(ValueError):
    """No external W4A8 policy exactly covers the requested context."""


def _fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: expected fields {sorted(required)}; allowed fields {sorted(allowed)}")


def _integer(value, name, minimum=1):
    if type(value) is not int or not minimum <= value <= 2**31 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**31-1]")
    return value


def _validate_match(value):
    _fields(value, _MATCH_FIELDS, _MATCH_FIELDS, "W4A8 profile match")
    for name in ("world_size", "max_tokens", "hidden", "intermediate", "local_experts", "top_k"):
        _integer(value[name], name)
    if value["hidden"] % 128 or value["intermediate"] % 128:
        raise ValueError("W4A8 profile hidden and intermediate must be divisible by 128")
    if value["weight_mxfp4"] is not True:
        raise ValueError("W4A8 profile entries require weight_mxfp4=true")
    if value["gate_up_clamp"] != "negative":
        raise ValueError("W4A8 profile entries currently require an unclamped activation")
    return dict(value)


def _validate_profile(value):
    top_fields = ("_license", "version", "kind", "entries", "provenance")
    _fields(value, top_fields, top_fields[1:], "W4A8 kernel profile")
    if type(value["version"]) is not int or value["version"] != PROFILE_VERSION:
        raise ValueError("unsupported W4A8 kernel profile version")
    if value["kind"] != PROFILE_KIND:
        raise ValueError("unsupported W4A8 kernel profile kind")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ValueError("W4A8 kernel profile requires entries")

    entries = []
    seen = set()
    for entry in value["entries"]:
        _fields(entry, ("match", "config"), ("match", "config"), "W4A8 kernel profile entry")
        match = _validate_match(entry["match"])
        required_config = tuple(name for name in _CONFIG_FIELDS if name != "fixed_token_count")
        _fields(entry["config"], _CONFIG_FIELDS, required_config, "W4A8 kernel config")
        config = asdict(W4A8KernelConfig(**entry["config"]))
        key = json.dumps(match, sort_keys=True, separators=(",", ":"))
        if key in seen:
            raise ValueError("duplicate exact W4A8 kernel profile match")
        seen.add(key)
        entries.append({"match": match, "config": config})

    provenance_fields = ("created_at", "measured_at", "hardware", "source", "scope")
    _fields(value["provenance"], provenance_fields, provenance_fields, "W4A8 kernel profile provenance")
    if any(
        not isinstance(value["provenance"][name], str) or not value["provenance"][name] for name in provenance_fields
    ):
        raise ValueError("W4A8 kernel profile provenance values must be nonempty strings")
    return {
        "version": PROFILE_VERSION,
        "kind": PROFILE_KIND,
        "entries": entries,
        "provenance": dict(value["provenance"]),
    }


def load_w4a8_kernel_profiles(profile_path=None):
    """Load and strictly validate a versioned external W4A8 policy profile."""
    path = DEFAULT_PROFILE_PATH if profile_path is None else Path(profile_path)
    with path.open(encoding="utf-8") as source:
        return deepcopy(_validate_profile(json.load(source)))


def resolve_w4a8_kernel_config(config, *, profile_path=None):
    """Resolve one exact W4A8 policy for ``config`` or raise without fallback."""
    from .api import MegaMoEConfig

    if not isinstance(config, MegaMoEConfig):
        raise TypeError("config must be a MegaMoEConfig")
    actual = {
        "world_size": config.world_size,
        "max_tokens": config.max_tokens,
        "hidden": config.hidden,
        "intermediate": config.intermediate,
        "local_experts": config.local_experts,
        "top_k": config.top_k,
        "weight_mxfp4": config.weight_mxfp4,
        "gate_up_clamp": "negative" if config.gate_up_clamp < 0 else "nonnegative",
    }
    profile = load_w4a8_kernel_profiles(profile_path)
    matches = [entry for entry in profile["entries"] if entry["match"] == actual]
    if len(matches) != 1:
        requested = ", ".join(f"{name}={actual[name]!r}" for name in _MATCH_FIELDS)
        raise W4A8KernelProfileMismatchError(f"no exact W4A8 kernel profile entry for {requested}")
    return W4A8KernelConfig(**matches[0]["config"])
