# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact-match reuse of packaged MegaMoE resident-kernel tuning results."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

PROFILE_VERSION = 1
PROFILE_KIND = "mscclpp-native-megamoe-resident-tuning"
DEFAULT_PROFILE_PATH = Path(__file__).with_name("megamoe_resident_tuning.json")


class ResidentTuningMismatchError(ValueError):
    """The packaged resident profile does not exactly cover the requested run."""


@dataclass(frozen=True)
class ResidentTuning:
    """Resolved resident-grid winner and the policy under which it was measured."""

    sm_margin: int
    ctas: int
    kernel_policy: dict
    measurement: dict
    provenance: dict
    profile_path: str

    def apply(self, config):
        """Return ``config`` with the resolved resident SM margin."""
        return replace(config, sm_margin=self.sm_margin)


def _fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: expected fields {sorted(required)}; allowed fields {sorted(allowed)}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_profile(value):
    top_fields = ("_license", "version", "kind", "selection", "kernel_policy", "entries", "provenance")
    _fields(value, top_fields, top_fields[1:], "resident tuning profile")
    if type(value["version"]) is not int or value["version"] != PROFILE_VERSION:
        raise ValueError("unsupported resident tuning profile version")
    if value["kind"] != PROFILE_KIND:
        raise ValueError("unsupported resident tuning profile kind")

    selection_fields = (
        "device_name",
        "compute_capability",
        "sm_count",
        "world_size",
        "hidden",
        "num_experts",
        "local_experts",
        "top_k",
        "weight_mxfp4",
        "gate_up_clamp",
        "graph_batch",
        "graph_timing",
    )
    _fields(value["selection"], selection_fields, selection_fields, "resident selection")
    selection = value["selection"]
    if not isinstance(selection["device_name"], str) or not selection["device_name"]:
        raise ValueError("resident selection requires a device name")
    capability = selection["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability)
    ):
        raise ValueError("compute_capability must contain two nonnegative integers")
    for field in ("sm_count", "world_size", "hidden", "num_experts", "local_experts", "top_k", "graph_batch"):
        _integer(selection[field], field, 1)
    if selection["weight_mxfp4"] is not True or selection["gate_up_clamp"] != "negative":
        raise ValueError("resident tuning profile currently requires unclamped W4A8")
    if selection["graph_timing"] not in ("isolated", "steady-state"):
        raise ValueError("unsupported graph timing mode")

    policy = value["kernel_policy"]
    policy_fields = (
        "tile_m",
        "tile_n",
        "tile_k",
        "load_stages",
        "transform_stages",
        "cluster_m",
        "num_warps",
        "transfer_registers",
        "load_warps",
        "split_pipelines",
        "epilogue_tokens",
        "dispatch_chunk",
        "dispatch_warps",
        "dispatch_stages",
        "tma_cache_hints",
    )
    _fields(policy, policy_fields, policy_fields, "resident kernel policy")
    for field in (
        "tile_m",
        "tile_n",
        "tile_k",
        "load_stages",
        "cluster_m",
        "num_warps",
        "transfer_registers",
        "load_warps",
        "epilogue_tokens",
        "dispatch_chunk",
        "dispatch_warps",
        "dispatch_stages",
    ):
        _integer(policy[field], field, 1)
    _integer(policy["transform_stages"], "transform_stages")
    if type(policy["split_pipelines"]) is not bool:
        raise ValueError("split_pipelines must be bool")
    hints = policy["tma_cache_hints"]
    _fields(hints, ("weights", "weight_scales", "activations", "activation_scales"), hints, "TMA cache hints")
    if {hints["weights"], hints["weight_scales"]} != {"evict_first"} or {
        hints["activations"],
        hints["activation_scales"],
    } != {"evict_last"}:
        raise ValueError("resident profile contains unsupported TMA cache hints")

    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ValueError("resident tuning profile requires entries")
    seen = set()
    for entry in value["entries"]:
        entry_fields = ("intermediate", "tokens", "winner", "measurement")
        _fields(entry, entry_fields, entry_fields, "resident tuning entry")
        intermediate = _integer(entry["intermediate"], "intermediate", 1)
        tokens = _integer(entry["tokens"], "tokens", 1)
        if (intermediate, tokens) in seen:
            raise ValueError("duplicate resident tuning entry")
        seen.add((intermediate, tokens))
        _fields(entry["winner"], ("sm_margin", "ctas"), ("sm_margin", "ctas"), "resident winner")
        margin = _integer(entry["winner"]["sm_margin"], "sm_margin")
        ctas = _integer(entry["winner"]["ctas"], "ctas", 2)
        if ctas % policy["cluster_m"]:
            raise ValueError("resident CTA count must preserve cluster alignment")
        if selection["sm_count"] - margin != ctas:
            raise ValueError("resident winner CTA count does not match sm_margin")
        measurement_fields = ("resident_median_us", "bf16_e2e_median_us", "samples", "measured_kernel_commit")
        _fields(entry["measurement"], measurement_fields, measurement_fields, "resident measurement")
        for field in ("resident_median_us", "bf16_e2e_median_us"):
            if not isinstance(entry["measurement"][field], (int, float)) or entry["measurement"][field] <= 0:
                raise ValueError(f"{field} must be positive")
        _integer(entry["measurement"]["samples"], "samples", 1)
        commit = entry["measurement"]["measured_kernel_commit"]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("measured_kernel_commit must be a full lowercase hexadecimal commit")
    provenance_fields = (
        "measured_at",
        "hardware",
        "cuda_runtime",
        "cuda_compiler",
        "torch",
        "source_commit",
        "scope",
    )
    _fields(value["provenance"], provenance_fields, provenance_fields, "resident provenance")
    if any(
        not isinstance(value["provenance"][field], str) or not value["provenance"][field] for field in provenance_fields
    ):
        raise ValueError("resident provenance values must be nonempty strings")
    if any(
        entry["measurement"]["measured_kernel_commit"] != value["provenance"]["source_commit"]
        for entry in value["entries"]
    ):
        raise ValueError("resident measurements do not match the profile source commit")
    return value


def _read_profile(path):
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        return _validate_profile(json.load(source))


def _hardware(device):
    import torch

    if device is None:
        device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
    }


def _normalize_hardware(value):
    _fields(
        value,
        ("device_name", "compute_capability", "sm_count"),
        ("device_name", "compute_capability", "sm_count"),
        "hardware",
    )
    result = {
        "device_name": value["device_name"],
        "compute_capability": list(value["compute_capability"]),
        "sm_count": value["sm_count"],
    }
    if not isinstance(result["device_name"], str) or not result["device_name"]:
        raise ValueError("hardware requires a device name")
    if len(result["compute_capability"]) != 2:
        raise ValueError("hardware compute_capability must have two values")
    for index, item in enumerate(result["compute_capability"]):
        _integer(item, f"compute_capability[{index}]")
    _integer(result["sm_count"], "sm_count", 1)
    return result


def resolve_resident_tuning(
    config,
    *,
    tokens=None,
    graph_batch=10,
    graph_timing="steady-state",
    device=None,
    hardware=None,
    profile_path=None,
):
    """Resolve one packaged resident winner without compiling or benchmarking.

    The match is exact across hardware, EP size, shape, precision, capacity,
    and graph timing. Missing coverage is an error rather than a fallback.
    """

    from .api import MegaMoEConfig

    if not isinstance(config, MegaMoEConfig):
        raise TypeError("config must be a MegaMoEConfig")
    if tokens is None:
        tokens = config.max_tokens
    _integer(tokens, "tokens", 1)
    _integer(graph_batch, "graph_batch", 1)
    if tokens != config.max_tokens:
        raise ResidentTuningMismatchError(
            f"resident tuning requires tokens == max_tokens, got {tokens} and {config.max_tokens}"
        )
    if graph_timing not in ("isolated", "steady-state"):
        raise ValueError("graph_timing must be isolated or steady-state")

    path = DEFAULT_PROFILE_PATH if profile_path is None else Path(profile_path)
    profile = _read_profile(path)
    detected = _hardware(device) if hardware is None else _normalize_hardware(hardware)
    actual = {
        **detected,
        "world_size": config.world_size,
        "hidden": config.hidden,
        "num_experts": config.num_experts,
        "local_experts": config.local_experts,
        "top_k": config.top_k,
        "weight_mxfp4": config.weight_mxfp4,
        "gate_up_clamp": "negative" if config.gate_up_clamp < 0 else "nonnegative",
        "graph_batch": graph_batch,
        "graph_timing": graph_timing,
    }
    expected = profile["selection"]
    mismatches = [
        f"{field}: expected {expected[field]!r}, got {actual[field]!r}"
        for field in expected
        if actual[field] != expected[field]
    ]
    if mismatches:
        raise ResidentTuningMismatchError("resident tuning profile mismatch: " + "; ".join(mismatches))

    matches = [
        entry
        for entry in profile["entries"]
        if entry["intermediate"] == config.intermediate and entry["tokens"] == tokens
    ]
    if len(matches) != 1:
        raise ResidentTuningMismatchError(
            f"no exact resident tuning entry for intermediate={config.intermediate}, tokens={tokens}"
        )
    entry = matches[0]
    return ResidentTuning(
        entry["winner"]["sm_margin"],
        entry["winner"]["ctas"],
        deepcopy(profile["kernel_policy"]),
        deepcopy(entry["measurement"]),
        deepcopy(profile["provenance"]),
        str(path),
    )
