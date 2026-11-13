# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Token-range MegaMoE policies: W4A8 JIT kernel plus runtime knobs, optionally for a fused shared expert.

A policy profile (``megamoe_policies.json``) is grouped by canonical GPU name. Each entry matches a
workload (world size, routed shape, dtypes, optional fused shared-expert shape) and an inclusive token
range, and carries a complete ``W4A8KernelConfig`` together with runtime values (``sm_margin``,
``fc1_delay_ns``) and optional measurements. Lookups pick the narrowest token range that covers the
requested token count. Profiles are produced by ``python -m mscclpp.ext.megamoe.tune``.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path

from .jit import W4A8KernelConfig

PROFILE_VERSION = 2
PROFILE_KIND = "mscclpp-native-megamoe-policies"
DEFAULT_PROFILE_PATH = Path(__file__).with_name("megamoe_policies.json")
_MATCH_FIELDS = (
    "world_size",
    "hidden",
    "intermediate",
    "local_experts",
    "top_k",
    "weight_dtype",
    "activation_dtype",
    "gate_up_clamp",
    "shared",
    "tokens",
)
_SHARED_FIELDS = ("hidden", "intermediate")
_RUNTIME_FIELDS = ("sm_margin", "fc1_delay_ns")
_CONFIG_FIELDS = tuple(field.name for field in fields(W4A8KernelConfig))


class MegaMoEPolicyMismatchError(ValueError):
    """No policy covers the requested GPU, workload, and token count."""


@dataclass(frozen=True)
class MegaMoEPolicy:
    """Resolved kernel specialization and runtime knobs for one token range."""

    tokens: tuple[int, int]
    """Inclusive token range covered by this policy."""
    kernel: W4A8KernelConfig
    """Routed (and fused shared-expert) W4A8 JIT specialization."""
    sm_margin: int | None = None
    """SMs left free by the routed kernel; ``None`` keeps the caller's value."""
    fc1_delay_ns: int | None = None
    """Fused shared FC1 start delay; ``None`` keeps the caller's or native default."""
    measurement: dict | None = None
    """Optional free-form tuning record (for example per-pattern latencies)."""

    def apply(self, config, shared_config=None):
        """Return ``(config, shared_config)`` with this policy's runtime values applied.

        ``config`` is a ``MegaMoEConfig`` and ``shared_config`` an optional ``SharedExpertConfig``.
        Values that the policy leaves unset are kept.
        """
        if self.sm_margin is not None:
            config = replace(config, sm_margin=self.sm_margin)
        if shared_config is not None and self.fc1_delay_ns is not None:
            shared_config = replace(shared_config, fc1_delay_ns=self.fc1_delay_ns)
        return config, shared_config


def _fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: expected fields {sorted(required)}; allowed fields {sorted(allowed)}")


def _integer(value, name, minimum=1, maximum=2**31 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_match(value):
    _fields(value, _MATCH_FIELDS, _MATCH_FIELDS, "MegaMoE policy match")
    for name in ("world_size", "hidden", "intermediate", "local_experts", "top_k"):
        _integer(value[name], name)
    if value["weight_dtype"] != "mxfp4" or value["activation_dtype"] not in ("fp8_static", "mxfp8"):
        raise ValueError("MegaMoE policies cover W4A8 (weight_dtype='mxfp4') workloads only")
    if value["gate_up_clamp"] not in ("negative", "nonnegative"):
        raise ValueError("policy gate_up_clamp must be 'negative' or 'nonnegative'")
    shared = value["shared"]
    if shared is not None:
        _fields(shared, _SHARED_FIELDS, _SHARED_FIELDS, "MegaMoE policy shared match")
        for name in _SHARED_FIELDS:
            _integer(shared[name], f"shared {name}")
        if value["activation_dtype"] != "fp8_static":
            raise ValueError("fused shared-expert policies require activation_dtype='fp8_static'")
        shared = dict(shared)
    tokens = value["tokens"]
    if not isinstance(tokens, list) or len(tokens) != 2:
        raise ValueError("policy tokens must be an inclusive [minimum, maximum] pair")
    low = _integer(tokens[0], "tokens minimum")
    high = _integer(tokens[1], "tokens maximum", low)
    return {**value, "shared": shared, "tokens": [low, high]}


def _validate_entry(entry):
    _fields(entry, ("match", "kernel", "runtime", "measurement"), ("match", "kernel"), "MegaMoE policy entry")
    match = _validate_match(entry["match"])
    _fields(entry["kernel"], _CONFIG_FIELDS, _CONFIG_FIELDS, "MegaMoE policy kernel")
    kernel = asdict(W4A8KernelConfig(**entry["kernel"]))
    runtime = entry.get("runtime", {})
    _fields(runtime, _RUNTIME_FIELDS, (), "MegaMoE policy runtime")
    if runtime.get("sm_margin") is not None:
        _integer(runtime["sm_margin"], "sm_margin", 0)
    if runtime.get("fc1_delay_ns") is not None:
        if match["shared"] is None:
            raise ValueError("fc1_delay_ns applies only to fused shared-expert policies")
        _integer(runtime["fc1_delay_ns"], "fc1_delay_ns", 0, 2**24 - 1)
    measurement = entry.get("measurement")
    if measurement is not None and not isinstance(measurement, dict):
        raise ValueError("policy measurement must be an object")
    return {"match": match, "kernel": kernel, "runtime": dict(runtime), "measurement": deepcopy(measurement)}


def _workload_key(match):
    return json.dumps({k: v for k, v in match.items() if k != "tokens"}, sort_keys=True, separators=(",", ":"))


def _validate_gpu_profile(value):
    _fields(value, ("entries", "provenance"), ("entries", "provenance"), "MegaMoE GPU policy profile")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ValueError("MegaMoE policy profile requires entries")
    entries = [_validate_entry(entry) for entry in value["entries"]]
    seen = set()
    for entry in entries:
        key = (_workload_key(entry["match"]), tuple(entry["match"]["tokens"]))
        if key in seen:
            raise ValueError("duplicate MegaMoE policy token range for one workload")
        seen.add(key)
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or not all(isinstance(v, str) and v for v in provenance.values()):
        raise ValueError("MegaMoE policy provenance must map names to nonempty strings")
    return {"entries": entries, "provenance": dict(provenance)}


def gpu_key(device_name):
    """Canonical GPU profile key, for example ``'GB300'`` for ``'NVIDIA GB300'``."""
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("GPU device name must be a nonempty string")
    return device_name.strip().removeprefix("NVIDIA ")


def validate_megamoe_policies(value):
    """Validate a policy profile object and return a normalized copy."""
    if not isinstance(value, dict):
        raise ValueError("MegaMoE policy profile must be an object")
    metadata = ("_license", "version", "kind")
    gpus = tuple(key for key in value if key not in metadata)
    _fields(value, metadata + gpus, ("version", "kind"), "MegaMoE policy profile")
    if type(value["version"]) is not int or value["version"] != PROFILE_VERSION:
        raise ValueError("unsupported MegaMoE policy profile version")
    if value["kind"] != PROFILE_KIND:
        raise ValueError("unsupported MegaMoE policy profile kind")
    profiles = {}
    for gpu in gpus:
        if gpu != gpu_key(gpu):
            raise ValueError(f"GPU profile key must be canonical, for example 'GB300': {gpu!r}")
        profiles[gpu] = _validate_gpu_profile(value[gpu])
    result = {"version": PROFILE_VERSION, "kind": PROFILE_KIND, **profiles}
    if "_license" in value:
        result["_license"] = value["_license"]
    return result


def load_megamoe_policies(profile_path=None):
    """Load and validate a policy profile (the packaged default when ``profile_path`` is None)."""
    path = DEFAULT_PROFILE_PATH if profile_path is None else Path(profile_path)
    with path.open(encoding="utf-8") as source:
        return validate_megamoe_policies(json.load(source))


def workload_match(config, shared_config=None):
    """Policy match fields (without ``tokens``) for a ``MegaMoEConfig`` and optional ``SharedExpertConfig``."""
    return {
        "world_size": config.world_size,
        "hidden": config.hidden,
        "intermediate": config.intermediate,
        "local_experts": config.local_experts,
        "top_k": config.top_k,
        "weight_dtype": config.weight_dtype,
        "activation_dtype": config.activation_dtype,
        "gate_up_clamp": "negative" if config.gate_up_clamp < 0 else "nonnegative",
        "shared": (
            None
            if shared_config is None
            else {"hidden": shared_config.hidden or config.hidden, "intermediate": shared_config.intermediate}
        ),
    }


def _policy(entry):
    runtime = entry["runtime"]
    return MegaMoEPolicy(
        tokens=tuple(entry["match"]["tokens"]),
        kernel=W4A8KernelConfig(**entry["kernel"]),
        sm_margin=runtime.get("sm_margin"),
        fc1_delay_ns=runtime.get("fc1_delay_ns"),
        measurement=deepcopy(entry["measurement"]),
    )


def resolve_megamoe_policies(config, shared_config=None, *, profile_path=None, device_name=None):
    """Return all policies for this workload ordered by token range (narrowest first per start)."""
    from .api import MegaMoEConfig, SharedExpertConfig

    if not isinstance(config, MegaMoEConfig):
        raise TypeError("config must be a MegaMoEConfig")
    if shared_config is not None and not isinstance(shared_config, SharedExpertConfig):
        raise TypeError("shared_config must be a SharedExpertConfig or None")
    profile = load_megamoe_policies(profile_path)
    if device_name is None:
        import torch

        device_name = torch.cuda.get_device_name()
    gpu = gpu_key(device_name)
    if gpu not in profile:
        raise MegaMoEPolicyMismatchError(f"no MegaMoE policy profile for GPU {gpu!r}")
    actual = workload_match(config, shared_config)
    entries = [
        entry
        for entry in profile[gpu]["entries"]
        if {k: v for k, v in entry["match"].items() if k != "tokens"} == actual
    ]
    if not entries:
        requested = ", ".join(f"{name}={value!r}" for name, value in actual.items())
        raise MegaMoEPolicyMismatchError(f"no MegaMoE policy for GPU {gpu!r}: {requested}")
    entries.sort(key=lambda entry: (entry["match"]["tokens"][0], entry["match"]["tokens"][1]))
    return [_policy(entry) for entry in entries]


def resolve_megamoe_policy(config, shared_config=None, *, tokens=None, profile_path=None, device_name=None):
    """Resolve the narrowest-token-range policy covering ``tokens`` (default ``config.max_tokens``)."""
    tokens = config.max_tokens if tokens is None else tokens
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    candidates = [
        policy
        for policy in resolve_megamoe_policies(
            config, shared_config, profile_path=profile_path, device_name=device_name
        )
        if policy.tokens[0] <= tokens <= policy.tokens[1]
    ]
    if not candidates:
        raise MegaMoEPolicyMismatchError(f"no MegaMoE policy token range covers {tokens} tokens")
    return min(candidates, key=lambda policy: (policy.tokens[1] - policy.tokens[0], policy.tokens[0]))


def bucket_limits(policies, max_tokens):
    """Token bucket upper bounds (ascending, capped at ``max_tokens``) implied by policy token ranges."""
    limits = sorted({min(policy.tokens[1], max_tokens) for policy in policies if policy.tokens[0] <= max_tokens})
    if not limits or limits[-1] < max_tokens:
        limits.append(max_tokens)
    return limits


def mean_latency(values):
    """Arithmetic mean of finite latencies, or ``math.inf`` when any are missing."""
    values = list(values)
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return math.inf
    return sum(values) / len(values)
