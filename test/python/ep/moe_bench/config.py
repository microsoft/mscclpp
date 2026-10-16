# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strict named-preset configuration for the MoE benchmark."""

from copy import deepcopy
from pathlib import Path

import yaml

# Common workload dimensions, constrained by MSCCL++ expert-major combine.
SUPPORTED_D_MODEL = (4096, 4352, 5120, 6656, 7168, 8192, 8704, 9216)
BENCHMARK_DEFAULTS = {"warmup": 10, "iterations": 100, "seed": 42, "output_dir": "benchmark_results"}
COMPUTE_DEFAULTS = {"activation_dtype": "bf16", "weight_dtype": "bf16"}
COMMUNICATION_DEFAULTS = {"mode": "latency", "layout": "expert_major"}
SUPPORTED_LAYOUTS = {
    "mscclpp": {"latency": ("expert_major", "rank_major"), "throughput": ("token_major", "rank_major")},
    "nccl": {"latency": ("expert_major", "rank_major"), "throughput": ("expert_major", "token_major")},
}
_FIELDS = {
    "model": {"d_model", "num_experts", "top_k", "intermediate_size", "activation"},
    "input": {"tokens_per_rank", "dtype"},
    "environment": {"num_gpus", "communication_backend", "compute_backend", "cuda_graph", "mode", "layout"},
    "routing": {"kind"},
    "benchmark": set(BENCHMARK_DEFAULTS),
}
_CATEGORIES = {"models": "model", "inputs": "input", "routing": "routing"}


def _mapping(value, path):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")


def _fields(value, fields, path, *, required=True):
    _mapping(value, path)
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
    missing = fields - set(value)
    if required and missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")


def _integer(value, path, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")


def _choice(value, choices, path):
    if not any(type(value) is type(choice) and value == choice for choice in choices):
        raise ValueError(f"{path} must be one of {choices!r}; got {value!r}")


def _validate_section(section, value, path):
    if section == "environment":
        _mapping(value, path)
        value = {**COMMUNICATION_DEFAULTS, **value}
    fields = _FIELDS[section]
    if section == "routing" and isinstance(value, dict) and value.get("kind") == "hot_experts":
        fields = fields | {"expert_ids", "probability_mass"}
    _fields(value, fields, path)
    if section == "model":
        for name in ("d_model", "num_experts", "top_k", "intermediate_size"):
            _integer(value[name], f"{path}.{name}")
        _choice(value["d_model"], SUPPORTED_D_MODEL, f"{path}.d_model")
        if value["top_k"] > min(9, value["num_experts"]):
            raise ValueError(f"{path}.top_k must be <= num_experts and <= 9 (EP combine limit)")
        _choice(value["activation"], ("swiglu",), f"{path}.activation")
    elif section == "input":
        _integer(value["tokens_per_rank"], f"{path}.tokens_per_rank")
        _choice(value["dtype"], ("bf16",), f"{path}.dtype")
    elif section == "environment":
        _choice(value["num_gpus"], (2, 4), f"{path}.num_gpus")
        _choice(value["communication_backend"], ("mscclpp", "nccl"), f"{path}.communication_backend")
        _choice(value["compute_backend"], ("torch", "cutlass_flashinfer"), f"{path}.compute_backend")
        _choice(value["cuda_graph"], (False, "mlp"), f"{path}.cuda_graph")
        _choice(value["mode"], ("latency", "throughput"), f"{path}.mode")
        _choice(value["layout"], ("expert_major", "rank_major", "token_major"), f"{path}.layout")
    elif section == "routing":
        _choice(value["kind"], ("balanced", "uniform", "hot_experts"), f"{path}.kind")
        if value["kind"] == "hot_experts":
            ids = value["expert_ids"]
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"{path}.expert_ids must be a nonempty list")
            for expert in ids:
                _integer(expert, f"{path}.expert_ids entry", 0)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{path}.expert_ids must be unique")
            probability = value["probability_mass"]
            if type(probability) not in (float, int) or not 0 < probability < 1:
                raise ValueError(f"{path}.probability_mass must be a number strictly between 0 and 1")
    else:
        _integer(value["warmup"], f"{path}.warmup", 0)
        _integer(value["iterations"], f"{path}.iterations")
        _integer(value["seed"], f"{path}.seed", 0)
        if value["seed"] >= 2**63:
            raise ValueError(f"{path}.seed must be less than 2**63")
        if not isinstance(value["output_dir"], str) or not value["output_dir"].strip():
            raise ValueError(f"{path}.output_dir must be a nonempty string")


def validate_config(config: dict) -> None:
    """Raise ValueError for malformed, unsupported, or incompatible settings."""
    _mapping(config, "config")
    _fields({key: value for key, value in config.items() if key != "compute"}, set(_FIELDS), "config")
    for section, value in config.items():
        if section == "compute":
            continue
        _validate_section(section, value, section)
    environment = {**COMMUNICATION_DEFAULTS, **config["environment"]}
    communication, mode, layout = (environment[key] for key in ("communication_backend", "mode", "layout"))
    supported = SUPPORTED_LAYOUTS[communication][mode]
    if layout not in supported:
        raise ValueError(
            f"{communication} mode={mode} does not support layout={layout}; supported layouts: {supported}"
        )
    if communication == "nccl" and mode == "throughput" and config["input"]["tokens_per_rank"] % 2:
        raise ValueError("NCCL-EP throughput benchmark requires even tokens_per_rank for native S2D transfer alignment")
    if communication == "mscclpp" and mode == "throughput":
        if config["model"]["num_experts"] // environment["num_gpus"] > 1024:
            raise ValueError("MSCCL++ throughput supports at most 1024 local experts")
        if environment["num_gpus"] * config["input"]["tokens_per_rank"] > 65536:
            raise ValueError("MSCCL++ throughput receive pool supports at most 65536 rows (num_gpus * tokens_per_rank)")
        capacity = environment["num_gpus"] * config["input"]["tokens_per_rank"]
        if capacity * config["model"]["d_model"] * 2 > 65536 * 16384:
            raise ValueError("MSCCL++ throughput worst-case receive payload exceeds the native 1 GiB pool")
    compute = config.get("compute", COMPUTE_DEFAULTS)
    _fields(compute, set(COMPUTE_DEFAULTS), "compute", required=False)
    activation = compute.get("activation_dtype", "bf16")
    weight = compute.get("weight_dtype", "bf16")
    _choice(activation, ("bf16", "mxfp8"), "compute.activation_dtype")
    if weight == "mxfp8":
        raise ValueError("MXFP8 weights (including BF16/MXFP8 weight-only compute) are not supported; use MXFP8/MXFP4")
    _choice(weight, ("bf16", "mxfp4"), "compute.weight_dtype")
    backend = config["environment"]["compute_backend"]
    pairs = (("bf16", "bf16"),) if backend == "torch" else (("mxfp8", "mxfp4"),)
    if (activation, weight) not in pairs:
        raise ValueError(
            f"{backend} requires (activation_dtype, weight_dtype) in {pairs}; "
            "set --activation-dtype and --weight-dtype explicitly"
        )
    if backend == "cutlass_flashinfer":
        for dimension in ("d_model", "intermediate_size"):
            if config["model"][dimension] % 128:
                raise ValueError(f"cutlass_flashinfer requires model.{dimension} divisible by 128")
        assignments = config["environment"]["num_gpus"] * config["input"]["tokens_per_rank"] * config["model"]["top_k"]
        if assignments + 128 * config["model"]["num_experts"] > 2**31 - 1:
            raise ValueError("cutlass_flashinfer padded assignment schedule must fit in int32")
    experts = config["model"]["num_experts"]
    if experts % config["environment"]["num_gpus"]:
        raise ValueError("model.num_experts must be divisible by environment.num_gpus")
    if config["input"]["tokens_per_rank"] * config["model"]["top_k"] > 2**31 - 1:
        raise ValueError("input.tokens_per_rank * model.top_k must fit in int32")
    route = config["routing"]
    if route["kind"] == "hot_experts":
        if len(route["expert_ids"]) >= experts or any(expert >= experts for expert in route["expert_ids"]):
            raise ValueError("routing.expert_ids must be a strict subset of valid expert IDs [0, num_experts)")


def load_config(
    path: str | Path,
    *,
    num_gpus: int,
    model: str,
    input_name: str,
    routing: str,
    environment_overrides: dict | None = None,
    compute_overrides: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    """Resolve presets using the GPU count detected by the single-node launcher."""
    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load configuration {path}: {error}") from error
    _fields(document, set(_CATEGORIES) | {"environment", "benchmark", "compute"}, "configuration", required=False)
    for category, section in _CATEGORIES.items():
        if category not in document:
            raise ValueError(f"configuration: missing preset category {category!r}")
        _mapping(document[category], category)
        for name, preset in document[category].items():
            _validate_section(section, preset, f"{category}.{name}")
    if "environment" not in document:
        raise ValueError("configuration: missing field 'environment'")
    environment_fields = _FIELDS["environment"] - {"num_gpus"}
    _fields(document["environment"], environment_fields, "environment", required=False)
    _validate_section("environment", {**document["environment"], "num_gpus": num_gpus}, "environment")
    benchmark = document.get("benchmark", {})
    _fields(benchmark, _FIELDS["benchmark"], "benchmark", required=False)
    resolved = {
        "benchmark": {**BENCHMARK_DEFAULTS, **benchmark},
        "environment": {**COMMUNICATION_DEFAULTS, **deepcopy(document["environment"]), "num_gpus": num_gpus},
    }
    compute = document.get("compute", {})
    _fields(compute, set(COMPUTE_DEFAULTS), "compute", required=False)
    resolved["compute"] = {**COMPUTE_DEFAULTS, **compute}
    selections = {"models": model, "inputs": input_name, "routing": routing}
    for category, name in selections.items():
        presets = document[category]
        if not isinstance(name, str) or name not in presets:
            available = ", ".join(sorted(presets)) or "(none)"
            raise ValueError(f"Unknown {category} preset {name!r}; available presets: {available}")
        resolved[_CATEGORIES[category]] = deepcopy(presets[name])
    if environment_overrides is not None:
        _fields(environment_overrides, environment_fields, "environment_overrides", required=False)
        resolved["environment"].update(
            {key: value for key, value in environment_overrides.items() if value is not None}
        )
    if overrides is not None:
        _fields(overrides, _FIELDS["benchmark"], "overrides", required=False)
        resolved["benchmark"].update({key: value for key, value in overrides.items() if value is not None})
    if compute_overrides is not None:
        _fields(compute_overrides, set(COMPUTE_DEFAULTS), "compute_overrides", required=False)
        resolved["compute"].update({key: value for key, value in compute_overrides.items() if value is not None})
    validate_config(resolved)
    return resolved
