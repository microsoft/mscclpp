# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strict named-preset configuration for the MoE benchmark."""

from copy import deepcopy
from pathlib import Path

import yaml

# Common workload dimensions, constrained by MSCCL++ expert-major combine.
SUPPORTED_D_MODEL = (4096, 4352, 5120, 6656, 7168, 8192, 8704, 9216)
BENCHMARK_DEFAULTS = {"warmup": 10, "iterations": 100, "seed": 42, "output_dir": "benchmark_results"}
_FIELDS = {
    "model": {"d_model", "num_experts", "top_k", "intermediate_size", "activation"},
    "input": {"tokens_per_rank", "dtype"},
    "environment": {"num_gpus", "communication_backend", "compute_backend", "cuda_graph"},
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
        _choice(value["compute_backend"], ("torch",), f"{path}.compute_backend")
        _choice(value["cuda_graph"], (False, "mlp"), f"{path}.cuda_graph")
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
    _fields(config, set(_FIELDS), "config")
    for section, value in config.items():
        _validate_section(section, value, section)
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
    model: str,
    input_name: str,
    routing: str,
    environment_overrides: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    """Resolve named presets and environment defaults with separate overrides."""
    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load configuration {path}: {error}") from error
    _fields(document, set(_CATEGORIES) | {"environment", "benchmark"}, "configuration", required=False)
    for category, section in _CATEGORIES.items():
        if category not in document:
            raise ValueError(f"configuration: missing preset category {category!r}")
        _mapping(document[category], category)
        for name, preset in document[category].items():
            _validate_section(section, preset, f"{category}.{name}")
    if "environment" not in document:
        raise ValueError("configuration: missing field 'environment'")
    _validate_section("environment", document["environment"], "environment")
    benchmark = document.get("benchmark", {})
    _fields(benchmark, _FIELDS["benchmark"], "benchmark", required=False)
    resolved = {
        "benchmark": {**BENCHMARK_DEFAULTS, **benchmark},
        "environment": deepcopy(document["environment"]),
    }
    selections = {"models": model, "inputs": input_name, "routing": routing}
    for category, name in selections.items():
        presets = document[category]
        if not isinstance(name, str) or name not in presets:
            available = ", ".join(sorted(presets)) or "(none)"
            raise ValueError(f"Unknown {category} preset {name!r}; available presets: {available}")
        resolved[_CATEGORIES[category]] = deepcopy(presets[name])
    if environment_overrides is not None:
        _fields(environment_overrides, _FIELDS["environment"], "environment_overrides", required=False)
        resolved["environment"].update(
            {key: value for key, value in environment_overrides.items() if value is not None}
        )
    if overrides is not None:
        _fields(overrides, _FIELDS["benchmark"], "overrides", required=False)
        resolved["benchmark"].update({key: value for key, value in overrides.items() if value is not None})
    validate_config(resolved)
    return resolved
