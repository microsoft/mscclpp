# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strict named-preset configuration for the MoE benchmark."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

# Common workload dimensions, constrained by MSCCL++ expert-major combine.
SUPPORTED_D_MODEL = (4096, 4352, 5120, 6656, 7168, 8192, 8704, 9216)
BENCHMARK_DEFAULTS = {"warmup": 10, "iterations": 100, "seed": 42, "output_dir": "benchmark_results"}
COMPUTE_DEFAULTS = {"activation_dtype": "mxfp8", "weight_dtype": "mxfp4"}
COMMUNICATION_DEFAULTS = {"mode": "latency", "layout": "expert_major"}
SUPPORTED_LAYOUTS = {
    "mscclpp": {"latency": ("expert_major", "rank_major"), "throughput": ("token_major", "rank_major")},
    "nccl": {"latency": ("expert_major", "rank_major"), "throughput": ("expert_major", "token_major")},
}
MSCCLPP_LATENCY_HIDDEN_SIZES = (1024, 2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216)
_FIELDS = {
    "model": {"d_model", "num_experts", "top_k", "intermediate_size", "activation"},
    "input": {"tokens_per_rank", "dtype"},
    "environment": {"num_gpus", "communication_backend", "compute_backend", "cuda_graph", "mode", "layout"},
    "routing": {"kind"},
    "benchmark": set(BENCHMARK_DEFAULTS),
}
_CATEGORIES = {"models": "model", "inputs": "input", "routing": "routing"}
ROUTING_KINDS = (
    "balanced",
    "uniform",
    "hot_experts",
    "zipf",
    "zipf_skewed",
    "lognormal",
    "hot_rank",
    "idle_ranks",
    "locality",
)
_ROUTING_OPTIONS = {
    "hot_experts": {"expert_ids", "probability_mass"},
    "zipf": {"alpha"},
    "zipf_skewed": {"alpha"},
    "lognormal": {"sigma"},
    "hot_rank": {"rank", "probability_mass"},
    "idle_ranks": {"stride"},
    "locality": {"scope", "local_fraction", "gpus_per_node"},
}


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
    if section == "routing":
        _mapping(value, path)
        if "kind" not in value:
            raise ValueError(f"{path}: missing fields: kind")
        _choice(value["kind"], ROUTING_KINDS, f"{path}.kind")
        _fields(value, fields | _ROUTING_OPTIONS.get(value["kind"], set()), path, required=False)
    else:
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
        _integer(value["num_gpus"], f"{path}.num_gpus")
        _choice(value["communication_backend"], ("mscclpp", "nccl"), f"{path}.communication_backend")
        _choice(value["compute_backend"], ("cutlass_flashinfer",), f"{path}.compute_backend")
        _choice(value["cuda_graph"], (False, "mlp", "layer"), f"{path}.cuda_graph")
        _choice(value["mode"], ("latency", "throughput"), f"{path}.mode")
        _choice(value["layout"], ("expert_major", "rank_major", "token_major"), f"{path}.layout")
    elif section == "routing":
        for option in ("alpha", "sigma"):
            if option in value:
                number = value[option]
                if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
                    raise ValueError(f"{path}.{option} must be a finite number >= 0")
        if value["kind"] == "hot_rank":
            if "rank" in value and type(value["rank"]) is not int:
                raise ValueError(f"{path}.rank must be an integer (negative counts from the last rank)")
            probability = value.get("probability_mass", 0.8)
            if type(probability) not in (float, int) or not 0 < probability < 1:
                raise ValueError(f"{path}.probability_mass must be a number strictly between 0 and 1")
        if value["kind"] == "idle_ranks" and "stride" in value:
            _integer(value["stride"], f"{path}.stride", 2)
        if value["kind"] == "locality":
            for option in ("scope", "local_fraction"):
                if option not in value:
                    raise ValueError(f"{path}: missing fields: {option}")
            _choice(value["scope"], ("rank", "node"), f"{path}.scope")
            fraction = value["local_fraction"]
            if type(fraction) not in (float, int) or not 0 <= fraction <= 1:
                raise ValueError(f"{path}.local_fraction must be a number in [0, 1]")
            if "gpus_per_node" in value:
                if value["scope"] != "node":
                    raise ValueError(f"{path}.gpus_per_node applies only to scope=node")
                _integer(value["gpus_per_node"], f"{path}.gpus_per_node")
        explicit = {"expert_ids", "probability_mass"} & set(value)
        if value["kind"] == "hot_experts" and explicit and len(explicit) != 2:
            raise ValueError(f"{path}: explicit hot_experts requires both expert_ids and probability_mass")
        if value["kind"] == "hot_experts" and explicit:
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
    if communication == "nccl" and mode == "throughput" and environment["cuda_graph"] == "layer":
        raise ValueError("NCCL-EP throughput does not support cuda_graph=layer; use latency mode or cuda_graph=mlp")
    if communication == "nccl" and mode == "throughput" and config["input"]["tokens_per_rank"] % 2:
        raise ValueError("NCCL-EP throughput benchmark requires even tokens_per_rank for native S2D transfer alignment")
    if communication == "mscclpp":
        if config["model"]["top_k"] > 8:
            raise ValueError("MSCCL++ EP supports top_k <= 8")
        if mode == "latency":
            _choice(config["model"]["d_model"], MSCCLPP_LATENCY_HIDDEN_SIZES, "model.d_model for MSCCL++ latency")
        elif config["model"]["num_experts"] // environment["num_gpus"] > 128:
            raise ValueError("MSCCL++ throughput supports at most 128 local experts")
    compute = config.get("compute", COMPUTE_DEFAULTS)
    _fields(compute, set(COMPUTE_DEFAULTS), "compute", required=False)
    _choice(compute.get("activation_dtype", "mxfp8"), ("mxfp8",), "compute.activation_dtype")
    _choice(compute.get("weight_dtype", "mxfp4"), ("mxfp4",), "compute.weight_dtype")
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
    world = config["environment"]["num_gpus"]
    local_experts = experts // world
    top_k = config["model"]["top_k"]
    if route["kind"] == "hot_experts" and "expert_ids" in route:
        if len(route["expert_ids"]) >= experts or any(expert >= experts for expert in route["expert_ids"]):
            raise ValueError("routing.expert_ids must be a strict subset of valid expert IDs [0, num_experts)")
    if route["kind"] == "hot_rank":
        if world < 2:
            raise ValueError("hot_rank routing needs at least two GPUs")
        if not -world <= route.get("rank", -1) < world:
            raise ValueError(f"routing.rank must be in [-{world}, {world})")
    if route["kind"] == "idle_ranks":
        stride = route.get("stride", 2)
        if world < 2:
            raise ValueError("idle_ranks routing needs at least two GPUs")
        if -(-world // stride) * local_experts < top_k:
            raise ValueError("idle_ranks leaves fewer receiving experts than model.top_k")
    if route["kind"] == "locality":
        group = 1 if route["scope"] == "rank" else route.get("gpus_per_node")
        if group is not None and world % group:
            raise ValueError("routing.gpus_per_node must divide environment.num_gpus")
        if group is not None and route["local_fraction"] < 1 and group >= world:
            scope = "rank" if route["scope"] == "rank" else "node"
            raise ValueError(f"locality routing with local_fraction < 1 needs more than one {scope}")


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
    """Resolve presets using the total GPU count detected by the launcher."""
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
