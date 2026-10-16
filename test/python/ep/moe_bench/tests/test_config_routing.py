# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only coverage of preset validation and reproducible routing."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import torch
import yaml

from ..config import BENCHMARK_DEFAULTS, SUPPORTED_D_MODEL, load_config, validate_config
from ..routing import make_routing

PRESETS = Path(__file__).resolve().parents[1] / "configs.yaml"


def _config(**kwargs):
    return load_config(
        PRESETS,
        num_gpus=kwargs.get("num_gpus", 4),
        model=kwargs.get("model", "small"),
        input_name=kwargs.get("input_name", "decode"),
        environment_overrides=kwargs.get("environment_overrides"),
        routing=kwargs.get("routing", "balanced"),
        overrides=kwargs.get("overrides"),
    )


def _load_document(document, **kwargs):
    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(document))):
        return _config(**kwargs)


def test_presets_and_defaults():
    config = _config()
    assert set(config) == {"model", "input", "environment", "routing", "benchmark", "compute"}
    assert config["benchmark"] == BENCHMARK_DEFAULTS
    assert config["model"]["d_model"] == 4096
    assert config["environment"] == {
        "mode": "latency",
        "layout": "expert_major",
        "num_gpus": 4,
        "communication_backend": "mscclpp",
        "compute_backend": "torch",
        "cuda_graph": False,
    }
    large = _config(model="large", input_name="prefill", routing="hot_pair")
    assert large["model"]["num_experts"] == 128
    assert large["input"]["tokens_per_rank"] == 1024
    assert large["environment"]["num_gpus"] == 4
    document = yaml.safe_load(PRESETS.read_text())
    del document["benchmark"]
    assert _load_document(document)["benchmark"] == BENCHMARK_DEFAULTS


def test_overrides_and_isolation():
    config = _config(overrides={"warmup": 0, "iterations": 3, "seed": None, "output_dir": "output"})
    assert config["benchmark"] == {"warmup": 0, "iterations": 3, "seed": 42, "output_dir": "output"}
    config["model"]["top_k"] = 1
    assert _config()["model"]["top_k"] == 2
    with pytest.raises(ValueError, match="unknown fields"):
        _config(overrides={"top_k": 1})
    with pytest.raises(ValueError, match="unknown fields"):
        _config(overrides={"validate": None})
    with pytest.raises(ValueError, match="unknown fields"):
        _config(overrides={"num_gpus": 2})
    with pytest.raises(ValueError, match="unknown fields"):
        _config(environment_overrides={"warmup": 0})
    with pytest.raises(ValueError, match="unknown fields"):
        _config(environment_overrides={"graph": None})


def test_mlp_graph_is_explicit_and_opt_in():
    assert _config()["environment"]["cuda_graph"] is False
    graph = _config(environment_overrides={"cuda_graph": "mlp"})
    assert graph["environment"]["cuda_graph"] == "mlp"
    assert graph["environment"]["num_gpus"] == 4


@pytest.mark.parametrize("backend", ["mscclpp", "nccl"])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("graph", [False, "mlp"])
def test_environment_combinations(backend, world, graph):
    config = _config(num_gpus=world, environment_overrides={"communication_backend": backend, "cuda_graph": graph})
    assert config["environment"] == {
        "mode": "latency",
        "layout": "expert_major",
        "num_gpus": world,
        "communication_backend": backend,
        "compute_backend": "torch",
        "cuda_graph": graph,
    }


@pytest.mark.parametrize("key,value", [("communication_backend", "nccl"), ("cuda_graph", "mlp")])
def test_independent_environment_overrides_and_isolation(key, value):
    document = yaml.safe_load(PRESETS.read_text())
    original = deepcopy(document)
    overrides = {key: value}
    with patch("yaml.safe_load", return_value=document):
        config = _config(environment_overrides=overrides)
    assert config["environment"] == {**original["environment"], "num_gpus": 4, key: value}
    config["environment"]["num_gpus"] = 2
    config["routing"]["kind"] = "uniform"
    assert document == original
    assert overrides == {key: value}


def test_graph_default_can_be_disabled():
    document = yaml.safe_load(PRESETS.read_text())
    document["environment"]["cuda_graph"] = "mlp"
    assert _load_document(document, environment_overrides={"cuda_graph": None})["environment"]["cuda_graph"] == "mlp"
    assert _load_document(document, environment_overrides={"cuda_graph": False})["environment"]["cuda_graph"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        {"communication_backend": "unknown"},
        {"num_gpus": 8},
        {"num_gpus": 2},
        {"num_gpus": True},
        {"num_gpus": "4"},
        {"cuda_graph": True},
        {"cuda_graph": 0},
        {"cuda_graph": "eager"},
        {"cuda_graph": "full"},
    ],
)
def test_invalid_environment_overrides(overrides):
    with pytest.raises(ValueError):
        _config(environment_overrides=overrides)


@pytest.mark.parametrize("environment", [None, [], "mscclpp", {1: 4}, {"local": {}}])
def test_malformed_environment(environment):
    document = yaml.safe_load(PRESETS.read_text())
    document["environment"] = environment
    with pytest.raises(ValueError, match="environment"):
        _load_document(document)


@pytest.mark.parametrize("field", ["communication_backend", "compute_backend", "cuda_graph"])
def test_missing_environment_fields(field):
    document = yaml.safe_load(PRESETS.read_text())
    value = document["environment"].pop(field)
    with pytest.raises(ValueError, match=f"environment: missing fields: {field}"):
        _load_document(document, environment_overrides={field: value})


def test_missing_environment_and_legacy_schema():
    document = yaml.safe_load(PRESETS.read_text())
    environment = document.pop("environment")
    with pytest.raises(ValueError, match="missing field 'environment'"):
        _load_document(document)
    document["environments"] = {"local": environment}
    with pytest.raises(ValueError, match="unknown fields: environments"):
        _load_document(document)


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_gpu_count_is_resolved_from_detection_not_yaml(num_gpus):
    config = _config(num_gpus=num_gpus)
    assert config["environment"]["num_gpus"] == num_gpus
    document = yaml.safe_load(PRESETS.read_text())
    document["environment"]["num_gpus"] = num_gpus
    with pytest.raises(ValueError, match="environment: unknown fields: num_gpus"):
        _load_document(document, num_gpus=num_gpus)


@pytest.mark.parametrize("num_gpus", [0, 1, 3, 8, True, "4"])
def test_invalid_detected_gpu_count(num_gpus):
    with pytest.raises(ValueError, match="environment.num_gpus"):
        _config(num_gpus=num_gpus)


@pytest.mark.parametrize("argument", ["model", "input_name", "routing"])
def test_unknown_preset(argument):
    with pytest.raises(ValueError, match="Unknown .* preset.*absent.*available"):
        _config(**{argument: "absent"})


@pytest.mark.parametrize("category", ["models", "inputs", "routing"])
def test_missing_category(category):
    document = yaml.safe_load(PRESETS.read_text())
    del document[category]
    with pytest.raises(ValueError, match=f"missing preset category '{category}'"):
        _load_document(document)


def test_yaml_safety_and_unknown_fields():
    with patch("builtins.open", mock_open(read_data="!!python/object/apply:os.system ['echo unsafe']")):
        with pytest.raises(ValueError, match="Cannot load configuration"):
            _config()
    for document in (None, [], {"unknown": 1}):
        with pytest.raises(ValueError):
            _load_document(document)
    document = yaml.safe_load(PRESETS.read_text())
    document["models"]["large"]["typo"] = True
    with pytest.raises(ValueError, match="models.large: unknown fields: typo"):
        _load_document(document)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "d_model", 2048),
        ("model", "d_model", 2560),
        ("model", "d_model", 3072),
        ("model", "d_model", 6144),
        ("model", "d_model", 4096.0),
        ("model", "num_experts", True),
        ("model", "num_experts", 7),
        ("model", "num_experts", 0),
        ("model", "top_k", 9),
        ("model", "top_k", False),
        ("model", "intermediate_size", -1),
        ("model", "activation", "gelu"),
        ("input", "tokens_per_rank", 0),
        ("input", "tokens_per_rank", "16"),
        ("input", "dtype", "fp16"),
        ("environment", "num_gpus", True),
        ("environment", "num_gpus", 8),
        ("environment", "communication_backend", "absent"),
        ("environment", "compute_backend", "triton"),
        ("environment", "cuda_graph", True),
        ("environment", "cuda_graph", 0),
        ("routing", "kind", "unknown"),
        ("benchmark", "warmup", -1),
        ("benchmark", "iterations", 0),
        ("benchmark", "iterations", True),
        ("benchmark", "seed", -1),
        ("benchmark", "seed", 2**63),
        ("benchmark", "seed", 1.5),
        ("benchmark", "output_dir", ""),
        ("benchmark", "output_dir", None),
    ],
)
def test_invalid_values(section, key, value):
    config = _config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("section", ["model", "input", "environment", "routing", "benchmark"])
def test_missing_and_unknown_resolved_fields(section):
    config = _config()
    del config[section]["num_gpus" if section == "environment" else next(iter(config[section]))]
    with pytest.raises(ValueError, match="missing fields"):
        validate_config(config)
    config = _config()
    config[section]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        validate_config(config)


@pytest.mark.parametrize("name", ["communication_mode", "validate", "profile"])
def test_no_extra_controls(name):
    config = _config()
    config[name] = False
    with pytest.raises(ValueError, match="unknown fields"):
        validate_config(config)


@pytest.mark.parametrize("dimension", SUPPORTED_D_MODEL)
def test_supported_dimensions(dimension):
    config = _config()
    config["model"]["d_model"] = dimension
    validate_config(config)


def test_top_k_limit_and_row_limit():
    config = _config(model="large")
    config["model"]["top_k"] = 9
    validate_config(config)
    config["model"]["top_k"] = 10
    with pytest.raises(ValueError, match="EP combine limit"):
        validate_config(config)
    config["model"]["top_k"] = 8
    config["input"]["tokens_per_rank"] = 2**30
    with pytest.raises(ValueError, match="fit in int32"):
        validate_config(config)


@pytest.mark.parametrize("ids", [[], [0, 0], [False], [-1], [8], list(range(8)), "0,1"])
def test_invalid_hot_experts(ids):
    config = _config(routing="hot_pair")
    config["routing"]["expert_ids"] = ids
    with pytest.raises(ValueError, match="expert_ids"):
        validate_config(config)


@pytest.mark.parametrize("mass", [0, 1, -0.1, 1.1, True, "0.8", float("nan"), float("inf")])
def test_invalid_hot_probability(mass):
    config = _config(routing="hot_pair")
    config["routing"]["probability_mass"] = mass
    with pytest.raises(ValueError, match="probability_mass"):
        validate_config(config)


@pytest.mark.parametrize("route", ["balanced", "uniform", "hot_pair"])
def test_reproducibility_shapes_and_rng_isolation(route):
    config = _config(routing=route)
    state = torch.random.get_rng_state().clone()
    ids, weights = make_routing(config, 1)
    repeated_ids, repeated_weights = make_routing(config, 1)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(ids, repeated_ids)
    assert torch.equal(weights, repeated_weights)
    assert ids.shape == weights.shape == (16, 2)
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32
    assert ids.device.type == weights.device.type == "cpu"
    assert torch.all((ids >= 0) & (ids < 8))
    assert torch.all(ids.sort(dim=1).values[:, 1:] != ids.sort(dim=1).values[:, :-1])
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(16))
    assert not torch.equal(weights, make_routing(config, 0)[1])


@pytest.mark.parametrize("tokens,experts,top_k,ranks", [(3, 8, 3, 2), (5, 12, 2, 4), (1, 8, 1, 2), (3, 8, 8, 4)])
def test_global_balanced_counts(tokens, experts, top_k, ranks):
    config = _config()
    config["input"]["tokens_per_rank"] = tokens
    config["model"].update(num_experts=experts, top_k=top_k)
    config["environment"]["num_gpus"] = ranks
    ids = torch.cat([make_routing(config, rank)[0] for rank in range(ranks)])
    counts = torch.bincount(ids.flatten(), minlength=experts)
    assert int(counts.max() - counts.min()) <= 1
    assert torch.all(ids.sort(dim=1).values[:, 1:] != ids.sort(dim=1).values[:, :-1])


def test_weights_independent_of_routing():
    weights = [make_routing(_config(routing=route), 0)[1] for route in ("balanced", "uniform", "hot_pair")]
    assert all(torch.equal(weights[0], value) for value in weights[1:])


def test_uniform_and_hotspot_distribution():
    config = _config(routing="uniform")
    config["input"]["tokens_per_rank"] = 4096
    config["model"]["top_k"] = 1
    uniform_ids, _ = make_routing(config, 0)
    uniform_counts = torch.bincount(uniform_ids.flatten(), minlength=8)
    assert torch.all((uniform_counts > 350) & (uniform_counts < 700))
    config["routing"] = deepcopy(_config(routing="hot_pair")["routing"])
    hot_ids, _ = make_routing(config, 0)
    hot_fraction = (hot_ids < 2).float().mean().item()
    assert 0.7 < hot_fraction < 0.9


@pytest.mark.parametrize("rank", [-1, 4, True, 0.5])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        make_routing(_config(), rank)
