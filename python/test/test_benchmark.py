# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mscclpp_benchmark.correctness import (
    _comparison_tolerance,
    _decode_bfloat16_array,
    _decode_fp8_array,
    _decode_fp8_scalar,
    _encode_bfloat16_values,
    _encode_correctness_input,
    _encode_fp8_values,
    _stats_values,
)
from mscclpp_benchmark.tuning_config import HardwareProfile, TunedConfig, TunedConfigStore


def test_allgather_requires_exact_match():
    case = SimpleNamespace(
        collective="allgather",
        dtype_spec=SimpleNamespace(name="float16", fp8_format=None, storage_dtype=np.float16),
    )

    assert _comparison_tolerance(case, 8) is None


def test_bfloat16_round_to_nearest_even():
    values = np.asarray([1.0, -2.5, 1.00390625, 1.01171875], dtype=np.float32)

    encoded = _encode_bfloat16_values(values)

    np.testing.assert_array_equal(encoded, np.asarray([0x3F80, 0xC020, 0x3F80, 0x3F82], dtype=np.uint16))


def test_bfloat16_raw_storage_round_trip():
    case = SimpleNamespace(dtype_spec=SimpleNamespace(name="bfloat16", fp8_format=None, storage_dtype=np.uint16))
    values = np.asarray([-1.0, -0.25, 0.5, 1.0], dtype=np.float32)

    encoded = _encode_correctness_input(case, values)

    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(_decode_bfloat16_array(encoded), values)
    np.testing.assert_array_equal(_stats_values(case, encoded), values)


def test_fp8_numpy_decoders_and_finite_round_trips():
    bits = np.arange(256, dtype=np.uint8)
    for fmt in ("e4m3fn", "e4m3fnuz", "e4m3b15"):
        decoded = _decode_fp8_array(fmt, bits)
        expected = np.asarray([_decode_fp8_scalar(fmt, int(byte)) for byte in bits])
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=0, equal_nan=True)
        finite = decoded[~np.isnan(decoded)]
        encoded = _encode_fp8_values(fmt, finite)
        np.testing.assert_array_equal(_decode_fp8_array(fmt, encoded), finite)


def test_selects_dtype_specific_configs_with_duplicate_sizes():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "sku": "MI300X",
                    "scale": 8,
                    "collectives": {
                        "allreduce": [
                            {
                                "message_size": 1024,
                                "algorithm": "bf16-small",
                                "dtype": "bfloat16",
                                "accum": "bfloat16",
                            },
                            {
                                "message_size": 2048,
                                "algorithm": "bf16-large",
                                "dtype": "bfloat16",
                                "accum": "bfloat16",
                            },
                            {
                                "message_size": 1024,
                                "algorithm": "fp16-small",
                                "dtype": "float16",
                                "accum": "float16",
                            },
                        ]
                    },
                }
            ]
        }
    )
    profile = HardwareProfile("MI300X", 8)

    assert store.select(profile, "allreduce", 1536, dtype="bfloat16", accum="bfloat16").algorithm == "bf16-small"
    assert store.select(profile, "allreduce", 1536, dtype="float16", accum="float16").algorithm == "fp16-small"
    assert store.select(profile, "allreduce", 1536, dtype="float32", accum="float32") is None


def test_generic_config_matches_any_dtype():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "collectives": {
                        "allgather": [
                            {
                                "message_size": 1024,
                                "algorithm": "generic-allgather",
                            }
                        ]
                    }
                }
            ]
        }
    )

    config = store.select(HardwareProfile("MI300X", 8), "allgather", 1024, dtype="bfloat16", accum="bfloat16")

    assert config is not None
    assert config.algorithm == "generic-allgather"


def test_ep_hybrid_cli_options(monkeypatch):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ep-bench",
            "--backend",
            "mscclpp",
            "--ep-layout",
            "rank_major",
            "--qps-per-peer",
            "4",
            "--n-tokens-per-chunk",
            "16",
            "-d",
            "4096",
            "-e",
            "512",
        ],
    )
    args = parse_args()
    assert args.qps_per_peer == 4
    assert args.n_tokens_per_chunk == 16
    assert args.hidden == 4096 and args.num_experts == 512


def test_ep_hybrid_cli_preserves_defaults(monkeypatch):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(sys, "argv", ["ep-bench", "--backend", "mscclpp"])
    args = parse_args()
    assert args.qps_per_peer is None and args.n_tokens_per_chunk is None
    assert args.routing == "random" and args.routing_sigma == 0.85


def test_ep_lognormal_cli(monkeypatch):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(sys, "argv", ["ep-bench", "--routing", "lognormal", "--routing-sigma", "0.85"])
    args = parse_args()
    assert args.routing == "lognormal" and args.routing_sigma == 0.85


@pytest.mark.parametrize("sigma", ["-1", "nan", "inf"])
def test_ep_lognormal_cli_rejects_bad_sigma(monkeypatch, sigma):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(sys, "argv", ["ep-bench", "--routing", "lognormal", f"--routing-sigma={sigma}"])
    with pytest.raises(SystemExit, match="finite and non-negative"):
        parse_args()


@pytest.mark.parametrize("sigma", [0.0, 0.85])
def test_ep_lognormal_popularity_distribution(sigma):
    from mscclpp_benchmark.ep.bench_common import lognormal_expert_probabilities

    generator = torch.Generator(device="cpu").manual_seed(123)
    weights = torch.exp(torch.randn(32, dtype=torch.float64, generator=generator) * sigma)
    actual = lognormal_expert_probabilities(32, sigma, 123)
    torch.testing.assert_close(actual, weights / weights.sum(), rtol=1e-14, atol=1e-14)
    assert torch.equal(actual, lognormal_expert_probabilities(32, sigma, 123))


@pytest.fixture
def ep_cpu_input_allocations(monkeypatch):
    for name in ("ones", "arange", "randn"):
        allocate = getattr(torch, name)

        def cpu_allocate(*args, _allocate=allocate, **kwargs):
            kwargs.pop("device", None)
            return _allocate(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_allocate)


def test_ep_lognormal_inputs_preserve_payloads_and_weights(ep_cpu_input_allocations):
    from mscclpp_benchmark.ep.bench_common import make_inputs, lognormal_expert_probabilities

    default = make_inputs(16, 256, 8, 64, 3, 123)
    random = make_inputs(16, 256, 8, 64, 3, 123, routing="random")
    skewed = make_inputs(16, 256, 8, 64, 3, 123, routing="lognormal")
    repeated = make_inputs(16, 256, 8, 64, 3, 123, routing="lognormal")
    assert all(torch.equal(a, b) for a, b in zip(default[:3], random[:3]))
    assert torch.equal(default[0], skewed[0]) and torch.equal(default[2], skewed[2])
    assert all(torch.equal(a, b) for a, b in zip(skewed[:3], repeated[:3]))
    assert not torch.equal(default[1], skewed[1])
    assert all(len(set(row.tolist())) == 8 for row in skewed[1])
    probabilities = lognormal_expert_probabilities(64, 0.85, 123)
    generator = torch.Generator(device="cpu").manual_seed(127)
    expected = torch.multinomial(probabilities.expand(16, -1), 8, replacement=False, generator=generator)
    assert torch.equal(skewed[1], expected)
    other_rank = make_inputs(16, 256, 8, 64, 4, 123, routing="lognormal")
    assert not torch.equal(skewed[1], other_rank[1])


def test_ep_routing_summary_counts_unique_destination_rows():
    from mscclpp_benchmark.ep.bench_common import routing_load_summary

    sources = [torch.tensor([[0, 1], [0, 2]]), torch.tensor([[2, 3], [-1, 3]])]
    comm = SimpleNamespace(rank=0, gather=lambda value, **kwargs: sources)
    result = routing_load_summary(sources[0], 4, 2, comm)
    assert result["expert_assignments"]["counts"] == [2, 1, 2, 2]
    assert result["rank_assignments"]["counts"] == [3, 4]
    assert result["rank_unique_rows"]["counts"] == [2, 3]
    assert result["source_to_destination_rows"] == [[2, 1], [0, 2]]
    assert result["rank_unique_rows"]["max_over_mean"] == 1.2


def test_ep_nccl_rank_major_fp8_cli(monkeypatch):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["ep-bench", "--backend", "nccl", "--ep-layout", "rank_major", "--dispatch-dtype", "fp8_e4m3"],
    )
    args = parse_args()
    assert args.dispatch_dtype == "fp8_e4m3" and args.ep_layout == "rank_major"


@pytest.mark.parametrize(
    "explicit_recipe,layout,dtype",
    [
        (False, "rank_major", "fp8_e4m3"),
        (True, "rank_major", "fp8_e4m3"),
        (True, "expert_major", "fp8_e4m3"),
        (True, "rank_major", "bf16"),
    ],
)
def test_ep_nccl_quantization_api_compatibility(monkeypatch, explicit_recipe, layout, dtype):
    from mscclpp_benchmark.ep.bench_nccl import setup_nccl

    calls = []
    handle = SimpleNamespace(dispatch=lambda *args, **kwargs: calls.append((args, kwargs)))
    group = SimpleNamespace(create_handle=lambda *args, **kwargs: handle)
    ncomm = SimpleNamespace(n_lsa_teams=2, gin_type=3)
    core = SimpleNamespace(
        get_unique_id=lambda: 0,
        Communicator=SimpleNamespace(init=lambda **kwargs: ncomm),
    )
    ep = SimpleNamespace(
        Algorithm=SimpleNamespace(LOW_LATENCY="ll"),
        Layout=SimpleNamespace(RANK_MAJOR="rank", EXPERT_MAJOR="expert", FLAT="flat"),
        GroupConfig=SimpleNamespace,
        AllocConfig=SimpleNamespace,
        Group=SimpleNamespace(create=lambda *args: group),
        Tensor=lambda tensor: tensor,
        HandleConfig=SimpleNamespace,
        DispatchInputs=SimpleNamespace,
        DispatchOutputs=SimpleNamespace,
        LayoutInfo=SimpleNamespace,
        CombineInputs=SimpleNamespace,
        CombineOutputs=SimpleNamespace,
        CombineConfig=SimpleNamespace,
        get_lib_version=lambda: "0.3.0" if explicit_recipe else "0.1.0",
        get_lib_path=lambda: "/isolated/libnccl_ep.so",
    )
    if explicit_recipe:
        ep.DispatchQuantizationRecipe = SimpleNamespace(DS_FP8E3M4=2)
        ep.DispatchConfig = SimpleNamespace
    else:
        ep.DispatchConfig = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(core=core, ep=ep))
    monkeypatch.setitem(sys.modules, "nccl.core", core)
    monkeypatch.setitem(sys.modules, "nccl.ep", ep)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("device", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    args = SimpleNamespace(
        combine_mode="rank_local_reduce",
        num_tokens=2,
        hidden=9216,
        num_experts=8,
        num_topk=2,
        dispatch_dtype=dtype,
        mode="latency",
        ep_layout=layout,
        num_warmup=0,
        num_iters=1,
        qps_per_peer=4,
        num_sms=0,
        validate=False,
        cuda_graph=False,
    )
    inputs = (
        torch.zeros((2, 9216), dtype=torch.bfloat16),
        torch.zeros((2, 2), dtype=torch.int64),
        torch.ones((2, 2)),
        4,
    )
    backend = setup_nccl(args, SimpleNamespace(bcast=lambda value, **kwargs: value), 0, 4, inputs)
    backend["dispatch"]()
    (_, outputs), kwargs = calls[0]
    if dtype == "bf16":
        assert outputs.scales is None
        assert backend["summary_meta"]["dispatch_quantization_recipe"] == "NONE"
    elif explicit_recipe:
        assert outputs.scales.shape == (*outputs.tokens.shape[:-1], 9216 // 128)
        assert outputs.scales.is_contiguous()
        assert kwargs["config"].quantization_recipe == ep.DispatchQuantizationRecipe.DS_FP8E3M4
    else:
        assert outputs.scales.shape == (2, 8, 9216 // 128)
        assert outputs.scales.stride() == (8 * (9216 // 128), 1, 8)
    assert backend["summary_meta"]["library_version"] == ep.get_lib_version()
    assert backend["summary_meta"]["library_path"] == ep.get_lib_path()


@pytest.mark.parametrize("chunk", ["0", "129"])
def test_ep_hybrid_cli_rejects_bad_chunk(monkeypatch, chunk):
    from mscclpp_benchmark.ep.bench import parse_args

    monkeypatch.setattr(sys, "argv", ["ep-bench", "--n-tokens-per-chunk", chunk])
    with pytest.raises(SystemExit, match=r"\[1, 128\]"):
        parse_args()


def test_ep_rank_major_unit_output_counts_destinations():
    from mscclpp_benchmark.ep.bench_common import rank_major_unit_output

    routes = torch.tensor([[0, 1, 16, -1], [31, 32, 33, 48], [-1, -1, -1, -1]])
    actual = rank_major_unit_output(routes, 16, 8)
    expected = torch.tensor([2, 3, 0], dtype=torch.bfloat16)[:, None].expand(-1, 8)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rank,ranks_per_domain,expected", [(0, 2, 1.0078125), (2, 2, 1.0078125), (0, 4, 1.0)])
def test_ep_direct_send_unit_output_uses_domain_reduction_order(rank, ranks_per_domain, expected):
    import mscclpp.ep as ep
    from mscclpp_benchmark.ep.bench_mscclpp import _expected_rank_major_unit_output

    routes = torch.tensor([[0, 4, 5, -1]])
    weights = torch.tensor([[1.00390625, 2**-24, 2**-24, 1000]], dtype=torch.float32)
    config = SimpleNamespace(
        num_experts=8, num_local_experts=2, hidden_size=8, combine_mode=ep.CombineMode.DIRECT_SEND, quant=None
    )
    actual = _expected_rank_major_unit_output(routes, weights, config, rank, ranks_per_domain)
    assert torch.equal(actual, torch.full((1, 8), expected, dtype=torch.bfloat16))


@pytest.mark.parametrize("scale,combine_scale", [(1.0, 1.0), (2.0, 2.0), ((4.0, 2.0), 2.0), ((4.0, None), 4.0)])
@pytest.mark.parametrize("rank,expected", [(0, 1.0), (2, 1.0078125)])
def test_ep_fp8_combine_unit_output_rounds_remote_domain(scale, combine_scale, rank, expected):
    import mscclpp.ep as ep
    from mscclpp_benchmark.ep.bench_mscclpp import _expected_rank_major_unit_output

    config = SimpleNamespace(
        num_experts=8,
        num_local_experts=2,
        hidden_size=8,
        combine_mode=ep.CombineMode.DIRECT_SEND,
        quant=SimpleNamespace(scale=scale, combine_format=ep.DispatchDataType.FP8_E4M3_STATIC),
    )
    routes = torch.tensor([[0, 4, -1, 8]])
    weights = torch.tensor([[2**-8, 1.00390625, 1000, 1000]], dtype=torch.float32)
    actual = _expected_rank_major_unit_output(routes, weights, config, rank, 2)
    assert torch.equal(actual, torch.full((1, 8), expected / combine_scale, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "scale,combine_scale", [(None, 1.0), (1.0, 1.0), (2.0, 2.0), ((4.0, 2.0), 2.0), ((4.0, None), 4.0)]
)
def test_ep_rank_local_unit_output_counts_unique_peers_with_scale(scale, combine_scale):
    import mscclpp.ep as ep
    from mscclpp_benchmark.ep.bench_mscclpp import _expected_rank_major_unit_output

    config = SimpleNamespace(
        num_experts=8,
        num_local_experts=2,
        hidden_size=8,
        combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
        quant=(
            None if scale is None else SimpleNamespace(scale=scale, combine_format=ep.DispatchDataType.FP8_E4M3_STATIC)
        ),
    )
    routes = torch.tensor([[0, 1, 4, 5, 6, -1]])
    actual = _expected_rank_major_unit_output(routes, torch.full(routes.shape, 1000.0), config, 0, 2)
    assert torch.equal(actual, torch.full((1, 8), 3 / combine_scale, dtype=torch.bfloat16))


@pytest.mark.parametrize("scale", [2.0, (2.0, 4.0), (2.0, None)])
def test_ep_simulated_gemm_uses_dispatch_scale(scale):
    from mscclpp_benchmark.ep.bench_common import simulated_gemm_output

    dispatch = SimpleNamespace(
        tokens=torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn),
        quant=SimpleNamespace(scale=scale),
    )
    actual = simulated_gemm_output(dispatch)
    assert torch.equal(actual, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))


def test_ep_validation_accepts_empty_rows():
    from mpi4py import MPI
    from mscclpp_benchmark.ep.bench_common import validate_combine_output_mpi

    empty = torch.empty((0, 8), dtype=torch.bfloat16)
    assert validate_combine_output_mpi(empty, empty, MPI.COMM_WORLD, exact=True) == 0


def test_upsert_and_write_preserve_dtype_qualifiers(tmp_path):
    store = TunedConfigStore.empty()
    profile = HardwareProfile("MI300X", 8)
    store.upsert(
        profile,
        "allreduce",
        1024,
        TunedConfig("bf16"),
        dtype="bfloat16",
        accum="bfloat16",
    )
    store.upsert(
        profile,
        "allreduce",
        1024,
        TunedConfig("fp16"),
        dtype="float16",
        accum="float16",
    )
    path = tmp_path / "config.json"

    store.write_path(path)
    entries = json.loads(path.read_text())["profiles"][0]["collectives"]["allreduce"]

    assert {(entry["dtype"], entry["accum"], entry["algorithm"]) for entry in entries} == {
        ("bfloat16", "bfloat16", "bf16"),
        ("float16", "float16", "fp16"),
    }


def test_upsert_defaults_accum_to_dtype():
    store = TunedConfigStore.empty()
    profile = HardwareProfile("MI300X", 8)

    store.upsert(profile, "allreduce", 1024, TunedConfig("fp16"), dtype="float16")

    config = store.select(profile, "allreduce", 1024, dtype="float16")
    assert config is not None
    assert config.algorithm == "fp16"


def test_load_defaults_accum_to_dtype():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "collectives": {
                        "allreduce": [
                            {
                                "message_size": 1024,
                                "algorithm": "fp16",
                                "dtype": "float16",
                            }
                        ]
                    }
                }
            ]
        }
    )

    config = store.select(HardwareProfile(), "allreduce", 1024, dtype="float16")
    assert config is not None
    assert config.algorithm == "fp16"
