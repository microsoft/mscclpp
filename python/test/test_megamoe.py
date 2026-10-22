# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side W4A8 MegaMoE API and quantization tests."""

import pytest

from mscclpp.ext.megamoe import (
    MegaMoEConfig,
    W4A8KernelConfig,
    dequantize_fp8_static,
    dequantize_mxfp4,
    dequantize_mxfp8,
    quantize_fp8_static,
    quantize_mxfp4,
    quantize_mxfp8,
)


def _config(**overrides):
    fields = dict(rank=0, world_size=4, max_tokens=32, hidden=4096, intermediate=4352, num_experts=64, top_k=7)
    fields.update(overrides)
    return MegaMoEConfig(**fields)


def test_config_dimensions_and_policy():
    config = _config(sm_margin=32)
    assert config.local_experts == 16
    assert config.intermediate == 4352
    assert config.sm_margin == 32
    assert W4A8KernelConfig().cluster_size == 2


def test_config_preserves_positional_gate_clamp():
    config = MegaMoEConfig(0, 4, 32, 4096, 4352, 64, 7, 32, 0.125)
    assert config.gate_up_clamp == 0.125


@pytest.mark.parametrize(
    "override",
    [
        {"rank": 4},
        {"world_size": 0},
        {"world_size": 73},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"hidden": 3968},
        {"hidden": 4097},
        {"intermediate": 0},
        {"num_experts": 63},
        {"top_k": 0},
        {"top_k": 33},
        {"sm_margin": -1},
        {"world_size": 1, "num_experts": 1, "top_k": 1},
        {"gate_up_clamp": float("nan")},
        {"gate_up_clamp": float("inf")},
        {"max_tokens": 2**30},
        {"hidden": 2**31},
        {"activation_dtype": "bf16"},
        {"activation_scale": 0.0},
        {"activation_scale": -1.0},
        {"activation_scale": float("nan")},
    ],
)
def test_config_rejects_invalid(override):
    with pytest.raises(ValueError):
        _config(**override)


def test_static_fp8_config_and_roundtrip():
    torch = pytest.importorskip("torch")
    cfg = _config(activation_scale=0.125)
    assert cfg.is_static_fp8
    assert cfg.activation_dtype == "fp8_static"
    assert cfg.activation_scale == 0.125
    cfg_default_scales = _config(activation_dtype="fp8_static")
    assert cfg_default_scales.is_static_fp8
    assert cfg_default_scales.activation_scale == 1.0
    source = torch.linspace(-2.0, 2.0, 64, dtype=torch.float32).reshape(2, 32)
    q = quantize_fp8_static(source, 0.05)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (2, 32)
    restored = dequantize_fp8_static(q, 0.05, dtype=torch.float32)
    torch.testing.assert_close(restored, source, atol=0.1, rtol=0.1)


def test_mxfp8_activation_roundtrip():
    torch = pytest.importorskip("torch")
    source = torch.arange(-64, 64, dtype=torch.float32).reshape(2, 64) / 64
    values, scales = quantize_mxfp8(source)
    assert values.shape == (2, 64)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 2) and scales.dtype == torch.uint8
    restored = dequantize_mxfp8(values, scales, dtype=torch.float32)
    torch.testing.assert_close(restored, source, atol=0.07, rtol=0.13)


def test_mxfp8_zeros_and_scale_layout():
    torch = pytest.importorskip("torch")
    values, scales = quantize_mxfp8(torch.zeros(2, 128, 32))
    assert torch.all(scales == 127)
    assert torch.count_nonzero(dequantize_mxfp8(values, scales)) == 0
    with pytest.raises(ValueError, match="scales"):
        dequantize_mxfp8(values, scales.squeeze(-1))
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp8(torch.ones(2, 31))
    with pytest.raises(ValueError, match="finite"):
        quantize_mxfp8(torch.full((2, 32), float("nan")))


def test_mxfp4_canonical_rounding_and_layout():
    torch = pytest.importorskip("torch")
    source = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0] * 2,
        dtype=torch.float32,
    ).reshape(1, 32)
    expected = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, 6.0] * 2,
        dtype=torch.float32,
    ).reshape(1, 32)
    values, scales = quantize_mxfp4(source)
    assert values.shape == (1, 16) and values.dtype == torch.uint8
    assert scales.shape == (1, 1) and scales.item() == 127
    torch.testing.assert_close(dequantize_mxfp4(values, scales, dtype=torch.float32), expected, rtol=0, atol=0)
    zeros, zero_scales = quantize_mxfp4(torch.zeros(2, 64))
    assert torch.count_nonzero(zeros) == 0 and torch.all(zero_scales == 127)
    with pytest.raises(ValueError, match="scales"):
        dequantize_mxfp4(values, scales.expand(1, 2))
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp4(torch.ones(2, 31))


@pytest.mark.parametrize("pattern", ["balanced", "zipf", "zipf_skewed", "hot_experts", "lognormal"])
def test_benchmark_unbalanced_routing_generation(pattern):
    torch = pytest.importorskip("torch")
    from mscclpp.ext.megamoe.benchmark import _build_global_expert_counts, _generate_routing

    config = MegaMoEConfig(
        rank=0, world_size=4, max_tokens=64, hidden=4096, intermediate=6656, num_experts=128, top_k=8
    )
    ids, scores = _generate_routing(config, pattern, 1.2, 1234, torch.device("cpu"))
    assert ids.shape == (64, 8) and ids.dtype == torch.int32
    assert scores.shape == (64, 8) and scores.dtype == torch.float32
    assert bool(((ids >= 0) & (ids < 128)).all())
    assert torch.allclose(scores.sum(dim=-1), torch.ones(64))
    if pattern != "balanced":
        counts = _build_global_expert_counts(config, pattern, 1.2, 1234)
        assert len(counts) == 128 and sum(counts) == 4 * 64 * 8
        assert max(counts) > min(counts)
