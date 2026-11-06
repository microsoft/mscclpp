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
        {"weight_dtype": "fp8"},
        {"weight_dtype": "mxfp8", "activation_dtype": "fp8_static"},
        {"weight_dtype": "mxfp8", "activation_scale": 1.0},
        {"activation_scale": 0.0},
        {"activation_scale": -1.0},
        {"activation_scale": float("nan")},
        {"combine_route_weights": 1},
    ],
)
def test_config_rejects_invalid(override):
    with pytest.raises(ValueError):
        _config(**override)


def test_w8a16_config_and_local_expert():
    cfg = _config(weight_dtype="mxfp8")
    assert cfg.is_w8a16
    assert cfg.weight_dtype == "mxfp8"
    assert cfg.activation_dtype == "bf16"
    cfg_bf16 = _config(activation_dtype="bf16")
    assert cfg_bf16.is_w8a16
    assert cfg_bf16.weight_dtype == "mxfp8"
    local_cfg = _config(
        world_size=1,
        max_tokens=16,
        hidden=128,
        intermediate=128,
        num_experts=1,
        top_k=1,
        weight_dtype="mxfp8",
    )
    assert local_cfg.is_w8a16 and local_cfg.local_experts == 1


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


def test_native_w8a16_local_and_shared_expert_graph():
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available

    torch = pytest.importorskip("torch")
    if (
        not is_available()
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7))
    ):
        pytest.skip("requires native MegaMoE on an SM100-family GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(
        world_size=1,
        max_tokens=33,
        hidden=128,
        intermediate=128,
        num_experts=1,
        top_k=1,
        sm_margin=32,
        weight_dtype="mxfp8",
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    generator = torch.Generator(device=device).manual_seed(1729)
    source1 = torch.randn((1, 256, 128), generator=generator, device=device).mul_(0.03125)
    source2 = torch.randn((1, 128, 128), generator=generator, device=device).mul_(0.03125)
    fc1, scale1 = quantize_mxfp8(source1)
    fc2, scale2 = quantize_mxfp8(source2)
    first = dequantize_mxfp8(fc1, scale1, dtype=torch.bfloat16).float()[0]
    second = dequantize_mxfp8(fc2, scale2, dtype=torch.bfloat16).float()[0]
    context = MegaMoE(config, Communicator(bootstrap), fc1, scale1, fc2, scale2)
    inputs = torch.randn((33, 128), generator=generator, device=device, dtype=torch.bfloat16).mul_(0.125)
    ids = torch.zeros((33, 1), dtype=torch.int32, device=device)
    ids[::5] = -1
    scores = torch.rand((33, 1), generator=generator, device=device, dtype=torch.float32)

    gate, up = (inputs.float() @ first.T).chunk(2, dim=-1)
    shared_hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    expected_shared = (shared_hidden.float() @ second.T).to(torch.bfloat16)
    out_shared = context.forward_shared(inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_shared, expected_shared, rtol=0.02, atol=0.001)

    active = (ids[:, 0] == 0).unsqueeze(-1)
    weighted_hidden = (torch.nn.functional.silu(gate) * up * torch.where(active, scores, 0.0)).to(torch.bfloat16)
    expected_local = torch.where(active, (weighted_hidden.float() @ second.T).to(torch.bfloat16), 0)
    out_local = context(inputs, ids, scores)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_local, expected_local, rtol=0.02, atol=0.001)
    del context


def test_native_w8a16_routed_and_graph():
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available

    torch = pytest.importorskip("torch")
    if (
        not is_available()
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7))
    ):
        pytest.skip("requires native MegaMoE on an SM100-family GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    config = _config(
        world_size=1,
        max_tokens=65,
        hidden=384,
        intermediate=640,
        num_experts=4,
        top_k=2,
        sm_margin=sm_count - 8,
        gate_up_clamp=0.125,
        weight_dtype="mxfp8",
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    generator = torch.Generator(device=device).manual_seed(4189)
    source1 = torch.randn((4, 1280, 384), generator=generator, device=device).mul_(0.03125)
    source2 = torch.randn((4, 384, 640), generator=generator, device=device).mul_(0.03125)
    fc1, sf1 = quantize_mxfp8(source1)
    fc2, sf2 = quantize_mxfp8(source2)
    context = MegaMoE(config, Communicator(bootstrap), fc1, sf1, fc2, sf2)
    assert context.effective_kernel_config["tile_n"] == 128
    assert context.effective_kernel_config["transform_stages"] == 4
    first = dequantize_mxfp8(fc1, sf1, dtype=torch.bfloat16).float()
    second = dequantize_mxfp8(fc2, sf2, dtype=torch.bfloat16).float()
    inputs = torch.randn((65, 384), generator=generator, device=device, dtype=torch.bfloat16).mul_(0.125)
    ids = torch.arange(65, device=device, dtype=torch.int32).remainder(4)
    ids = torch.stack((ids, (ids + 1).remainder(4)), dim=1).contiguous()
    scores = torch.rand((65, 2), generator=generator, device=device)
    scores /= scores.sum(dim=1, keepdim=True)
    output = torch.empty_like(inputs)

    def reference(tokens):
        partial = torch.zeros((tokens, 2, 384), device=device, dtype=torch.float32)
        for expert in range(4):
            rows, slots = torch.where(ids[:tokens] == expert)
            if rows.numel() == 0:
                continue
            gate, up = (inputs[rows].float() @ first[expert].T).chunk(2, dim=-1)
            gate = gate.clamp(max=0.125)
            up = up.clamp(-0.125, 0.125)
            hidden = (torch.nn.functional.silu(gate) * up * scores[rows, slots, None]).to(torch.bfloat16)
            partial[rows, slots] = (hidden.float() @ second[expert].T).to(torch.bfloat16).float()
        return partial.sum(dim=1).to(torch.bfloat16)

    for tokens in (65, 32, 1, 0, 49):
        context(inputs[:tokens], ids[:tokens], scores[:tokens], output=output[:tokens])
        expected = reference(tokens)
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:tokens], expected, rtol=0.02, atol=0.001)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        context(inputs[:49], ids[:49], scores[:49], output=output[:49], stream=stream)
    with torch.cuda.stream(stream):
        inputs.neg_()
        for _ in range(10):
            graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output[:49], reference(49), rtol=0.02, atol=0.001)
    torch.cuda.synchronize()
    del graph, context

    config_combine = _config(
        world_size=1,
        max_tokens=65,
        hidden=384,
        intermediate=640,
        num_experts=4,
        top_k=2,
        sm_margin=sm_count - 8,
        gate_up_clamp=0.125,
        weight_dtype="mxfp8",
        combine_route_weights=True,
    )
    context_combine = MegaMoE(config_combine, Communicator(bootstrap), fc1, sf1, fc2, sf2, tag=18100)

    def reference_combine(tokens):
        partial = torch.zeros((tokens, 2, 384), device=device, dtype=torch.float32)
        for expert in range(4):
            rows, slots = torch.where(ids[:tokens] == expert)
            if rows.numel() == 0:
                continue
            gate, up = (inputs[rows].float() @ first[expert].T).chunk(2, dim=-1)
            gate = gate.clamp(max=0.125)
            up = up.clamp(-0.125, 0.125)
            hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            partial[rows, slots] = (hidden.float() @ second[expert].T).to(torch.bfloat16).float() * scores[
                rows, slots, None
            ]
        return partial.sum(dim=1).to(torch.bfloat16)

    context_combine(inputs[:49], ids[:49], scores[:49], output=output[:49])
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:49], reference_combine(49), rtol=0.02, atol=0.001)
    del context_combine


@pytest.mark.parametrize("activation_dtype,activation_scale", [("mxfp8", None), ("fp8_static", 0.25)])
def test_native_w4a8_combine_route_weights(activation_dtype, activation_scale):
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available
    from mscclpp.ext.megamoe.benchmark import _reference, _weights

    torch = pytest.importorskip("torch")
    if (
        not is_available()
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7))
    ):
        pytest.skip("requires native MegaMoE on an SM100-family GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    config = _config(
        world_size=1,
        max_tokens=33,
        hidden=4096,
        intermediate=5120,
        num_experts=4,
        top_k=2,
        sm_margin=sm_count - 8,
        activation_dtype=activation_dtype,
        activation_scale=activation_scale,
        combine_route_weights=True,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    torch.manual_seed(5179)
    weights = _weights(config, device)
    context = MegaMoE(config, Communicator(bootstrap), *weights)
    inputs = torch.randn((33, 4096), device=device, dtype=torch.bfloat16).mul_(0.125)
    ids = torch.arange(33, device=device, dtype=torch.int32).remainder(4)
    ids = torch.stack((ids, (ids + 1).remainder(4)), dim=1).contiguous()
    ids[::7, 1] = -1
    scores = torch.rand((33, 2), device=device, dtype=torch.float32)
    scores[ids == -1] = float("nan")
    output = context(inputs, ids, scores, validate_routing=False)
    torch.cuda.synchronize()
    expected = _reference(config, inputs, ids, torch.where(ids >= 0, scores, 0.0), weights)
    torch.testing.assert_close(output, expected, rtol=0.05, atol=0.02)
    del context
