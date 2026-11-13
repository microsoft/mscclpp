# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side W4A8 MegaMoE API and quantization tests."""

from contextlib import nullcontext
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import pytest

from mscclpp.ext.megamoe import (
    MegaMoEConfig,
    SharedExpertConfig,
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


@pytest.fixture
def mocked_workspace_api(monkeypatch):
    import mscclpp
    from mscclpp.ext.megamoe import api

    torch = pytest.importorskip("torch")
    calls = []

    def create(*args):
        calls.append(args)
        owner = args[-1]
        return SimpleNamespace(
            workspace_id=owner.workspace_id if owner is not None else len(calls) * 4096,
            packed_weight_pointers=tuple(len(calls) * 256 + i * 32 for i in range(4)),
            symmetric_bytes=8192,
            private_bytes=4096,
        )

    fused_calls = []

    def create_fused_shared(*args):
        fused_calls.append(args)
        owner = args[13]
        return SimpleNamespace(
            workspace_id=owner.workspace_id if owner is not None else (len(calls) + len(fused_calls)) * 4096,
            packed_weight_pointers=tuple((len(calls) + len(fused_calls)) * 256 + i * 32 for i in range(4)),
            symmetric_bytes=8192,
            private_bytes=6144,
        )

    native = SimpleNamespace(
        CppMegaMoeConfig=SimpleNamespace,
        CppMegaMoeSharedExpertConfig=SimpleNamespace,
        CppMegaMoeContext=SimpleNamespace(create=create, create_fused_shared=create_fused_shared),
        CppMegaMoeWeightDtype=SimpleNamespace(Mxfp4=0, Mxfp8=1, Fp8Nv=2),
        CppMegaMoeActivationDtype=SimpleNamespace(Mxfp8=0, Fp8Static=1, Bf16=2),
    )
    monkeypatch.setattr(mscclpp, "_mscclpp", native, raising=False)
    monkeypatch.setattr(api, "is_available", lambda: True)
    monkeypatch.setattr(api, "_tensor", lambda *args: None)
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(api.MegaMoE, "_stream", lambda self, requested: stream)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(multi_processor_count=152))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    weights = [SimpleNamespace(data_ptr=lambda pointer=pointer: pointer) for pointer in range(1, 5)]
    native.fused_calls = fused_calls
    return api, native, calls, weights


def test_shared_workspace_constructor_and_readonly_identity(mocked_workspace_api):
    api, _, calls, weights = mocked_workspace_api
    communicator = object()
    owner = api.MegaMoE(_config(), communicator, *weights)
    alias = api.MegaMoE(_config(), communicator, *weights, shared_workspace=owner)
    independent = api.MegaMoE(_config(), communicator, *weights)
    chained = api.MegaMoE(_config(), communicator, *weights, shared_workspace=alias)
    assert calls[0][-1] is None and calls[2][-1] is None
    assert calls[1][-1] is owner._native and calls[3][-1] is alias._native
    assert calls[1][0] is communicator
    assert owner.workspace_id == alias.workspace_id == chained.workspace_id != independent.workspace_id
    assert owner.workspace_bytes == alias.workspace_bytes == {"symmetric": 8192, "private": 4096}
    pointers = [p for context in (owner, alias, independent, chained) for p in context.packed_weight_pointers]
    assert len(set(pointers)) == 16
    with pytest.raises(AttributeError):
        alias.workspace_id = 0
    with pytest.raises(AttributeError):
        alias.packed_weight_pointers = ()
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None
    assert chained.workspace_id == alias.workspace_id


def test_shared_workspace_type_validation(mocked_workspace_api):
    api, _, calls, weights = mocked_workspace_api
    with pytest.raises(TypeError, match="shared_workspace must be a MegaMoE or None"):
        api.MegaMoE(_config(), object(), *weights, shared_workspace=object())
    assert not calls


def test_shared_workspace_native_validation_is_not_silenced(mocked_workspace_api, monkeypatch):
    api, native, _, weights = mocked_workspace_api
    owner = api.MegaMoE(_config(), object(), *weights)

    def reject(*args):
        assert args[-1] is owner._native
        raise ValueError("MegaMoE shared_workspace must use the same native configuration")

    monkeypatch.setattr(native.CppMegaMoeContext, "create", reject)
    with pytest.raises(ValueError, match="same native configuration"):
        api.MegaMoE(_config(max_tokens=64), object(), *weights, shared_workspace=owner)


def test_shared_expert_config_validation():
    shared = SharedExpertConfig(2048, 0.25, hidden=8704)
    assert shared.intermediate == 2048 and shared.hidden == 8704
    for intermediate in (0, 129, True, 2**31):
        with pytest.raises(ValueError):
            SharedExpertConfig(intermediate)
    for hidden in (0, 129, True, 2**31):
        with pytest.raises(ValueError):
            SharedExpertConfig(2048, hidden=hidden)
    for clamp in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SharedExpertConfig(2048, clamp)


def test_fused_shared_constructor_contract(mocked_workspace_api):
    api, native, _, weights = mocked_workspace_api
    config = _config(activation_scale=0.125)
    shared = api.SharedExpertConfig(2048, 0.5, hidden=8704)
    with pytest.raises(ValueError, match="supplied together"):
        api.MegaMoE(config, object(), *weights, shared_config=shared)
    with pytest.raises(ValueError, match="requires routed"):
        api.MegaMoE(
            _config(),
            object(),
            *weights,
            shared_config=shared,
            shared_fc1=weights[0],
            shared_fc1_scale=weights[1],
            shared_fc2=weights[2],
            shared_fc2_scale=weights[3],
        )
    context = api.MegaMoE(
        config,
        object(),
        *weights,
        shared_config=shared,
        shared_fc1=weights[0],
        shared_fc1_scale=weights[1],
        shared_fc2=weights[2],
        shared_fc2_scale=weights[3],
    )
    assert context.shared_config == shared
    assert len(native.fused_calls) == 1
    native_shared = native.fused_calls[0][6]
    assert native_shared.hidden == 8704 and native_shared.intermediate == 2048 and native_shared.gate_up_clamp == 0.5
    assert native_shared.fc1_delay_ns == -1
    assert native.fused_calls[0][14:] == (context._kernel.path, "")
    delayed = api.SharedExpertConfig(2048, 0.5, hidden=8704, fc1_delay_ns=1500)
    api.MegaMoE(
        config,
        object(),
        *weights,
        shared_config=delayed,
        shared_fc1=weights[0],
        shared_fc1_scale=weights[1],
        shared_fc2=weights[2],
        shared_fc2_scale=weights[3],
    )
    assert native.fused_calls[1][6].fc1_delay_ns == 1500
    for delay in (-1, 2**24, True, 1.5):
        with pytest.raises(ValueError):
            api.SharedExpertConfig(2048, fc1_delay_ns=delay)


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
        {"weight_dtype": "fp8_nv"},
        {"weight_dtype": "fp8_nv", "activation_dtype": "fp8_static"},
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


def test_w8a8_local_config():
    local = dict(world_size=1, max_tokens=16, hidden=128, intermediate=128, num_experts=1, top_k=1)
    cfg = _config(**local, weight_dtype="mxfp8", activation_dtype="fp8_static")
    assert cfg.is_w8a8 and cfg.is_static_fp8 and not cfg.is_w8a16
    assert cfg.activation_scale == 1.0
    scaled = _config(**local, weight_dtype="mxfp8", activation_scale=0.5)
    assert scaled.is_w8a8 and scaled.activation_dtype == "fp8_static" and scaled.activation_scale == 0.5
    with pytest.raises(ValueError):
        _config(**local, weight_dtype="mxfp8", activation_dtype="fp8_static", combine_route_weights=True)
    with pytest.raises(ValueError):
        _config(**{**local, "num_experts": 2}, weight_dtype="mxfp8", activation_dtype="fp8_static")
    nv = _config(**local, weight_dtype="fp8_nv", activation_scale=0.25)
    assert nv.is_w8a8 and nv.activation_dtype == "fp8_static" and nv.activation_scale == 0.25
    for activation_dtype in ("bf16", "mxfp8"):
        with pytest.raises(ValueError):
            _config(**local, weight_dtype="fp8_nv", activation_dtype=activation_dtype)


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


@pytest.mark.parametrize("weight_dtype", ["mxfp8", "fp8_nv"])
def test_native_w8a8_shared_expert_graph(weight_dtype):
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
    hidden, intermediate, scale = 384, 256, 0.0625
    config = _config(
        world_size=1,
        max_tokens=130,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=32,
        weight_dtype=weight_dtype,
        activation_dtype="fp8_static",
        activation_scale=scale,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    generator = torch.Generator(device=device).manual_seed(1729)
    source1 = torch.randn((1, 2 * intermediate, hidden), generator=generator, device=device) / hidden**0.5
    source2 = torch.randn((1, hidden, intermediate), generator=generator, device=device) / intermediate**0.5
    if weight_dtype == "fp8_nv":
        # One FP32 scale per matrix (amax / 448); far from 1, so the epilogue weight-scale multipliers matter.
        scale1 = (source1.abs().amax() / 448).reshape(1)
        scale2 = (source2.abs().amax() / 448).reshape(1)
        fc1 = (source1 / scale1).to(torch.float8_e4m3fn)
        fc2 = (source2 / scale2).to(torch.float8_e4m3fn)
        first = fc1[0].float() * scale1
        second = fc2[0].float() * scale2
        for bad in (0.0, float("nan")):
            with pytest.raises(Exception, match="fp8_nv weight scales"):
                MegaMoE(config, communicator, fc1, torch.full_like(scale1, bad), fc2, scale2)
    else:
        fc1, scale1 = quantize_mxfp8(source1)
        fc2, scale2 = quantize_mxfp8(source2)
        first = dequantize_mxfp8(fc1, scale1, dtype=torch.float32)[0]
        second = dequantize_mxfp8(fc2, scale2, dtype=torch.float32)[0]
    context = MegaMoE(config, communicator, fc1, scale1, fc2, scale2)
    assert context.effective_kernel_config["cluster_size"] == 2

    def expected(inputs):
        quantized = quantize_fp8_static(inputs.float(), scale).float()
        gate, up = ((quantized @ first.T) * scale).chunk(2, dim=-1)
        hidden_fp8 = quantize_fp8_static(torch.nn.functional.silu(gate) * up, scale).float()
        return ((hidden_fp8 @ second.T) * scale).to(torch.bfloat16)

    inputs = torch.randn((130, hidden), generator=generator, device=device, dtype=torch.bfloat16)
    unaligned = torch.empty(130 * hidden + 1, device=device, dtype=torch.bfloat16)[1:].view(130, hidden)
    unaligned.copy_(inputs)
    # Multiple token blocks, the 64- and 32-row activation boxes, and the staged unaligned-input path.
    for tokens in (130, 33, 16, 1):
        for source in (inputs[:tokens], unaligned[:tokens]):
            out = context.forward_shared(source)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected(source), rtol=0.03, atol=0.01)
    with pytest.raises(ValueError):
        context(inputs[:16], torch.zeros((16, 1), dtype=torch.int32, device=device), torch.ones((16, 1), device=device))

    stream = torch.cuda.Stream(device=device)
    output = torch.empty((16, hidden), device=device, dtype=torch.bfloat16)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.forward_shared(inputs[:16], output=output, stream=stream)
    for _ in range(3):
        output.zero_()
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected(inputs[:16]), rtol=0.03, atol=0.01)
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


def test_native_w4a8_fused_fp8nv_shared_expert():
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, SharedExpertConfig, is_available
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
    scale = 0.0625
    config = _config(
        world_size=1,
        max_tokens=33,
        hidden=4096,
        intermediate=4352,
        num_experts=2,
        top_k=2,
        sm_margin=sm_count - 8,
        activation_dtype="fp8_static",
        activation_scale=scale,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    generator = torch.Generator(device=device).manual_seed(6137)
    routed_weights = _weights(config, device)
    shared_hidden, shared_intermediate = 8704, 512
    source1 = (
        torch.randn((1, 2 * shared_intermediate, shared_hidden), generator=generator, device=device)
        / shared_hidden**0.5
    )
    source2 = (
        torch.randn((1, shared_hidden, shared_intermediate), generator=generator, device=device)
        / shared_intermediate**0.5
    )
    shared_scale1 = (source1.abs().amax() / 448).reshape(1)
    shared_scale2 = (source2.abs().amax() / 448).reshape(1)
    shared_fc1 = (source1 / shared_scale1).to(torch.float8_e4m3fn)
    shared_fc2 = (source2 / shared_scale2).to(torch.float8_e4m3fn)
    context = MegaMoE(
        config,
        communicator,
        *routed_weights,
        shared_config=SharedExpertConfig(shared_intermediate, 0.25, hidden=shared_hidden),
        shared_fc1=shared_fc1,
        shared_fc1_scale=shared_scale1,
        shared_fc2=shared_fc2,
        shared_fc2_scale=shared_scale2,
    )
    inputs = torch.randn((33, config.hidden), generator=generator, device=device, dtype=torch.bfloat16)
    shared_inputs = torch.randn((33, shared_hidden), generator=generator, device=device, dtype=torch.bfloat16)
    ids = torch.arange(33, device=device, dtype=torch.int32).remainder(2)
    ids = torch.stack((ids, 1 - ids), dim=1).contiguous()
    scores = torch.rand((33, 2), generator=generator, device=device)
    scores /= scores.sum(dim=1, keepdim=True)

    def shared_reference(source):
        quantized = quantize_fp8_static(source.float(), scale).float()
        first = shared_fc1[0].float() * shared_scale1
        second = shared_fc2[0].float() * shared_scale2
        gate, up = ((quantized @ first.T) * scale).chunk(2, dim=-1)
        gate = gate.clamp(max=0.25)
        up = up.clamp(-0.25, 0.25)
        hidden = quantize_fp8_static(torch.nn.functional.silu(gate) * up, scale).float()
        return ((hidden @ second.T) * scale).to(torch.bfloat16)

    routed_output, shared_output = context.forward_fused_shared(inputs, ids, scores, shared_inputs=shared_inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        routed_output, _reference(config, inputs, ids, scores, routed_weights), rtol=0.125, atol=0.16
    )
    torch.testing.assert_close(shared_output, shared_reference(shared_inputs), rtol=0.03, atol=0.01)

    empty_routed, empty_shared = context.forward_fused_shared(
        inputs[:0], ids[:0], scores[:0], shared_inputs=shared_inputs[:0]
    )
    torch.cuda.synchronize()
    assert empty_routed.shape == (0, config.hidden) and empty_shared.shape == (0, shared_hidden)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    graph_routed = torch.empty_like(inputs)
    graph_shared = torch.empty_like(shared_inputs)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        context.forward_fused_shared(
            inputs,
            ids,
            scores,
            shared_inputs=shared_inputs,
            output=graph_routed,
            shared_output=graph_shared,
            stream=stream,
        )
    graph.replay()
    stream.synchronize()
    torch.testing.assert_close(graph_shared, shared_reference(shared_inputs), rtol=0.03, atol=0.01)


@pytest.mark.nranks(2)
def test_native_w4a8_fused_shared_remote_routes_and_graph():
    from mpi4py import MPI
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, SharedExpertConfig, is_available
    from mscclpp.ext.megamoe.benchmark import _weights

    torch = pytest.importorskip("torch")
    group = MPI.COMM_WORLD
    if group.size != 2:
        pytest.skip("requires exactly two MPI ranks")
    supported = (
        is_available()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7))
    )
    if not all(group.allgather(supported)):
        pytest.skip("requires native MegaMoE on two NVLink-connected SM100-family GPUs")
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    scale = 0.0625
    config = _config(
        rank=group.rank,
        world_size=2,
        max_tokens=33,
        hidden=4096,
        intermediate=4352,
        num_experts=4,
        top_k=2,
        sm_margin=sm_count - 8,
        activation_dtype="fp8_static",
        activation_scale=scale,
    )
    bootstrap = TcpBootstrap.create(group.rank, group.size)
    unique_id = TcpBootstrap.create_unique_id() if group.rank == 0 else None
    bootstrap.initialize(group.bcast(unique_id, root=0))
    communicator = Communicator(bootstrap)
    generator = torch.Generator(device=device).manual_seed(9201 + group.rank)
    routed_weights = _weights(config, device)
    shared_intermediate = 512
    source1 = (
        torch.randn((1, 2 * shared_intermediate, config.hidden), generator=generator, device=device)
        / config.hidden**0.5
    )
    source2 = (
        torch.randn((1, config.hidden, shared_intermediate), generator=generator, device=device)
        / shared_intermediate**0.5
    )
    shared_scale1 = (source1.abs().amax() / 448).reshape(1)
    shared_scale2 = (source2.abs().amax() / 448).reshape(1)
    shared_fc1 = (source1 / shared_scale1).to(torch.float8_e4m3fn)
    shared_fc2 = (source2 / shared_scale2).to(torch.float8_e4m3fn)
    context = MegaMoE(
        config,
        communicator,
        *routed_weights,
        shared_config=SharedExpertConfig(shared_intermediate, 0.25),
        shared_fc1=shared_fc1,
        shared_fc1_scale=shared_scale1,
        shared_fc2=shared_fc2,
        shared_fc2_scale=shared_scale2,
    )
    tokens = 33 if group.rank == 0 else 17
    inputs = torch.randn((tokens, config.hidden), generator=generator, device=device, dtype=torch.bfloat16)
    rows = torch.arange(tokens, device=device, dtype=torch.int32)
    remote_base, local_base = (2, 0) if group.rank == 0 else (0, 2)
    ids = torch.stack((remote_base + rows.remainder(2), local_base + (rows // 2).remainder(2)), dim=1)
    ids[::7, 1] = -1
    ids[::11] = -1
    scores = torch.full((tokens, 2), 0.5, device=device)
    scores[ids == -1] = float("nan")

    def shared_reference(source):
        quantized = quantize_fp8_static(source.float(), scale).float()
        gate, up = ((quantized @ (shared_fc1[0].float() * shared_scale1).T) * scale).chunk(2, dim=-1)
        gate = gate.clamp(max=0.25)
        up = up.clamp(-0.25, 0.25)
        hidden = quantize_fp8_static(torch.nn.functional.silu(gate) * up, scale).float()
        return ((hidden @ (shared_fc2[0].float() * shared_scale2).T) * scale).to(torch.bfloat16)

    routed_reference = context(inputs, ids, scores, validate_routing=False)
    torch.cuda.synchronize()
    bootstrap.barrier()
    for _ in range(3):
        routed, shared = context.forward_fused_shared(inputs, ids, scores)
        torch.cuda.synchronize()
        bootstrap.barrier()
        torch.testing.assert_close(routed, routed_reference, rtol=0.05, atol=0.02)
        torch.testing.assert_close(shared, shared_reference(inputs), rtol=0.03, atol=0.01)

    graph_routed = torch.empty_like(inputs)
    graph_shared = torch.empty_like(inputs)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    bootstrap.barrier()
    with torch.cuda.graph(graph, stream=stream):
        context.forward_fused_shared(
            inputs, ids, scores, output=graph_routed, shared_output=graph_shared, stream=stream
        )
    bootstrap.barrier()
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph.replay()
    stream.synchronize()
    bootstrap.barrier()
    torch.testing.assert_close(graph_routed, routed_reference, rtol=0.05, atol=0.02)
    torch.testing.assert_close(graph_shared, shared_reference(inputs), rtol=0.03, atol=0.01)
    del graph, context
    gc.collect()
    bootstrap.barrier()


@pytest.mark.parametrize("weight_dtype,local_expert", [("mxfp4", False), ("mxfp8", False), ("mxfp8", True)])
def test_native_shared_workspace_weights_lifetime_and_graph(weight_dtype, local_expert):
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
    config = _config(
        world_size=1,
        max_tokens=17,
        hidden=4096 if weight_dtype == "mxfp4" else 128,
        intermediate=4352 if weight_dtype == "mxfp4" else 128,
        num_experts=1 if local_expert else 2,
        top_k=1 if local_expert else 2,
        sm_margin=torch.cuda.get_device_properties(device).multi_processor_count - 8,
        weight_dtype=weight_dtype,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    torch.manual_seed(2718)
    weights = [_weights(config, device) for _ in range(2)]
    assert not torch.equal(weights[0][0].view(torch.uint8), weights[1][0].view(torch.uint8))
    owner = MegaMoE(config, communicator, *weights[0])
    alias = MegaMoE(config, communicator, *weights[1], shared_workspace=owner)
    chained = MegaMoE(config, communicator, *weights[0], shared_workspace=alias)
    independent = MegaMoE(config, communicator, *weights[1])
    assert owner.workspace_id == alias.workspace_id == chained.workspace_id != independent.workspace_id
    pointers = [p for context in (owner, alias, chained, independent) for p in context.packed_weight_pointers]
    assert len(set(pointers)) == len(pointers)
    assert owner.input_view().data_ptr() == alias.input_view().data_ptr() == chained.input_view().data_ptr()
    with pytest.raises(ValueError, match="same native configuration"):
        MegaMoE(replace(config, max_tokens=18), communicator, *weights[0], shared_workspace=owner)
    with pytest.raises(ValueError, match="same CUDA device and communicator"):
        MegaMoE(config, Communicator(bootstrap), *weights[0], shared_workspace=owner)

    inputs = torch.randn((17, config.hidden), device=device, dtype=torch.bfloat16).mul_(0.5)
    ids = torch.arange(config.top_k, device=device, dtype=torch.int32).repeat(17, 1)
    scores = torch.full((17, config.top_k), 1.0 / config.top_k, device=device)
    outputs = [torch.empty_like(inputs) for _ in range(2)]
    expected = [_reference(config, inputs, ids, scores, bank) for bank in weights]
    assert not torch.allclose(expected[0], expected[1], rtol=0.05, atol=0.02)
    owner(inputs, ids, scores, output=outputs[0])
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0], expected[0], rtol=0.05, atol=0.02)
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None

    stream = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    alias(inputs, ids, scores, output=outputs[1], stream=stream, signal_start=True)
    chained.wait_until_started(consumer)
    stream.wait_stream(consumer)
    chained(inputs, ids, scores, output=outputs[0], stream=stream)
    with pytest.raises(ValueError, match="preceding signalStart"):
        alias.wait_until_started(consumer)
    stream.synchronize()
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output, reference, rtol=0.05, atol=0.02)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        alias(inputs, ids, scores, output=outputs[1], stream=stream, signal_start=True)
        chained.wait_until_started(stream)
        if local_expert:
            chained.forward_shared(inputs, output=outputs[0], stream=stream)
        else:
            chained(inputs, ids, scores, output=outputs[0], stream=stream)
    with torch.cuda.stream(stream):
        for _ in range(5):
            graph.replay()
            stream.synchronize()
            for output, reference in zip(outputs, expected):
                torch.testing.assert_close(output, reference, rtol=0.05, atol=0.02)
            chained(inputs, ids, scores, output=outputs[0], stream=stream)
            alias(inputs, ids, scores, output=outputs[1], stream=stream)
            stream.synchronize()
            for output, reference in zip(outputs, expected):
                torch.testing.assert_close(output, reference, rtol=0.05, atol=0.02)
    del graph, alias, chained, independent


@pytest.mark.nranks(2)
def test_native_shared_workspace_collective_owner_validation():
    from mpi4py import MPI
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available
    from mscclpp.ext.megamoe.benchmark import _weights

    torch = pytest.importorskip("torch")
    group = MPI.COMM_WORLD
    if group.size != 2:
        pytest.skip("requires exactly two MPI ranks")
    supported = (
        is_available()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7))
    )
    if not all(group.allgather(supported)):
        pytest.skip("requires native MegaMoE on two NVLink-connected SM100-family GPUs")
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(
        rank=group.rank,
        world_size=2,
        max_tokens=17,
        hidden=128,
        intermediate=128,
        num_experts=4,
        top_k=2,
        sm_margin=torch.cuda.get_device_properties(device).multi_processor_count - 8,
        weight_dtype="mxfp8",
    )
    bootstrap = TcpBootstrap.create(group.rank, group.size)
    unique_id = TcpBootstrap.create_unique_id() if group.rank == 0 else None
    bootstrap.initialize(group.bcast(unique_id, root=0))
    communicator = Communicator(bootstrap)
    torch.manual_seed(3141 + group.rank)
    weights = [_weights(config, device) for _ in range(2)]
    # Deliberately reuse the tag: independent owners must still be distinguishable.
    first = MegaMoE(config, communicator, *weights[0])
    second = MegaMoE(config, communicator, *weights[1])
    for owner in (None if group.rank == 0 else first, first if group.rank == 0 else second):
        with pytest.raises(ValueError, match="mode and original workspace owner"):
            MegaMoE(config, communicator, *weights[1], shared_workspace=owner)
    del owner
    alias = MegaMoE(config, communicator, *weights[1], shared_workspace=first)
    assert alias.workspace_id == first.workspace_id != second.workspace_id
    inputs = torch.randn((17, 128), device=device, dtype=torch.bfloat16).mul_(0.125)
    ids = torch.arange(34, device=device, dtype=torch.int32).reshape(17, 2).remainder(4)
    scores = torch.full((17, 2), 0.5, device=device)
    expected = second(inputs, ids, scores)
    outputs = [torch.empty_like(inputs) for _ in range(2)]
    first(inputs, ids, scores, output=outputs[0])
    alias(inputs, ids, scores, output=outputs[1])
    torch.cuda.synchronize()
    bootstrap.barrier()
    torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
    del first
    gc.collect()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        alias(inputs, ids, scores, output=outputs[1], stream=stream)
    with torch.cuda.stream(stream):
        for _ in range(5):
            graph.replay()
    stream.synchronize()
    bootstrap.barrier()
    torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
    del graph, alias, second
    gc.collect()
    bootstrap.barrier()
