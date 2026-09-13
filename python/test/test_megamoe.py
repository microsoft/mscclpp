# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side API tests; numerical GPU validation is also available via --check."""

import pytest

from mscclpp.ext.megamoe import MegaMoEConfig, dequantize_mxfp8, quantize_mxfp8


def _config(**overrides):
    fields = dict(rank=0, world_size=4, max_tokens=32, hidden=4096, intermediate=4352, num_experts=64, top_k=7)
    fields.update(overrides)
    return MegaMoEConfig(**fields)


def test_config_dimensions():
    config = _config(sm_margin=32)
    assert config.local_experts == 16
    assert config.intermediate == 4352
    assert config.sm_margin == 32


@pytest.mark.parametrize(
    "override",
    [
        {"rank": 4},
        {"world_size": 0},
        {"world_size": 73},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"hidden": 4097},
        {"intermediate": 0},
        {"num_experts": 63},
        {"top_k": 0},
        {"top_k": 33},
        {"sm_margin": -1},
        {"weight_e5m2": 1},
        {"gate_up_clamp": float("nan")},
        {"gate_up_clamp": float("inf")},
        {"max_tokens": 2**30},
        {"hidden": 2**31},
    ],
)
def test_config_rejects_invalid(override):
    with pytest.raises(ValueError):
        _config(**override)


@pytest.mark.parametrize("e5m2", [False, True])
def test_mxfp8_canonical_roundtrip(e5m2):
    torch = pytest.importorskip("torch")
    source = torch.arange(-64, 64, dtype=torch.float32).reshape(2, 64) / 64
    weights, scales = quantize_mxfp8(source, e5m2=e5m2)
    assert weights.shape == (2, 64)
    assert scales.shape == (2, 2)
    assert scales.dtype == torch.uint8
    assert weights.dtype == (torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn)
    restored = dequantize_mxfp8(weights, scales, dtype=torch.float32)
    torch.testing.assert_close(restored, source, atol=0.07, rtol=0.13)


def test_mxfp8_zeros_and_scale_layout():
    torch = pytest.importorskip("torch")
    weights, scales = quantize_mxfp8(torch.zeros(2, 128, 32))
    assert torch.all(scales == 127)
    assert torch.count_nonzero(dequantize_mxfp8(weights, scales)) == 0
    with pytest.raises(ValueError, match="scales"):
        dequantize_mxfp8(weights, scales.squeeze(-1))
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp8(torch.ones(2, 31))
    with pytest.raises(ValueError, match="finite"):
        quantize_mxfp8(torch.full((2, 32), float("nan")))


def test_native_views_staging_and_graph_lifetime():
    import gc
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available

    torch = pytest.importorskip("torch")
    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on an SM100 GPU")
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    config = _config(world_size=1, max_tokens=2, hidden=128, intermediate=128, num_experts=2, top_k=1, sm_margin=32)
    device = torch.device("cuda", torch.cuda.current_device())
    first, first_scale = quantize_mxfp8(torch.full((2, 256, 128), 1 / 128, device=device))
    second, second_scale = quantize_mxfp8(torch.full((2, 128, 128), 1 / 128, device=device))
    context = MegaMoE(config, Communicator(bootstrap), first, first_scale, second, second_scale)
    ids = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    scores = torch.ones((2, 1), dtype=torch.float32, device=device)
    inputs = torch.ones((2, 128), dtype=torch.bfloat16, device=device)
    expected = context(inputs, ids, scores)
    view = context.input_view(2)
    view.copy_(inputs)
    output = context(view, ids, scores, validate_routing=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(TypeError, match="dtype"):
        context(view, ids.long(), scores)
    with pytest.raises(ValueError, match="routing IDs"):
        context(view, ids + 2, scores, validate_routing=True)
    with pytest.raises(ValueError, match="workspace"):
        context(view, ids, scores, output=view)
    with pytest.raises(ValueError, match="capacity"):
        context.input_view(3)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=stream):
        context(view, ids, scores, output=output, stream=stream)
    with torch.cuda.stream(stream):
        view.zero_()
        graph.replay()
    stream.synchronize()
    assert torch.count_nonzero(output) == 0
    with torch.cuda.stream(stream):
        view.fill_(1)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert context(view[:0], ids[:0], scores[:0]).shape == (0, 128)
    torch.cuda.synchronize()
    del graph, context
    gc.collect()
    # DLPack owns the native context independently of the Python API wrapper.
    view.fill_(2)
    torch.cuda.synchronize()
    assert torch.all(view == 2)


def test_native_started_stream_wait_replays():
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available

    torch = pytest.importorskip("torch")
    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on an SM100 GPU")
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(world_size=1, max_tokens=2, hidden=128, intermediate=128, num_experts=1, top_k=1, sm_margin=32)
    fc1, scale1 = quantize_mxfp8(torch.full((1, 256, 128), 1 / 128, device=device))
    fc2, scale2 = quantize_mxfp8(torch.full((1, 128, 128), 1 / 128, device=device))
    context = MegaMoE(config, Communicator(bootstrap), fc1, scale1, fc2, scale2)
    inputs = torch.ones((2, 128), dtype=torch.bfloat16, device=device)
    ids = torch.zeros((2, 1), dtype=torch.int32, device=device)
    scores = torch.ones((2, 1), dtype=torch.float32, device=device)
    output = torch.empty_like(inputs)
    marker = torch.zeros((), dtype=torch.int32, device=device)
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    producer.wait_stream(torch.cuda.current_stream(device))
    with pytest.raises(ValueError, match="signalStart"):
        context.wait_until_started(consumer)
    with pytest.raises(TypeError, match="signal_start"):
        context(inputs, ids, scores, signal_start=1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=producer):
        for _ in range(2):
            context(inputs, ids, scores, output=output, stream=producer, signal_start=True)
            context.wait_until_started(consumer)
            with torch.cuda.stream(consumer):
                marker.add_(1)
            producer.wait_stream(consumer)
    for value in (0.0, 1.0, 0.5):
        with torch.cuda.stream(producer):
            inputs.fill_(value)
            marker.zero_()
            for _ in range(10):
                graph.replay()
        producer.synchronize()
        assert marker.item() == 20
        expected = context(inputs, ids, scores)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="signalStart"):
        context.wait_until_started(consumer)


@pytest.mark.parametrize("e5m2", [False, True])
def test_native_ragged_tiles_and_pipeline_reuse(e5m2):
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available

    torch = pytest.importorskip("torch")
    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on an SM100 GPU")
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
        weight_e5m2=e5m2,
        gate_up_clamp=0.125,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    generator = torch.Generator(device=device).manual_seed(4189)
    source1 = torch.randn((4, 1280, 384), generator=generator, device=device).mul_(0.03125)
    source2 = torch.randn((4, 384, 640), generator=generator, device=device).mul_(0.03125)
    fc1, sf1 = quantize_mxfp8(source1, e5m2=e5m2)
    fc2, sf2 = quantize_mxfp8(source2, e5m2=e5m2)
    context = MegaMoE(config, Communicator(bootstrap), fc1, sf1, fc2, sf2)
    first = dequantize_mxfp8(fc1, sf1).float()
    second = dequantize_mxfp8(fc2, sf2).float()
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
        for _ in range(20):
            graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output[:49], reference(49), rtol=0.02, atol=0.001)
    torch.cuda.synchronize()
    del graph, context
