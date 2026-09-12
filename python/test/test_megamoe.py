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
