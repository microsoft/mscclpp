# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU benchmark tests and explicitly opt-in, single-GPU SM100 integration."""

import gc
import os
from types import SimpleNamespace

import pytest

from mscclpp.ext.megamoe import dequantize_mxfp8, quantize_mxfp8
from mscclpp.ext.megamoe.benchmark_shared import (
    MODES,
    _assert_ctas,
    _configs,
    _error_stats,
    _Frontend,
    _parse_args,
    _Postprocess,
    _scope_report,
    _shared_reference,
    _summarize_samples,
    _trace_summary,
)


@pytest.mark.parametrize("world", [4, 32])
def test_default_geometry_and_independent_shared_config(world):
    args = _parse_args([])
    routed, shared = _configs(args, world - 1, world, 152)
    assert (args.tokens, args.original_hidden, args.graph_batch) == (32, 8704, 5)
    assert (routed.hidden, routed.intermediate, routed.num_experts, routed.top_k) == (4096, 4352, 16 * world, 7)
    assert routed.local_experts == 16
    assert routed.rank == world - 1
    assert routed.sm_margin == 32
    assert (shared.rank, shared.world_size, shared.num_experts, shared.top_k) == (0, 1, 1, 1)
    assert (shared.hidden, shared.intermediate, shared.sm_margin) == (8704, 2048, 120)
    assert args.post_norm and args.residual
    assert args.residual_dtype == "fp32" and args.rms_eps == 1e-6


@pytest.mark.parametrize(
    "argv",
    [
        ["--tokens", "-1"],
        ["--iterations", "0"],
        ["--graph-batch", "0"],
        ["--warmup", "0"],
        ["--route-sm-margin", "-1"],
        ["--shared-sms", "1"],
        ["--shared-sms", "31"],
        ["--reference-relative-l2", "nan"],
        ["--reference-relative-l2", "0"],
        ["--bootstrap-port", "65536"],
        ["--trace-eager"],
        ["--rms-eps", "0"],
        ["--rms-eps", "-1"],
        ["--rms-eps", "nan"],
        ["--rms-eps", "inf"],
        ["--residual-dtype", "fp16"],
    ],
)
def test_argument_rejection(argv):
    with pytest.raises(SystemExit):
        _parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--original-hidden", "129"],
        ["--hidden", "129"],
        ["--intermediate", "0"],
        ["--shared-intermediate", "129"],
        ["--experts", "63"],
        ["--top-k", "33"],
        ["--shared-sms", "154"],
        ["--route-sm-margin", "151"],
        ["--route-sm-margin", "31"],
    ],
)
def test_config_rejection(argv):
    with pytest.raises(ValueError):
        _configs(_parse_args(argv), 0, 4, 152)


def test_zero_tokens_reserve_valid_native_capacity():
    args = _parse_args(["--tokens", "0", "--bootstrap-port", "29501"])
    routed, shared = _configs(args, 0, 4, 152)
    assert routed.max_tokens == shared.max_tokens == 1
    assert args.tokens == 0


@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_postprocess_flags_and_reported_scope(post_norm, residual):
    args = _parse_args(
        [
            "--post-norm" if post_norm else "--no-post-norm",
            "--residual" if residual else "--no-residual",
            "--residual-dtype",
            "bf16",
            "--rms-eps",
            "0.0001",
        ]
    )
    report = _scope_report(args)
    assert args.post_norm == post_norm and args.residual == residual
    assert args.rms_eps == 1e-4
    assert ("postnorm" in report["excluded"]) == (not post_norm)
    assert ("residual" in report["excluded"]) == (not residual)
    assert report["postprocess"]["final_dtype"] == "BF16"
    for mode in ("routed-only", "serial", "overlap"):
        stages = report["included_by_mode"][mode]
        assert ("post_rmsnorm" in stages) == post_norm
        assert ("residual_add" in stages) == residual
        assert ("postnorm_bypass_bf16_copy" in stages) == (not post_norm)
    assert report["included_by_mode"]["shared-only"] == ["shared_input_staging", "shared_expert"]
    assert _scope_report(_parse_args([]))["postprocess"]["final_dtype"] == "FP32"
    assert _scope_report(_parse_args(["--no-residual"]))["postprocess"]["final_dtype"] == "BF16"


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), True, "0.001"])
def test_postprocess_rejects_invalid_epsilon(epsilon):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="epsilon"):
        _Postprocess(torch.zeros(1, 8), epsilon=epsilon)


def test_postprocess_rejects_invalid_configuration_and_shapes():
    torch = pytest.importorskip("torch")
    for kwargs in ({"post_norm": 1}, {"add_residual": 1}, {"residual_dtype": "fp16"}):
        with pytest.raises(ValueError):
            _Postprocess(torch.zeros(1, 8), **kwargs)
    for source in (torch.zeros(8), torch.zeros(1, 0), torch.zeros(1, 8, dtype=torch.int32)):
        with pytest.raises(ValueError, match="residual"):
            _Postprocess(source)
    with pytest.raises(ValueError, match="inference-only"):
        _Postprocess(torch.zeros(1, 8, requires_grad=True))
    p = _Postprocess(torch.zeros(1, 8))
    with pytest.raises(ValueError, match="branch_output"):
        p.forward(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="reference residual shape"):
        p.reference(torch.zeros(1, 8), torch.zeros(2, 8))


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
def test_postprocess_reference_buffers_and_no_replay_feedback(tokens, post_norm, add_residual, residual_dtype):
    torch = pytest.importorskip("torch")
    torch.manual_seed(83)
    source = torch.randn(tokens, 16, dtype=torch.float32) + 0.0001
    branch = torch.randn(tokens, 16, dtype=torch.bfloat16)
    p = _Postprocess(source, post_norm=post_norm, add_residual=add_residual, residual_dtype=residual_dtype)
    dtype = torch.float32 if residual_dtype == "fp32" else torch.bfloat16
    assert p.residual.dtype == dtype
    assert p.output.dtype == (dtype if add_residual else torch.bfloat16)
    assert p.normalized.dtype == torch.bfloat16
    assert p.weight.dtype == torch.float32 and torch.unique(p.weight).numel() == 16
    assert not torch.equal(p.weight, p.weight.bfloat16().float())
    if tokens:
        assert p.residual.data_ptr() != source.data_ptr()
        assert p.normalized.data_ptr() != branch.data_ptr()
    pointers = {name: tensor.data_ptr() for name, tensor in vars(p).items() if isinstance(tensor, torch.Tensor)}
    original_residual = p.residual.clone()
    source.fill_(9)
    torch.testing.assert_close(p.residual, original_residual, rtol=0, atol=0)
    expected = p.reference(branch)
    for _ in range(3):
        actual = p.forward(branch)
        assert actual is p.output and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.008)
        torch.testing.assert_close(p.residual, original_residual, rtol=0, atol=0)
    branch.neg_()
    p.residual.add_(0.125)
    expected = p.reference(branch)
    assert p.normalize(branch) is p.normalized
    torch.testing.assert_close(p.add_residual(), expected, rtol=0.01, atol=0.008)
    if not post_norm:
        torch.testing.assert_close(p.normalized, branch, rtol=0, atol=0)
    assert pointers == {name: tensor.data_ptr() for name, tensor in vars(p).items() if isinstance(tensor, torch.Tensor)}


def test_postprocess_fp32_rms_operands_and_normalize_before_residual(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    branch = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.2501, 1.0001, 4.0001, 8.0001]], dtype=torch.float32)
    p = _Postprocess(residual, epsilon=0.125)
    original_rms_norm = functional.rms_norm
    calls = []

    def checked_rms_norm(input, normalized_shape, weight, eps):
        assert input.dtype == weight.dtype == torch.float32
        assert weight is p.weight and eps == p.epsilon
        calls.append(normalized_shape)
        return original_rms_norm(input, normalized_shape, weight=weight, eps=eps)

    monkeypatch.setattr(functional, "rms_norm", checked_rms_norm)
    normalized = branch.float() * torch.rsqrt(branch.float().square().mean(-1, keepdim=True) + p.epsilon) * p.weight
    expected = normalized.bfloat16().float() + residual
    torch.testing.assert_close(p.forward(branch), expected, rtol=0, atol=0)
    assert calls == [(4,)]
    wrong_input = branch.float() + residual
    wrong_order = (
        wrong_input * torch.rsqrt(wrong_input.square().mean(-1, keepdim=True) + p.epsilon) * p.weight
    ).bfloat16()
    assert not torch.allclose(p.output, wrong_order.float())
    default_epsilon = _Postprocess(residual).reference(branch)
    assert not torch.equal(p.output, default_epsilon)


def test_postprocess_keeps_fp32_residual_fractional_bits_and_independent_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    source = torch.full((2, 8), 1 + 2**-12, dtype=torch.float32)
    branch = torch.zeros((2, 8), dtype=torch.bfloat16)
    p = _Postprocess(source)
    torch.testing.assert_close(p.forward(branch), source, rtol=0, atol=0)
    assert not torch.equal(p.output, source.bfloat16().float())

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU oracle must not invoke the implementation")

    monkeypatch.setattr(functional, "rms_norm", forbidden)
    monkeypatch.setattr(p, "forward", forbidden)
    torch.testing.assert_close(p.reference(branch), source, rtol=0, atol=0)
    alternate = source + 0.25
    torch.testing.assert_close(p.reference(branch, alternate), alternate, rtol=0, atol=0)


def test_actual_cta_counts_must_match_requested_caps():
    args = _parse_args([])
    _assert_ctas(SimpleNamespace(cta_count=120), SimpleNamespace(cta_count=32), 152, args)
    with pytest.raises(RuntimeError, match="occupancy selected"):
        _assert_ctas(SimpleNamespace(cta_count=118), SimpleNamespace(cta_count=32), 152, args)
    with pytest.raises(RuntimeError, match="occupancy selected"):
        _assert_ctas(SimpleNamespace(cta_count=120), SimpleNamespace(cta_count=30), 152, args)


def _small_args(tokens=2):
    return _parse_args(
        [
            "--tokens",
            str(tokens),
            "--original-hidden",
            "256",
            "--hidden",
            "128",
            "--intermediate",
            "128",
            "--shared-intermediate",
            "128",
            "--experts",
            "2",
            "--top-k",
            "1",
            "--graph-batch",
            "2",
        ]
    )


@pytest.mark.parametrize("tokens", [0, 2])
def test_frontend_shapes_router_precision_and_buffer_reuse(tokens):
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    args = _small_args(tokens)
    args.experts, args.top_k = 8, 3
    routed, _ = _configs(args, 0, 1, 152)
    inputs = torch.randn((tokens, 256), dtype=torch.bfloat16)
    f = _Frontend(inputs, routed)
    pointers = {name: value.data_ptr() for name, value in vars(f).items() if isinstance(value, torch.Tensor)}
    f.router()
    expected_logits = inputs.float() @ f.router_weight
    selected = expected_logits.topk(3, dim=-1)
    torch.testing.assert_close(f.logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(f.ids, selected.indices.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(f.scores, selected.values.softmax(dim=-1), rtol=0, atol=0)
    assert f.logits.dtype == f.scores.dtype == torch.float32
    assert f.ids.dtype == f.shared_ids.dtype == torch.int32
    assert f.shared_ids.shape == f.shared_scores.shape == (tokens, 1)
    assert f.ids.shape == f.scores.shape == (tokens, 3)
    assert f.router_weight.shape == (256, 8)
    assert f.squash_weight.shape == (256, 128)
    assert f.unsquash_weight.shape == (128, 256)
    f.squash()
    assert f.squashed.shape == (tokens, 128)
    torch.testing.assert_close(f.squashed, inputs @ f.squash_weight, rtol=0, atol=0)
    f.routed_output.copy_(f.squashed)
    f.unsquash()
    torch.testing.assert_close(f.unsquashed, f.routed_output @ f.unsquash_weight, rtol=0, atol=0)
    f.shared_output.fill_(0.25)
    f.combine()
    torch.testing.assert_close(f.combined, f.unsquashed + f.shared_output, rtol=0, atol=0)
    f.postprocess.forward(f.combined)
    torch.testing.assert_close(f.output, f.postprocess.reference(f.combined), rtol=0.01, atol=0.008)
    assert f.output.shape == (tokens, 256)
    assert f.output.dtype == torch.float32
    if tokens:
        original_ids = f.ids.clone()
        f.inputs.neg_()
        f.router()
        assert not torch.equal(original_ids, f.ids)
    assert pointers == {name: value.data_ptr() for name, value in vars(f).items() if isinstance(value, torch.Tensor)}


@pytest.mark.parametrize("postprocess_enabled", [False, True])
def test_schedule_postprocess_cpu_wiring_and_shared_only_scope(monkeypatch, postprocess_enabled):
    from contextlib import nullcontext
    from mscclpp.ext.megamoe.benchmark_shared import _Layer

    torch = pytest.importorskip("torch")
    routed_config, _ = _configs(_small_args(), 0, 1, 152)
    inputs = torch.ones(2, 256, dtype=torch.bfloat16)
    p = _Postprocess(inputs, post_norm=postprocess_enabled, add_residual=postprocess_enabled)
    f = _Frontend(inputs, routed_config, postprocess=p)

    class Context:
        def forward(self, inputs, ids, scores, *, output, stream, signal_start=False):
            torch.mul(inputs, 0.5, out=output)

        def forward_shared(self, inputs, *, output, stream):
            torch.mul(inputs, 0.5, out=output)

        def wait_until_started(self, stream):
            pass

    class Stream:
        def wait_event(self, event):
            pass

    class Event:
        def record(self, stream):
            pass

    layer = _Layer.__new__(_Layer)
    layer.frontend, layer.routed, layer.shared = f, Context(), Context()
    layer.stream, layer.shared_stream, layer.shared_done = Stream(), Stream(), Event()
    layer.annotate = False
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    for mode in MODES:
        f.combined.fill_(999)
        p.output.fill_(7)
        actual = layer.launch(mode)
        if mode == "shared-only":
            torch.testing.assert_close(actual, inputs * 0.5, rtol=0, atol=0)
            assert torch.all(p.output == 7)
        else:
            branch = f.unsquashed if mode == "routed-only" else f.combined
            torch.testing.assert_close(actual, p.reference(branch), rtol=0.01, atol=0.008)
            assert actual is p.output
            assert actual.dtype == (torch.float32 if postprocess_enabled else torch.bfloat16)


@pytest.mark.parametrize("tokens", [0, 2])
def test_shared_reference_is_local_with_unsquashed_shape_and_bf16_handoff(monkeypatch, tokens):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    def forbidden(*args, **kwargs):
        raise AssertionError("the shared oracle must not use the global distributed group")

    monkeypatch.setattr(dist, "all_gather", forbidden)
    monkeypatch.setattr(dist, "all_reduce", forbidden)
    h, i = 256, 128
    first = torch.cat((torch.full((i, h), 1 / h), torch.full((i, h), -0.5 / h))).unsqueeze(0)
    fc1, sf1 = quantize_mxfp8(first)
    fc2, sf2 = quantize_mxfp8(torch.full((1, h, i), 1 / i))
    inputs = torch.arange(1, tokens + 1).to(torch.bfloat16).view(tokens, 1).expand(tokens, h).contiguous()
    output = _shared_reference(inputs, (fc1, sf1, fc2, sf2))
    gate = torch.arange(1, tokens + 1).float()
    expected_scalar = (torch.nn.functional.silu(gate) * (-0.5 * gate)).to(torch.bfloat16)
    expected = expected_scalar.view(tokens, 1).expand(tokens, h)
    assert output.shape == (tokens, h)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unsquashed input"):
        _shared_reference(inputs[:, :128], (fc1, sf1, fc2, sf2))
    with pytest.raises(ValueError, match="exactly one expert"):
        _shared_reference(inputs, (fc1.expand(2, -1, -1), sf1, fc2, sf2))


def test_reference_metrics_check_shape_and_finiteness():
    torch = pytest.importorskip("torch")
    assert _error_stats(torch.empty(0, 128), torch.empty(0, 128)) == {"relative_l2": 0, "max_abs": 0, "mean_abs": 0}
    assert _error_stats(torch.ones(2, 3), torch.ones(2, 3))["relative_l2"] == 0
    assert _error_stats(torch.full((2, 3), 1.1), torch.ones(2, 3))["relative_l2"] == pytest.approx(0.1)
    with pytest.raises(AssertionError, match="nonfinite"):
        _error_stats(torch.tensor([float("inf")]), torch.ones(1))
    with pytest.raises(AssertionError, match="shape"):
        _error_stats(torch.zeros(2, 1), torch.zeros(2))


def _trace_fixture(shared_offset=20):
    events = [{"cat": "cpu_op", "name": "megamoe/shared", "ts": 0, "dur": 1}]
    for index in range(3):
        for start, duration, grid, stream in (
            (200 * index, 100, 120, 7),
            (200 * index + shared_offset, 50, 32, 9),
        ):
            events.append(
                {
                    "cat": "kernel",
                    "name": "void mscclpp::megamoe::detail::megaMoe<false>()",
                    "ts": start,
                    "dur": duration,
                    "args": {"grid": [grid, 1, 1], "stream": stream},
                }
            )
    return {"traceEvents": events}


def test_trace_uses_gpu_start_times_grids_streams_and_actual_overlap():
    result = _trace_summary(_trace_fixture(), 120, 32)
    assert result["native_kernel_rows"] == 6
    assert result["paired_iterations"] == result["overlapped_iterations"] == 3
    assert result["routed_first"] and result["multistream"]
    assert result["total_overlap_us"] == 150
    assert result["total_paired_span_us"] == 300
    assert result["routed_streams"] == [7] and result["shared_streams"] == [9]
    assert result["pairs"][0]["shared_starts_after_routed_us"] == 20


def test_trace_does_not_invent_overlap_from_multiple_streams():
    result = _trace_summary(_trace_fixture(shared_offset=120), 120, 32)
    assert result["multistream"]
    assert result["overlapped_iterations"] == result["total_overlap_us"] == 0


def test_trace_rejects_wrong_order_missing_kernels_and_single_stream():
    with pytest.raises(RuntimeError, match="started before"):
        _trace_summary(_trace_fixture(shared_offset=-1), 120, 32)
    trace = _trace_fixture()
    trace["traceEvents"].pop()
    with pytest.raises(RuntimeError, match="observed"):
        _trace_summary(trace, 120, 32)
    trace = _trace_fixture()
    for event in trace["traceEvents"][1:]:
        event["args"]["stream"] = 7
    with pytest.raises(RuntimeError, match="separate"):
        _trace_summary(trace, 120, 32)
    with pytest.raises(RuntimeError, match="distinct"):
        _trace_summary(_trace_fixture(), 32, 32)


def test_latency_reduction_uses_maximum_across_ranks_per_iteration():
    reports = [{"samples_us": {mode: values for mode in MODES}} for values in ([100, 1, 1], [1, 100, 1], [1, 1, 100])]
    result = _summarize_samples(reports, "overlap")
    assert result["samples"] == [100, 100, 100]
    assert result["median"] == result["minimum"] == result["maximum"] == 100


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_SHARED") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_SHARED=1 to opt into the single-GPU native SM100 test",
)
def test_native_shared_schedule_graph_changed_input_and_zero_tokens():
    torch = pytest.importorskip("torch")
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, is_available
    from mscclpp.ext.megamoe.benchmark import _weights
    from mscclpp.ext.megamoe.benchmark_shared import _capture, _Layer

    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on SM100")
    device = torch.device("cuda", torch.cuda.current_device())
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    args = _small_args()
    args.original_hidden = 128
    route_config, shared_config = _configs(args, 0, 1, physical_sms)
    bootstraps = [TcpBootstrap.create(0, 1), TcpBootstrap.create(0, 1)]
    for bootstrap in bootstraps:
        bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicators = [Communicator(bootstrap) for bootstrap in bootstraps]
    torch.manual_seed(41)
    route_weights, shared_weights = _weights(route_config, device), _weights(shared_config, device)
    routed = MegaMoE(route_config, communicators[0], *route_weights)
    shared = MegaMoE(shared_config, communicators[1], *shared_weights)
    _assert_ctas(routed, shared, physical_sms, args)
    inputs = torch.randn((2, 128), device=device, dtype=torch.bfloat16)
    f = _Frontend(inputs, route_config)
    layer = _Layer(f, routed, shared)
    layer.launch("serial")
    layer.stream.synchronize()
    reference = _shared_reference(inputs, shared_weights)
    assert _error_stats(f.shared_output, reference)["relative_l2"] < 0.05
    assert torch.count_nonzero(f.shared_output).item() > 0
    layer.stream.wait_stream(torch.cuda.current_stream(device))
    graphs = {mode: _capture(layer, mode, 2) for mode in ("serial", "overlap")}
    original = inputs.clone()
    original_residual = f.postprocess.residual.clone()
    layer.stream.wait_stream(torch.cuda.current_stream(device))
    for scale in (1, -1):
        with torch.cuda.stream(layer.stream):
            inputs.copy_(original * scale)
            f.postprocess.residual.copy_(original_residual * scale + 0.0001)
        layer.launch("serial")
        layer.stream.synchronize()
        expected = f.output.clone()
        assert f.output.dtype == torch.float32 and torch.isfinite(f.output).all()
        assert _error_stats(f.output, f.postprocess.reference(f.combined))["relative_l2"] < 0.01
        layer.stream.wait_stream(torch.cuda.current_stream(device))
        layer.launch("overlap")
        layer.stream.synchronize()
        torch.testing.assert_close(f.output, expected, rtol=0, atol=0)
        for graph in graphs.values():
            for _ in range(2):
                with torch.cuda.stream(layer.stream):
                    graph.replay()
                layer.stream.synchronize()
                torch.testing.assert_close(f.output, expected, rtol=0, atol=0)
    with torch.cuda.stream(layer.stream):
        f.postprocess.residual.add_(0.25)
        graphs["overlap"].replay()
    layer.stream.synchronize()
    assert not torch.equal(f.output, expected)
    expected = f.output.clone()
    layer.stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(layer.stream):
        graphs["overlap"].replay()
    layer.stream.synchronize()
    torch.testing.assert_close(f.output, expected, rtol=0, atol=0)
    torch.cuda.synchronize(device)
    empty = _Layer(_Frontend(inputs[:0], route_config), routed, shared)
    assert empty.launch("overlap").shape == (0, 128)
    zero_graph = _capture(empty, "overlap", 2)
    with torch.cuda.stream(empty.stream):
        zero_graph.replay()
    torch.cuda.synchronize(device)
    del graph, zero_graph
    graphs.clear()
    gc.collect()
    for bootstrap in bootstraps:
        bootstrap.barrier()
    del empty, layer, routed, shared
    gc.collect()
    for bootstrap in bootstraps:
        bootstrap.barrier()
    del communicators
    gc.collect()
    del bootstrap, bootstraps
    gc.collect()


_LOCAL_EXPERT_CAPACITY = 193
_LOCAL_EXPERT_TOKEN_COUNTS = (193, 129, 65, 33, 1, 0, 177)


def _local_expert_weights(hidden, intermediate, e5m2):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(3701)
    weights = []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        source = torch.randn((1, rows, columns), generator=generator, dtype=torch.float32) / columns**0.5
        row_gain = torch.exp2((torch.arange(rows) % 3 - 1).float()).view(1, rows, 1)
        block_gain = torch.exp2((torch.arange(columns // 32) % 5 - 2).float()).repeat_interleave(32)
        source *= row_gain * block_gain
        source[0, 0, :32] = 0
        weights.extend(quantize_mxfp8(source, e5m2=e5m2))
    return weights


def _local_expert_sample(tokens, hidden, phase):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(4813 + phase)
    inputs = (torch.randn((tokens, hidden), generator=generator) * 0.375).to(torch.bfloat16)
    rows = torch.arange(tokens)
    ids = torch.zeros((tokens, 1), dtype=torch.int32)
    if phase == 1:
        ids[((rows + 1) % 3 == 0) | (rows % 32 == 0)] = -1
    elif phase == 2:
        ids[rows % 5 == 2] = -1
    elif phase == 3:
        ids.fill_(-1)
    scores = torch.tensor([-1.25, 0.0, 0.375, 0.7, 1.875], dtype=torch.float32)
    scores = scores[(rows + phase) % scores.numel()].view(tokens, 1).contiguous()
    inputs[ids[:, 0] == -1] = float("nan")
    return inputs, ids, scores


def _local_expert_reference(inputs, ids, scores, weights, clamp):
    """Independent local CPU oracle; masked NaN rows never enter either GEMM."""
    torch = pytest.importorskip("torch")
    inputs, ids, scores = inputs.cpu(), ids.cpu(), scores.cpu()
    fc1, sf1, fc2, sf2 = (weight.cpu() for weight in weights)
    output = torch.zeros(inputs.shape, dtype=torch.bfloat16)
    selected = ids[:, 0] == 0
    if not bool(selected.any()):
        return output
    first = dequantize_mxfp8(fc1[0], sf1[0], dtype=torch.float32)
    second = dequantize_mxfp8(fc2[0], sf2[0], dtype=torch.float32)
    gate, up = (inputs[selected].float() @ first.T).chunk(2, dim=-1)
    if clamp >= 0:
        gate = gate.clamp(max=clamp)
        up = up.clamp(-clamp, clamp)
    activation = (torch.nn.functional.silu(gate) * up * scores[selected]).to(torch.bfloat16)
    output[selected] = (activation.float() @ second.T).to(torch.bfloat16)
    return output


@pytest.mark.parametrize("e5m2", [False, True])
def test_local_expert_fixture_has_nonuniform_fp8_values_and_k32_scales(e5m2):
    torch = pytest.importorskip("torch")
    weights = _local_expert_weights(128, 256, e5m2)
    for values, scales in zip(weights[::2], weights[1::2]):
        assert values.dtype == (torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn)
        assert scales.dtype == torch.uint8 and scales.shape == (*values.shape[:-1], values.shape[-1] // 32)
        assert torch.unique(values.float()).numel() > 16
        assert torch.unique(scales).numel() > 2
        assert not torch.equal(scales[..., :1].expand_as(scales), scales)


@pytest.mark.parametrize("tokens", _LOCAL_EXPERT_TOKEN_COUNTS)
def test_local_expert_fixture_changes_masks_and_poisoned_rows(tokens):
    torch = pytest.importorskip("torch")
    cases = [_local_expert_sample(tokens, 128, phase) for phase in range(5)]
    for inputs, ids, scores in cases:
        assert inputs.shape == (tokens, 128) and ids.shape == scores.shape == (tokens, 1)
        assert torch.isfinite(scores).all()
        assert torch.isnan(inputs[ids[:, 0] == -1]).all()
        assert torch.isfinite(inputs[ids[:, 0] == 0]).all()
    if tokens:
        assert (cases[0][1] == 0).all() and (cases[3][1] == -1).all() and (cases[4][1] == 0).all()
        assert not torch.equal(cases[1][1], cases[2][1])
    if tokens > 1:
        assert torch.count_nonzero(cases[1][1]) != torch.count_nonzero(cases[2][1])
        assert (cases[0][2] == 0).any() and (cases[0][2] < 0).any()


@pytest.mark.parametrize("e5m2", [False, True])
@pytest.mark.parametrize("clamp", [-1.0, 0.125])
def test_local_expert_oracle_weights_clamp_and_masked_nan_without_distributed(monkeypatch, e5m2, clamp):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    def forbidden(*args, **kwargs):
        raise AssertionError("local expert oracle must not use a distributed group")

    monkeypatch.setattr(dist, "all_gather", forbidden)
    monkeypatch.setattr(dist, "all_reduce", forbidden)
    hidden = intermediate = 128
    first = torch.cat(
        (torch.full((intermediate, hidden), 1 / hidden), torch.full((intermediate, hidden), 0.5 / hidden))
    ).unsqueeze(0)
    weights = [
        *quantize_mxfp8(first, e5m2=e5m2),
        *quantize_mxfp8(torch.full((1, hidden, intermediate), 1 / intermediate), e5m2=e5m2),
    ]
    values = torch.tensor([-2.0, 0.5, float("nan"), 3.0, float("nan")], dtype=torch.float32)
    inputs = values[:, None].expand(-1, hidden).to(torch.bfloat16).contiguous()
    ids = torch.tensor([[0], [0], [-1], [0], [-1]], dtype=torch.int32)
    scores = torch.tensor([[-1.25], [0.0], [0.375], [1.875], [-0.5]], dtype=torch.float32)
    result = _local_expert_reference(inputs, ids, scores, weights, clamp)
    selected = ids[:, 0] == 0
    gate, up = values[selected], 0.5 * values[selected]
    if clamp >= 0:
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    scalar = (torch.nn.functional.silu(gate) * up * scores[selected, 0]).to(torch.bfloat16)
    torch.testing.assert_close(result[selected], scalar[:, None].expand(-1, hidden), rtol=0, atol=0)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[~selected]) == 0
    assert torch.count_nonzero(result[scores[:, 0] == 0]) == 0
    assert _local_expert_reference(inputs[:0], ids[:0], scores[:0], weights, clamp).shape == (0, hidden)
    masked = _local_expert_reference(
        torch.full_like(inputs, float("nan")), torch.full_like(ids, -1), scores, weights, clamp
    )
    assert torch.count_nonzero(masked) == 0


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_SHARED") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_SHARED=1 to opt into native local-expert SM100 tests",
)
@pytest.mark.parametrize(
    "e5m2,hidden,intermediate,cta_cap,clamp",
    [
        (False, 128, 256, 32, -1.0),
        (False, 256, 128, 8, 0.125),
        (True, 128, 256, 8, -1.0),
        (True, 256, 128, 32, 0.125),
    ],
)
def test_native_local_expert_weights_masks_and_token_graph_tails(e5m2, hidden, intermediate, cta_cap, clamp):
    torch = pytest.importorskip("torch")
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig, is_available

    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on SM100")
    device = torch.device("cuda", torch.cuda.current_device())
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    config = MegaMoEConfig(
        rank=0,
        world_size=1,
        max_tokens=_LOCAL_EXPERT_CAPACITY,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=physical_sms - cta_cap,
        weight_e5m2=e5m2,
        gate_up_clamp=clamp,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    weights = _local_expert_weights(hidden, intermediate, e5m2)
    gpu_weights = [weight.to(device) for weight in weights]
    context = MegaMoE(config, communicator, *gpu_weights)
    registered = context.input_view(_LOCAL_EXPERT_CAPACITY)
    staged = torch.empty((_LOCAL_EXPERT_CAPACITY, hidden), device=device, dtype=torch.bfloat16)
    ids_buffer = torch.empty((_LOCAL_EXPERT_CAPACITY, 1), device=device, dtype=torch.int32)
    scores_buffer = torch.empty((_LOCAL_EXPERT_CAPACITY, 1), device=device, dtype=torch.float32)
    guarded = torch.empty((_LOCAL_EXPERT_CAPACITY + 2, hidden), device=device, dtype=torch.bfloat16)
    producer, execution = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    producer.wait_stream(torch.cuda.current_stream(device))
    execution.wait_stream(torch.cuda.current_stream(device))
    graphs = []
    graph = direct = None
    try:
        assert 0 < context.cta_count <= cta_cap
        with pytest.raises(ValueError, match="capacity"):
            context.input_view(_LOCAL_EXPERT_CAPACITY + 1)
        # Public value validation still rejects internal masks and nonfinite
        # scores; only validate_routing=False exercises the internal -1 mask.
        with torch.cuda.stream(execution):
            staged[:1].zero_()
            ids_buffer[:1].fill_(-1)
            scores_buffer[:1].fill_(0.375)
            with pytest.raises(ValueError, match="routing IDs"):
                context.forward(
                    staged[:1],
                    ids_buffer[:1],
                    scores_buffer[:1],
                    output=guarded[1:2],
                    stream=execution,
                    validate_routing=True,
                )
            ids_buffer[:1].fill_(1)
            with pytest.raises(ValueError, match="routing IDs"):
                context.forward(
                    staged[:1],
                    ids_buffer[:1],
                    scores_buffer[:1],
                    output=guarded[1:2],
                    stream=execution,
                    validate_routing=True,
                )
            ids_buffer[:1].zero_()
            scores_buffer[:1].fill_(float("nan"))
            with pytest.raises(ValueError, match="finite"):
                context.forward(
                    staged[:1],
                    ids_buffer[:1],
                    scores_buffer[:1],
                    output=guarded[1:2],
                    stream=execution,
                    validate_routing=True,
                )
            scores_buffer[:1].fill_(-1.25)
            context.forward(
                staged[:1],
                ids_buffer[:1],
                scores_buffer[:1],
                output=guarded[1:2],
                stream=execution,
                validate_routing=True,
            )
        execution.synchronize()
        for tokens in _LOCAL_EXPERT_TOKEN_COUNTS:
            direct = registered[:tokens]
            inputs, ids, scores = staged[:tokens], ids_buffer[:tokens], scores_buffer[:tokens]
            output = guarded[1 : tokens + 1]

            def stage(sample):
                producer.wait_stream(execution)
                with torch.cuda.stream(producer):
                    registered.fill_(float("nan"))
                    staged.fill_(float("nan"))
                    inputs.copy_(sample[0])
                    ids.copy_(sample[1])
                    scores.copy_(sample[2])
                    guarded.fill_(7)
                    output.fill_(float("nan"))
                execution.wait_stream(producer)

            def launch_direct():
                direct.copy_(inputs)
                context.forward(direct, ids, scores, output=output, stream=execution, validate_routing=False)

            def check(expected, sample):
                host = guarded.cpu()
                observed = host[1 : tokens + 1]
                assert torch.all(host[:1] == 7) and torch.all(host[tokens + 1 :] == 7)
                assert _error_stats(observed, expected)["relative_l2"] < 0.02
                torch.testing.assert_close(observed, expected, rtol=0.05, atol=0.005)
                assert torch.count_nonzero(observed[sample[1][:, 0] == -1]) == 0
                assert torch.count_nonzero(observed[sample[2][:, 0] == 0]) == 0
                return observed

            stage(_local_expert_sample(tokens, hidden, 0))
            with torch.cuda.stream(execution):
                launch_direct()
            execution.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=execution):
                launch_direct()
                launch_direct()
            graphs.append(graph)
            for phase in range(5):
                sample = _local_expert_sample(tokens, hidden, phase)
                expected = _local_expert_reference(*sample, weights, clamp)
                stage(sample)
                with torch.cuda.stream(execution):
                    context.forward(inputs, ids, scores, output=output, stream=execution, validate_routing=False)
                execution.synchronize()
                eager = check(expected, sample)
                for _ in range(2):
                    stage(sample)
                    with torch.cuda.stream(execution):
                        graph.replay()
                    execution.synchronize()
                    observed = check(expected, sample)
                    torch.testing.assert_close(observed, eager, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize(device)
        graph = None
        graphs.clear()
        gc.collect()
        bootstrap.barrier()
        direct = registered = context = None
        gc.collect()
        bootstrap.barrier()
        communicator = None
        gc.collect()
        bootstrap = None
        gc.collect()


@pytest.mark.parametrize("world,experts,top_k", [(1, 2, 1), (2, 2, 1), (1, 2, 2)])
def test_forward_shared_rejects_nonlocal_or_multiple_experts_before_launch(world, experts, top_k):
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig

    context = MegaMoE.__new__(MegaMoE)
    context.config = MegaMoEConfig(
        rank=0,
        world_size=world,
        max_tokens=1,
        hidden=128,
        intermediate=128,
        num_experts=experts,
        top_k=top_k,
    )
    with pytest.raises(ValueError, match="forward_shared requires"):
        context.forward_shared(None)


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_SHARED") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_SHARED=1 to opt into native forward_shared SM100 tests",
)
@pytest.mark.parametrize(
    "e5m2,hidden,intermediate,clamp",
    [(False, 128, 256, -1.0), (True, 256, 128, 0.125)],
)
def test_native_forward_shared_matches_unit_routing_and_input_view_graphs(e5m2, hidden, intermediate, clamp):
    torch = pytest.importorskip("torch")
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig, is_available

    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on SM100")
    device = torch.device("cuda", torch.cuda.current_device())
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    config = MegaMoEConfig(
        rank=0,
        world_size=1,
        max_tokens=_LOCAL_EXPERT_CAPACITY,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=physical_sms - 32,
        weight_e5m2=e5m2,
        gate_up_clamp=clamp,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    weights = _local_expert_weights(hidden, intermediate, e5m2)
    gpu_weights = [weight.to(device) for weight in weights]
    context = MegaMoE(config, communicator, *gpu_weights)
    registered = context.input_view(_LOCAL_EXPERT_CAPACITY)
    source = torch.empty((_LOCAL_EXPERT_CAPACITY, hidden), device=device, dtype=torch.bfloat16)
    ids_buffer = torch.zeros((_LOCAL_EXPERT_CAPACITY, 1), device=device, dtype=torch.int32)
    scores_buffer = torch.ones((_LOCAL_EXPERT_CAPACITY, 1), device=device, dtype=torch.float32)
    routed_guard = torch.empty((_LOCAL_EXPERT_CAPACITY + 2, hidden), device=device, dtype=torch.bfloat16)
    shared_guard = torch.empty_like(routed_guard)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graphs = []
    graph = direct = None
    try:
        assert 0 < context.cta_count <= 32
        for tokens in _LOCAL_EXPERT_TOKEN_COUNTS:
            direct = registered[:tokens]
            inputs, ids, scores = source[:tokens], ids_buffer[:tokens], scores_buffer[:tokens]
            routed_output = routed_guard[1 : tokens + 1]
            shared_output = shared_guard[1 : tokens + 1]
            direct_pointer, output_pointer = direct.data_ptr(), shared_output.data_ptr()

            def launch_shared():
                direct.copy_(inputs)
                return context.forward_shared(direct, output=shared_output, stream=stream)

            def check_shared(expected):
                host = shared_guard.cpu()
                observed = host[1 : tokens + 1]
                assert torch.isfinite(observed).all()
                assert torch.all(host[:1] == 7) and torch.all(host[tokens + 1 :] == 7)
                torch.testing.assert_close(observed.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
                assert direct.data_ptr() == direct_pointer and shared_output.data_ptr() == output_pointer

            with torch.cuda.stream(stream):
                source.zero_()
                shared_guard.fill_(7)
                launch_shared()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch_shared()
                launch_shared()
            graphs.append(graph)
            for phase in (0, 4):
                sample_input = _local_expert_sample(tokens, hidden, phase)[0]
                reference = _local_expert_reference(
                    sample_input,
                    torch.zeros((tokens, 1), dtype=torch.int32),
                    torch.ones((tokens, 1), dtype=torch.float32),
                    weights,
                    clamp,
                )
                with torch.cuda.stream(stream):
                    registered.fill_(float("nan"))
                    source.fill_(float("nan"))
                    inputs.copy_(sample_input)
                    ids.zero_()
                    scores.fill_(1)
                    routed_guard.fill_(7)
                    shared_guard.fill_(7)
                    routed_output.fill_(float("nan"))
                    shared_output.fill_(float("nan"))
                    context.forward(inputs, ids, scores, output=routed_output, stream=stream, validate_routing=True)
                    # Leave stale masks/scores behind; forward_shared must ignore them.
                    ids.fill_(-1)
                    scores.fill_(-0.75)
                    context.forward(inputs, ids, scores, output=shared_output, stream=stream, validate_routing=False)
                    shared_output.fill_(float("nan"))
                    result = context.forward_shared(inputs, output=shared_output, stream=stream)
                stream.synchronize()
                assert result is shared_output
                routed_host = routed_guard.cpu()
                expected = routed_host[1 : tokens + 1]
                assert torch.all(routed_host[:1] == 7) and torch.all(routed_host[tokens + 1 :] == 7)
                assert _error_stats(expected, reference)["relative_l2"] < 0.02
                check_shared(expected)
                for _ in range(3):
                    with torch.cuda.stream(stream):
                        registered.fill_(float("nan"))
                        shared_guard.fill_(7)
                        shared_output.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    check_shared(expected)
    finally:
        torch.cuda.synchronize(device)
        graph = None
        graphs.clear()
        gc.collect()
        bootstrap.barrier()
        direct = registered = context = None
        gc.collect()
        bootstrap.barrier()
        communicator = None
        gc.collect()
        bootstrap = None
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_SHARED") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_SHARED=1 to opt into native misaligned-output SM100 tests",
)
@pytest.mark.parametrize("unweighted", [False, True])
def test_native_local_expert_misaligned_contiguous_output_graphs(unweighted):
    torch = pytest.importorskip("torch")
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import MegaMoE, MegaMoEConfig, is_available

    if not is_available() or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires native MegaMoE on SM100")
    device = torch.device("cuda", torch.cuda.current_device())
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    tokens = 33
    hidden = intermediate = 128
    config = MegaMoEConfig(
        rank=0,
        world_size=1,
        max_tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=physical_sms - 32,
        gate_up_clamp=0.125,
    )
    bootstrap = TcpBootstrap.create(0, 1)
    bootstrap.initialize(TcpBootstrap.create_unique_id())
    communicator = Communicator(bootstrap)
    weights = _local_expert_weights(hidden, intermediate, False)
    gpu_weights = [weight.to(device) for weight in weights]
    context = MegaMoE(config, communicator, *gpu_weights)
    view = context.input_view()
    source = torch.zeros((tokens, hidden), device=device, dtype=torch.bfloat16)
    ids = torch.zeros((tokens, 1), device=device, dtype=torch.int32)
    scores = torch.ones((tokens, 1), device=device, dtype=torch.float32)
    storage = torch.full((tokens * hidden + 2,), 7, device=device, dtype=torch.bfloat16)
    output = storage[1:-1].view(tokens, hidden)
    aligned = torch.empty_like(output)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = None
    try:
        assert output.is_contiguous() and output.storage_offset() == 1
        assert output.data_ptr() % 16 == 2 and aligned.data_ptr() % 16 == 0
        output_pointer = output.data_ptr()

        def launch(target):
            view.copy_(source)
            if unweighted:
                return context.forward_shared(view, output=target, stream=stream)
            return context.forward(view, ids, scores, output=target, stream=stream, validate_routing=False)

        def check(expected):
            host = storage.cpu()
            assert host[0] == 7 and host[-1] == 7
            observed = host[1:-1].view(tokens, hidden)
            assert torch.isfinite(observed).all()
            torch.testing.assert_close(observed.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
            assert output.data_ptr() == output_pointer

        with torch.cuda.stream(stream):
            launch(output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch(output)
            launch(output)
        for phase in ((0, 4) if unweighted else (1, 2)):
            sample = _local_expert_sample(tokens, hidden, phase)
            if unweighted:
                sample[1].zero_()
                sample[2].fill_(1)
            reference = _local_expert_reference(*sample, weights, config.gate_up_clamp)
            with torch.cuda.stream(stream):
                source.copy_(sample[0])
                ids.copy_(sample[1])
                scores.copy_(sample[2])
                storage.fill_(7)
                output.fill_(float("nan"))
                launch(aligned)
                result = launch(output)
            stream.synchronize()
            assert result is output
            expected = aligned.cpu()
            assert _error_stats(expected, reference)["relative_l2"] < 0.02
            check(expected)
            for _ in range(3):
                with torch.cuda.stream(stream):
                    storage.fill_(7)
                    output.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                check(expected)
    finally:
        torch.cuda.synchronize(device)
        graph = None
        gc.collect()
        bootstrap.barrier()
        view = context = None
        gc.collect()
        bootstrap.barrier()
        communicator = None
        gc.collect()
        bootstrap = None
        gc.collect()
