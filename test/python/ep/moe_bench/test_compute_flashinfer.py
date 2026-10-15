# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional MX integration tests, with independent scale/nibble and math references."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import pytest
import torch
import torch.nn.functional as F
import yaml

if __package__:
    from .compute import create_mlp, estimate_tensor_bytes, make_expert_weights
    from .compute_flashinfer import fc1_weight, make_schedule, preflight, quantize_activations, quantize_weights
    from .config import SUPPORTED_LAYOUTS, load_config, validate_config
    from .test_moe_bench import assert_idle_gpus, tiny_config
else:
    from compute import create_mlp, estimate_tensor_bytes, make_expert_weights
    from compute_flashinfer import fc1_weight, make_schedule, preflight, quantize_activations, quantize_weights
    from config import SUPPORTED_LAYOUTS, load_config, validate_config
    from test_moe_bench import assert_idle_gpus, tiny_config


def mx_config():
    config = tiny_config()
    config["model"]["intermediate_size"] = 128
    config["environment"]["compute_backend"] = "cutlass_flashinfer"
    config["compute"] = {
        "activation_dtype": "mxfp8",
        "weight_dtype": "mxfp4",
    }
    return config


@pytest.mark.parametrize(
    "backend,activation,weight",
    list(
        itertools.product(("torch", "cutlass_flashinfer"), ("bf16", "mxfp8", "fp8"), ("bf16", "mxfp8", "mxfp4", "fp4"))
    ),
)
def test_config_pairs(backend, activation, weight):
    config = mx_config()
    config["environment"]["compute_backend"] = backend
    config["compute"] = {"activation_dtype": activation, "weight_dtype": weight}
    valid = (backend, activation, weight) in (
        ("torch", "bf16", "bf16"),
        ("cutlass_flashinfer", "mxfp8", "mxfp4"),
    )
    if valid:
        validate_config(config)
    else:
        with pytest.raises(ValueError):
            validate_config(config)


def test_weight_only_rejected():
    config = mx_config()
    config["compute"] = {"activation_dtype": "bf16", "weight_dtype": "mxfp8"}
    with pytest.raises(ValueError, match="BF16/MXFP8 weight-only compute.*not supported"):
        validate_config(config)
    with pytest.raises(ValueError, match="requires MXFP8 activations and MXFP4 weights"):
        preflight(config, "cpu")


def test_optional_compute_defaults_and_overrides():
    config = tiny_config()
    del config["compute"]
    validate_config(config)
    assert tiny_config()["compute"] == {"activation_dtype": "bf16", "weight_dtype": "bf16"}
    document = yaml.safe_load(Path(__file__).with_name("configs.yaml").read_text())
    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(document))):
        resolved = load_config(
            "unused",
            model="small",
            input_name="decode",
            routing="balanced",
            environment_overrides={"compute_backend": "cutlass_flashinfer"},
            compute_overrides={"activation_dtype": "mxfp8", "weight_dtype": "mxfp4"},
        )
    assert resolved["compute"] == {"activation_dtype": "mxfp8", "weight_dtype": "mxfp4"}
    assert resolved["input"]["dtype"] == "bf16"
    for compute in ({"dtype": "mxfp8"}, [], None):
        config["compute"] = compute
        with pytest.raises(ValueError):
            validate_config(config)
    for malformed in (None, [], "config"):
        with pytest.raises(ValueError, match="config must be a mapping"):
            validate_config(malformed)


@pytest.mark.parametrize("dimension", ["d_model", "intermediate_size"])
def test_alignment_rejected(dimension):
    config = mx_config()
    config["model"][dimension] = 4097 if dimension == "d_model" else 129
    with pytest.raises(ValueError):
        validate_config(config)


def test_cli_compute_flags():
    from .run import parse_args

    base = ["--model", "small", "--input", "decode", "--routing", "balanced"]
    args = parse_args(base)
    assert args.compute_backend is args.activation_dtype is args.weight_dtype is None
    args = parse_args(
        base + ["--compute-backend", "cutlass_flashinfer", "--activation-dtype", "mxfp8", "--weight-dtype", "mxfp4"]
    )
    assert (args.compute_backend, args.activation_dtype, args.weight_dtype) == ("cutlass_flashinfer", "mxfp8", "mxfp4")
    for flag, value in (
        ("--compute-backend", "flashinfer"),
        ("--activation-dtype", "fp8"),
        ("--weight-dtype", "fp4"),
        ("--weight-dtype", "mxfp8"),
    ):
        with pytest.raises(SystemExit):
            parse_args(base + [flag, value])


@pytest.mark.parametrize("backend", ["mscclpp", "nccl"])
def test_launcher_compute_overrides(tmp_path, monkeypatch, backend):
    from . import compute_flashinfer, run as launcher

    config = mx_config()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "small",
            "--input",
            "decode",
            "--routing",
            "balanced",
            "--backend",
            backend,
            "--compute-backend",
            "cutlass_flashinfer",
            "--activation-dtype",
            config["compute"]["activation_dtype"],
            "--weight-dtype",
            config["compute"]["weight_dtype"],
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    checked = Mock()
    monkeypatch.setattr(compute_flashinfer, "preflight", checked)

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="revision\n" if command[-1] == "HEAD" else "")
        path = Path(command[-1])
        resolved = json.loads((path / "run.json").read_text())["config"]
        assert resolved["environment"]["compute_backend"] == "cutlass_flashinfer"
        assert resolved["environment"]["communication_backend"] == backend
        assert resolved["compute"] == config["compute"]
        assert resolved["input"]["dtype"] == "bf16"
        (path / "summary.csv").write_text("")
        (path / "iterations.csv").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.main()
    assert checked.call_count == 4


@pytest.mark.parametrize("stage", ["compile", "capture"])
def test_initialization_failure_coordination_and_cleanup(monkeypatch, stage):
    from . import worker

    config = tiny_config()
    monkeypatch.setattr(worker.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(worker, "preflight_backend", Mock())
    monkeypatch.setattr(worker.dist, "all_reduce", lambda value, **kwargs: None)

    def gather(values, value):
        values[:] = [value, value]

    monkeypatch.setattr(worker.dist, "all_gather_object", gather)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**40, 2**40))
    backend = SimpleNamespace(close=Mock())
    factory = Mock(return_value=backend)
    monkeypatch.setattr(worker, "create_backend", factory)
    if stage == "compile":
        monkeypatch.setattr(worker, "create_mlp", Mock(side_effect=RuntimeError("compiler failure")))
    else:
        monkeypatch.setattr(worker.Pipeline, "_initialize_compute", Mock(side_effect=RuntimeError("capture failure")))
    with pytest.raises(RuntimeError, match="(Runtime/compute|Compute) .*failed"):
        worker.Pipeline(config, 0, "cpu")
    assert factory.call_count == (stage == "capture")
    assert backend.close.call_count == (stage == "capture")


def test_torch_has_no_optional_compute_imports():
    source = """
import builtins
import torch
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'flashinfer', 'cutlass', 'cuda', 'mscclpp', 'nccl'}:
        raise AssertionError('Unexpected optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from compute import create_mlp
from test_moe_bench import tiny_config
config = tiny_config()
config['model']['d_model'] = 16
create_mlp(config, 0, [1, 0, 1, 0], 'cpu')
import worker
"""
    subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).parent, check=True, timeout=30)


def test_preflight_rejects_versions_and_devices(monkeypatch):
    from . import compute_flashinfer as module

    monkeypatch.setattr(module, "version", lambda name: "wrong")
    with pytest.raises(RuntimeError, match="requires tested"):
        preflight(mx_config(), 0)
    versions = {
        "flashinfer-python": "0.7.1",
        "nvidia-cutlass-dsl": "4.8.0",
    }
    monkeypatch.setattr(module, "version", versions.__getitem__)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    assert preflight(mx_config(), 0) == versions
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="SM100/SM103"):
        preflight(mx_config(), 0)
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(module, "version", Mock(side_effect=PackageNotFoundError("flashinfer-python")))
    with pytest.raises(RuntimeError, match="requires optional flashinfer-python"):
        preflight(mx_config(), 0)


def test_schedule_and_weight_orientation():
    counts = [7, 0, 129]
    slots, experts, limits, tiles = make_schedule(counts, "cpu")
    valid = slots[slots >= 0]
    assert valid.tolist() == list(range(sum(counts)))
    assert 1 not in experts.tolist()
    assert tiles.item() == len(experts) == 3
    assert len(slots) == 384
    assert limits[-1] == 128 + 129
    gate = torch.arange(256 * 128).reshape(256, 128)
    up = -gate
    packed = fc1_weight(gate, up)
    for column in range(128):
        up_row = (column // 64) * 128 + column % 64
        gate_row = up_row + 64
        assert torch.equal(packed[up_row], up[:, column])
        assert torch.equal(packed[gate_row], gate[:, column])


def test_memory_estimate():
    config = mx_config()
    config["model"]["num_experts"] = 128
    config["model"]["intermediate_size"] = 8192
    counts = [3] * 64
    estimate = estimate_tensor_bytes(config, counts)
    elements = 3 * 64 * 4096 * 8192
    packed_weights = elements // 2 + elements // 32
    assert estimate > packed_weights + 6 * 4096 * 8192
    assert estimate < elements * 2
    config["environment"]["cuda_graph"] = "mlp"
    assert estimate_tensor_bytes(config, counts) > estimate


def unswizzle(sf, rows, k):
    return (
        sf.reshape(-1)
        .view((rows + 127) // 128, (k // 32 + 3) // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape((rows + 127) // 128 * 128, (k // 32 + 3) // 4 * 4)[:rows, : k // 32]
    )


def decode(q, sf, bits, swizzled=False):
    """Independent byte decoding, not a FlashInfer dequantization call."""
    rows, cols = q.shape
    k = cols * (2 if bits == 4 else 1)
    codes = unswizzle(sf, rows, k) if swizzled else sf.reshape(rows, k // 32)
    scales = torch.exp2(codes.to(torch.int32).float() - 127)
    if bits == 4:
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=q.device)
        values = lut[torch.stack((q & 15, q >> 4), dim=-1).reshape(rows, k).long()]
    else:
        values = q.float()
    return (values.reshape(rows, k // 32, 32) * scales[:, :, None]).reshape(rows, k)


def reference_quant8(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    maximum = blocks.abs().amax(-1)
    exponent = torch.ceil(torch.log2((maximum / 448).clamp_min(2.0**-126)))
    scale = torch.exp2(exponent)
    values = (blocks / scale[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (values * scale[:, :, None]).reshape_as(x)


def relative_error(actual, expected):
    assert torch.isfinite(actual).all()
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


@torch.inference_mode()
def layout_reference(device):
    rows, k = 129, 256
    exponent = (torch.arange(rows, device=device)[:, None] + 3 * torch.arange(k // 32, device=device)[None, :]) % 11 - 5
    for quantizer, bits, maximum, swizzled in (
        (quantize_weights, 4, 6.0, True),
        (quantize_activations, 8, 448.0, False),
    ):
        values = torch.zeros(rows, k // 32, 32, device=device)
        values[:, :, 0], values[:, :, 1], values[:, :, 2] = maximum, -maximum, maximum / 2
        values *= torch.exp2(exponent.float())[:, :, None]
        x = values.reshape(rows, k).bfloat16()
        q, sf = quantizer(x)
        codes = unswizzle(sf, rows, k) if swizzled else sf.reshape(rows, k // 32)
        assert torch.equal(codes, (exponent + 127).to(torch.uint8))
        assert torch.equal(decode(q, sf, bits, swizzled), x.float())
        q, sf = quantizer(torch.zeros_like(x))
        assert not decode(q, sf, bits, swizzled).count_nonzero()


def projection_reference(x, weights):
    first, down = weights
    proj = reference_quant8(x) @ first.T
    chunks = proj.reshape(x.shape[0], -1, 2, 64)
    up, gate = chunks[:, :, 0].reshape(x.shape[0], -1), chunks[:, :, 1].reshape(x.shape[0], -1)
    act = reference_quant8(F.silu(gate) * up)
    return (act @ down.T).bfloat16()


@torch.inference_mode()
def multitile_reference(rank, device):
    config = mx_config()
    config["model"]["num_experts"] = 6
    config["input"]["tokens_per_rank"] = 65
    counts = [7, 0, 129]
    free, _ = torch.cuda.mem_get_info(device)
    assert estimate_tensor_bytes(config, counts) < free * 0.8
    mlp = create_mlp(config, rank, counts, device)
    assert mlp.metadata["compute_implementation"] == "cute_native_mxfp8_mxfp4"
    assert mlp.metadata["native_mixed_mma"]
    assert mlp.metadata["initial_activation_quantization_timed"]
    assert mlp.metadata["communication_dtype"] == "bf16"
    generator = torch.Generator(device=device).manual_seed(8107 + rank)
    tokens = torch.randn(3, 130, mlp.hidden, device=device, dtype=torch.bfloat16, generator=generator)
    out = torch.full_like(tokens, float("nan"))
    mlp(tokens, out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mlp(tokens, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        mlp(tokens, out)
    maxima = {"arithmetic_relative_l2": 0.0, "total_quantization_relative_l2": 0.0}
    for _ in range(2):
        tokens.mul_(0.61).add_(0.17)
        graph.replay()
        captured = out.clone()
        mlp(tokens, out)
        padded_offset = 0
        for expert, count in enumerate(counts):
            if not count:
                assert torch.isnan(out[expert]).all()
                continue
            actual = out[expert, :count]
            assert torch.equal(actual, captured[expert, :count])
            assert torch.isnan(out[expert, count:]).all()
            weights = [decode(w[expert], sf[expert], 4, True) for w, sf in zip(mlp.weights, mlp.scales)]
            expected = projection_reference(tokens[expert, :count], weights)
            assert relative_error(actual, expected) < 0.025
            all_act = decode(mlp.activation, mlp.activation_scales, 8, True)
            act = all_act[padded_offset : padded_offset + count]
            first = reference_quant8(tokens[expert, :count]) @ weights[0].T
            chunks = first.reshape(count, -1, 2, 64)
            up, gate = chunks[:, :, 0].reshape(count, -1), chunks[:, :, 1].reshape(count, -1)
            assert relative_error(act, reference_quant8(F.silu(gate) * up)) < 0.015
            padded_offset += (count + 127) // 128 * 128
            arithmetic = relative_error(actual, act @ weights[1].T)
            assert arithmetic < 0.004
            maxima["arithmetic_relative_l2"] = max(maxima["arithmetic_relative_l2"], arithmetic)
            gate, up, down = make_expert_weights(
                config["model"], rank * 3 + expert, config["benchmark"]["seed"], device
            )
            x = tokens[expert, :count].float()
            unquantized = (F.silu(x @ gate.float()) * (x @ up.float())) @ down.float()
            error = relative_error(actual, unquantized)
            # This loose sanity bound is NOT an arithmetic tolerance or a claim
            # about trained-model accuracy. Quantization error is reported separately.
            assert error < 0.4
            maxima["total_quantization_relative_l2"] = max(maxima["total_quantization_relative_l2"], error)
    print(f"rank {rank} MXFP8/MXFP4 multitile/graph metrics: {maxima}", flush=True)


@pytest.mark.parametrize("backend", ["mscclpp", "nccl"])
def test_distributed_mx_reference(backend):
    if os.environ.get("MOE_BENCH_MX_GPU_TESTS") != "1":
        pytest.skip("Set MOE_BENCH_MX_GPU_TESTS=1 on idle SM100/SM103 GPUs")
    assert_idle_gpus()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            backend,
        ],
        check=True,
        timeout=900,
    )


@torch.inference_mode()
def distributed_reference(backend):
    import torch.distributed as dist
    from worker import Pipeline, initialize

    rank, world, device = initialize()
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = None
    try:
        layout_reference(device)
        multitile_reference(rank, device)
        combinations = [(mode, layout) for mode, layouts in SUPPORTED_LAYOUTS[backend].items() for layout in layouts]
        for (mode, layout), graph, routing in itertools.product(
            combinations,
            (False, "mlp"),
            (
                {"kind": "balanced"},
                {"kind": "uniform"},
                {"kind": "hot_experts", "expert_ids": [0, 1], "probability_mass": 0.999},
            ),
        ):
            config = mx_config()
            config["environment"].update(communication_backend=backend, cuda_graph=graph, mode=mode, layout=layout)
            if backend == "nccl" and mode == "throughput":
                config["input"]["tokens_per_rank"] = 4
            config["routing"] = routing
            pipeline = Pipeline(config, rank, device)
            mlp = getattr(pipeline.mlp, "mlp", pipeline.mlp)
            if rank == 0:
                print(
                    f"Testing MX {backend}/{mode}/{layout}, graph={graph}, routing={routing['kind']}",
                    flush=True,
                )
            decoded = [
                [decode(w[e], sf[e], 4, True) for w, sf in zip(mlp.weights, mlp.scales)] for e in range(len(mlp.counts))
            ]
            global_decoded = []
            for expert in range(config["model"]["num_experts"]):
                raw = make_expert_weights(config["model"], expert, config["benchmark"]["seed"], device)
                # Build weight layout independently of production fc1_weight.
                up, gate = raw[1].T, raw[0].T
                first = torch.empty(2 * mlp.inter, mlp.hidden, device=device, dtype=torch.bfloat16)
                for start in range(0, mlp.inter, 64):
                    first[2 * start : 2 * start + 64] = up[start : start + 64]
                    first[2 * start + 64 : 2 * start + 128] = gate[start : start + 64]
                global_decoded.append([decode(*quantize_weights(w), 4, True) for w in (first, raw[2].T)])
            previous = None
            for revision in range(2):
                pipeline.step()
                torch.cuda.synchronize()
                graph_result = pipeline.output.clone()
                for expert, count in enumerate(pipeline.counts):
                    if count:
                        expected = projection_reference(pipeline.recv[expert, :count], decoded[expert])
                        assert relative_error(pipeline.expert_output[expert, :count], expected) < 0.025
                # Replay and eager must consume the same new BF16 dispatch data.
                eager_experts = torch.empty_like(pipeline.expert_output)
                mlp(pipeline.recv, eager_experts)
                for expert, count in enumerate(pipeline.counts):
                    assert torch.equal(pipeline.expert_output[expert, :count], eager_experts[expert, :count])
                routes = torch.zeros(
                    config["input"]["tokens_per_rank"],
                    config["model"]["top_k"],
                    mlp.hidden,
                    device=device,
                    dtype=torch.bfloat16,
                )
                for expert, weights in enumerate(global_decoded):
                    token, slot = (pipeline.ids == expert).nonzero(as_tuple=True)
                    if len(token):
                        routes[token, slot] = projection_reference(pipeline.x[token], weights)
                expected = torch.zeros_like(pipeline.x, dtype=torch.float32)
                for destination in range(world if backend == "mscclpp" else 1):
                    partial = torch.zeros_like(expected)
                    for slot in range(config["model"]["top_k"]):
                        weight = pipeline.weights[:, slot]
                        if backend == "mscclpp":
                            weight = weight.masked_fill(pipeline.ids[:, slot] // len(mlp.counts) != destination, 0)
                        partial.addcmul_(routes[:, slot].float(), weight[:, None])
                    expected.add_(partial.bfloat16().float() if backend == "mscclpp" else partial)
                assert relative_error(graph_result, expected.bfloat16()) < 0.03
                if previous is not None:
                    assert not torch.equal(previous, graph_result)
                previous = graph_result
                pipeline.x.mul_(0.61).add_(0.17)
            if routing["kind"] == "hot_experts":
                assert all(count == 0 for count in pipeline.global_counts[2:])
                assert ((pipeline.ids // len(mlp.counts)) == 0).all()
            dist.barrier()
            pipeline.close()
            pipeline = None
        if rank == 0:
            print(
                f"MX distributed {backend}: {6 * len(combinations)} pipelines, "
                f"{12 * len(combinations)} changed-input comparisons passed",
                flush=True,
            )
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    distributed_reference(sys.argv[1])
