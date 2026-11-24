# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU test: MXFP8/MXFP4 expert compute and full pipelines against an independent reference.

Run on two idle Blackwell GPUs:
    python -m torch.distributed.run --standalone --nproc-per-node=2 test_compute_flashinfer.py mscclpp|nccl
"""

import itertools
import sys

import torch
import torch.nn.functional as F

from common import tiny_config
from compute_backends import create_mlp
from compute_backends.flashinfer import fc1_activation, quantize_activations, quantize_weights
from config import SUPPORTED_LAYOUTS
from memory import estimate_tensor_bytes
from reference import (
    compare_outputs,
    decode,
    decoded_expert_weights,
    expert_output,
    quantize_mxfp8,
    relative_error,
    unswizzle,
)
from workload import make_expert_weights


def mx_config():
    return tiny_config()


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
    assert mlp.metadata["fc1_activation"].startswith("swiglu")
    assert mlp.fc1_activation == fc1_activation("swiglu")
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
            expected = expert_output(tokens[expert, :count], weights)
            assert relative_error(actual, expected) < 0.025
            all_act = decode(mlp.activation, mlp.activation_scales, 8, True)
            act = all_act[padded_offset : padded_offset + count]
            first = quantize_mxfp8(tokens[expert, :count]) @ weights[0].T
            chunks = first.reshape(count, -1, 2, 64)
            up, gate = chunks[:, :, 0].reshape(count, -1), chunks[:, :, 1].reshape(count, -1)
            assert relative_error(act, quantize_mxfp8(F.silu(gate) * up)) < 0.015
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
        pipelines = comparisons = 0
        worst = 0.0
        for (mode, layout), graph, routing in itertools.product(
            combinations,
            (False, "mlp", "layer"),
            (
                {"kind": "balanced"},
                {"kind": "zipf_skewed"},
                {"kind": "hot_experts", "expert_ids": [0, 1], "probability_mass": 0.999},
            ),
        ):
            if backend == "nccl" and mode == "throughput" and graph == "layer":
                continue
            config = mx_config()
            config["environment"].update(communication_backend=backend, cuda_graph=graph, mode=mode, layout=layout)
            if backend == "nccl" and mode == "throughput":
                config["input"]["tokens_per_rank"] = 4
            config["routing"] = routing
            pipeline = Pipeline(config, rank, device)
            pipelines += 1
            assert (pipeline.mlp_graph is not None) == (graph == "mlp")
            assert (pipeline.layer_graph is not None) == (graph == "layer")
            mlp = getattr(pipeline.mlp, "mlp", pipeline.mlp)
            if rank == 0:
                print(
                    f"Testing MX {backend}/{mode}/{layout}, graph={graph}, routing={routing['kind']}",
                    flush=True,
                )
            decoded = [
                [decode(w[e], sf[e], 4, True) for w, sf in zip(mlp.weights, mlp.scales)] for e in range(len(mlp.counts))
            ]
            global_decoded = [
                decoded_expert_weights(config["model"], expert, config["benchmark"]["seed"], device)
                for expert in range(config["model"]["num_experts"])
            ]
            previous = None
            # Full-layer graphs replay repeatedly; each replay must consume new input.
            for revision in range(3 if graph == "layer" else 2):
                pipeline.step()
                torch.cuda.synchronize()
                graph_result = pipeline.output.clone()
                if graph == "layer":
                    latencies = pipeline.read_latencies(None)
                    phases = [latencies[key] for key in ("dispatch_us", "mlp_us", "combine_us")]
                    assert all(value > 0 for value in phases)
                    assert abs(sum(phases) - latencies["total_us"]) <= 0.01
                for expert, count in enumerate(pipeline.counts):
                    if count:
                        expected = expert_output(pipeline.recv[expert, :count], decoded[expert])
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
                        routes[token, slot] = expert_output(pipeline.x[token], weights)
                expected = torch.zeros_like(pipeline.x, dtype=torch.float32)
                for destination in range(world if backend == "mscclpp" else 1):
                    partial = torch.zeros_like(expected)
                    for slot in range(config["model"]["top_k"]):
                        weight = pipeline.weights[:, slot]
                        if backend == "mscclpp":
                            weight = weight.masked_fill(pipeline.ids[:, slot] // len(mlp.counts) != destination, 0)
                        partial.addcmul_(routes[:, slot].float(), weight[:, None])
                    expected.add_(partial.bfloat16().float() if backend == "mscclpp" else partial)
                magnitude = sum(
                    routes[:, slot].float().abs() * pipeline.weights[:, slot, None]
                    for slot in range(config["model"]["top_k"])
                )
                stats, first = compare_outputs(graph_result, expected.bfloat16(), magnitude=magnitude)
                assert stats["mismatched_elements"] == stats["non_finite_elements"] == 0, (stats, first)
                worst = max(worst, stats["max_tolerance_fraction"])
                if previous is not None:
                    assert not torch.equal(previous, graph_result)
                previous = graph_result
                comparisons += 1
                pipeline.x.mul_(0.61).add_(0.17)
            if routing["kind"] == "hot_experts":
                assert all(count == 0 for count in pipeline.global_counts[2:])
                assert ((pipeline.ids // len(mlp.counts)) == 0).all()
            dist.barrier()
            pipeline.close()
            pipeline = None
        if rank == 0:
            print(
                f"MX distributed {backend}: {pipelines} pipelines, {comparisons} changed-input comparisons passed; "
                f"worst element used {worst:.1%} of its tolerance",
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
