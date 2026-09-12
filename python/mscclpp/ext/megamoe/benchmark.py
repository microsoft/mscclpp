# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routed-only native MegaMoE benchmark; launch with torchrun (see README)."""

import argparse
import json
import os
import statistics
import time


def _weights(config, device):
    import torch
    from .quantization import quantize_mxfp8

    dtype = torch.float8_e5m2 if config.weight_e5m2 else torch.float8_e4m3fn
    tensors = []
    for m, k in ((2 * config.intermediate, config.hidden), (config.hidden, config.intermediate)):
        values = torch.empty((config.local_experts, m, k), dtype=dtype, device=device)
        scales = torch.empty((config.local_experts, m, k // 32), dtype=torch.uint8, device=device)
        for expert in range(config.local_experts):
            source = torch.randn((m, k), dtype=torch.float32, device=device) / k**0.5
            quantized, scale = quantize_mxfp8(source, e5m2=config.weight_e5m2)
            values[expert].copy_(quantized)
            scales[expert].copy_(scale)
        tensors.extend((values, scales))
    return tensors


def _reference(config, inputs, ids, scores, weights):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from .quantization import dequantize_mxfp8

    def gather(tensor):
        host = tensor.cpu()
        gathered = [torch.empty_like(host) for _ in range(config.world_size)]
        dist.all_gather(gathered, host)
        return torch.cat(gathered).to(inputs.device)

    all_inputs, all_ids, all_scores = gather(inputs), gather(ids), gather(scores)
    partial = torch.zeros((all_inputs.shape[0], config.top_k, config.hidden), device=inputs.device, dtype=torch.float32)
    fc1, fc1_scale, fc2, fc2_scale = weights
    for expert in range(config.local_experts):
        token, slot = torch.where(all_ids == config.rank * config.local_experts + expert)
        if not token.numel():
            continue
        first = dequantize_mxfp8(fc1[expert], fc1_scale[expert]).float()
        second = dequantize_mxfp8(fc2[expert], fc2_scale[expert]).float()
        gate, up = (all_inputs[token].float() @ first.T).chunk(2, dim=-1)
        if config.gate_up_clamp >= 0:
            gate = gate.clamp(max=config.gate_up_clamp)
            up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
        hidden = (functional.silu(gate) * up * all_scores[token, slot, None]).to(torch.bfloat16)
        partial[token, slot] = (hidden.float() @ second.T).to(torch.bfloat16).float()
    host_partial = partial.cpu()
    dist.all_reduce(host_partial)
    first_token = config.rank * inputs.shape[0]
    return host_partial[first_token : first_token + inputs.shape[0]].sum(dim=1).to(inputs.device, torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="tokens per rank")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4352, help="post-SwiGLU intermediate width")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--sm-margin", type=int, default=32)
    parser.add_argument("--gate-up-clamp", type=float, default=-1.0)
    parser.add_argument("--e5m2", action="store_true")
    parser.add_argument("--input-mode", choices=("staged", "direct"), default="staged")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-batch", type=int, default=10, help="collectives captured per replay")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--check", action="store_true", help="compare against an independent Torch reference")
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap-port", type=int, default=None, help="MSCCL++ port (default MASTER_PORT+1)")
    parser.add_argument("--json-output", default=None, help="optional output path")
    args = parser.parse_args()
    if min(args.tokens, args.graph_batch, args.warmup, args.iterations) < 1:
        parser.error("tokens, graph-batch, warmup, and iterations must be positive")

    import torch
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from .api import MegaMoE, MegaMoEConfig, is_available

    if not is_available():
        parser.error("MSCCL++ lacks native MegaMoE: build with MSCCLPP_BUILD_EXT_MEGAMOE=ON")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not torch.cuda.is_available():
        parser.error("native MegaMoE requires SM100 CUDA GPUs")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    # Gloo is used only for rendezvous, reporting, and the untimed reference.
    # All timed expert communication uses the native MSCCL++ CudaIpc mappings.
    dist.init_process_group("gloo", rank=rank, world_size=world)
    bootstrap = TcpBootstrap.create(rank, world)
    port = args.bootstrap_port or int(os.environ["MASTER_PORT"]) + 1
    bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
    communicator = Communicator(bootstrap)
    config = MegaMoEConfig(
        rank=rank,
        world_size=world,
        max_tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.experts,
        top_k=args.top_k,
        sm_margin=args.sm_margin,
        weight_e5m2=args.e5m2,
        gate_up_clamp=args.gate_up_clamp,
    )
    device = torch.device("cuda", local_rank)
    weights = _weights(config, device)
    start = time.perf_counter()
    context = MegaMoE(config, communicator, *weights)
    initialization_ms = (time.perf_counter() - start) * 1000
    inputs = torch.randn((args.tokens, args.hidden), device=device, dtype=torch.bfloat16)
    ids = torch.rand((args.tokens, args.experts), device=device).topk(args.top_k, dim=-1).indices.to(torch.int32)
    scores = torch.rand((args.tokens, args.top_k), device=device, dtype=torch.float32)
    scores /= scores.sum(dim=-1, keepdim=True)
    if args.input_mode == "direct":
        direct = context.input_view(args.tokens)
        direct.copy_(inputs)
        inputs = direct
    output = torch.empty_like(inputs)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def launch():
        context(inputs, ids, scores, output=output, stream=stream)

    for _ in range(args.warmup):
        launch()
    stream.synchronize()
    correctness = None
    if args.check:
        reference = _reference(config, inputs, ids, scores, weights)
        error = (output.float() - reference.float()).abs()
        correctness = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
        torch.testing.assert_close(output, reference, rtol=args.rtol, atol=args.atol)
    del weights
    graph = None
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.graph_batch):
                launch()
        graph.replay()
        torch.cuda.synchronize(device)
        if args.check:
            torch.testing.assert_close(output, reference, rtol=args.rtol, atol=args.atol)
    samples = []
    repetitions = args.graph_batch if graph else 1
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        dist.barrier()
        with torch.cuda.stream(stream):
            begin.record()
            graph.replay() if graph else launch()
            end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / repetitions)
    reports = [None] * world
    dist.all_gather_object(
        reports,
        {
            "rank": rank,
            "cta_count": context.cta_count,
            "workspace_bytes": context.workspace_bytes,
            "initialization_ms": initialization_ms,
            "samples_us": samples,
            "correctness": correctness,
        },
    )
    if rank == 0:
        slowest = [max(report["samples_us"][sample] for report in reports) for sample in range(args.iterations)]
        result = {
            "implementation": "mscclpp-native-cuda-megamoe",
            "scope": "routed_experts_only",
            "excluded": ["shared_expert", "squash", "unsquash", "router", "residual"],
            "configuration": {**vars(args), "world_size": world},
            "hardware": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "latency_us_max_across_ranks": {
                "median": statistics.median(slowest),
                "mean": statistics.mean(slowest),
                "minimum": min(slowest),
                "maximum": max(slowest),
            },
            "ranks": reports,
        }
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.json_output:
            with open(args.json_output, "w") as output_file:
                output_file.write(text + "\n")
    # Release only after every GPU has completed every peer access.
    torch.cuda.synchronize(device)
    bootstrap.barrier()
    del graph, context, inputs
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
