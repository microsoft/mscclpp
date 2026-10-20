# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routed-only N128 native MegaMoE benchmark; launch with torchrun (see README)."""

import argparse
from dataclasses import replace
import json
import math
import os
import statistics
import time

ROUTING_PATTERNS = (
    "balanced",
    "zipf",
    "zipf_skewed",
    "hot_experts",
)


def _scale_counts(weights, total_routes, max_per_expert):
    """Scale nonnegative expert weights to integer route counts bounded by max_per_expert."""
    n = len(weights)
    if total_routes > n * max_per_expert:
        raise ValueError("total_routes exceeds capacity n * max_per_expert")
    total_w = float(sum(weights))
    if total_w <= 0:
        base, rem = divmod(total_routes, n)
        return [base + (1 if i < rem else 0) for i in range(n)]
    raw = [min(float(max_per_expert), float(w) * total_routes / total_w) for w in weights]
    counts = [min(max_per_expert, int(math.floor(v))) for v in raw]
    # Preserve small nonzero tail experts (at least 16 rows when possible) to match real MoE tails.
    for i, w in enumerate(weights):
        if w > 0 and counts[i] == 0 and sum(counts) + 16 <= total_routes:
            counts[i] = min(16, max_per_expert)
    deficit = total_routes - sum(counts)
    if deficit > 0:
        order = sorted(range(n), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
        while deficit > 0:
            progressed = False
            for i in order:
                if deficit == 0:
                    break
                if counts[i] < max_per_expert and (weights[i] > 0 or all(weights[j] == 0 for j in range(n))):
                    step = min(deficit, max_per_expert - counts[i], 16 if deficit >= 16 else deficit)
                    counts[i] += step
                    deficit -= step
                    progressed = True
            if not progressed:
                for i in order:
                    if deficit == 0:
                        break
                    if counts[i] < max_per_expert:
                        counts[i] += 1
                        deficit -= 1
    elif deficit < 0:
        excess = -deficit
        order = sorted(range(n), key=lambda i: counts[i], reverse=True)
        for i in order:
            if excess == 0:
                break
            take = min(excess, counts[i])
            counts[i] -= take
            excess -= take
    return counts


def _build_global_expert_counts(config, pattern, zipf_alpha, seed):
    import random

    world = config.world_size
    local_e = config.local_experts
    total_tokens = world * config.max_tokens
    total_routes = total_tokens * config.top_k
    # Allow hot experts to exceed single-rank token count (simulating multi-rank aggregation even on EP1).
    max_per_expert = total_routes
    rng = random.Random(seed)

    if pattern == "zipf":
        # Intra-rank Zipfian popularity: each rank has hot, warm, cold/tail, and zero experts.
        weights = []
        for r in range(world):
            rank_w = [
                0.0 if j >= max(1, int(local_e * 0.85)) else 1.0 / ((j + 1) ** zipf_alpha) for j in range(local_e)
            ]
            rng.shuffle(rank_w)
            weights.extend(rank_w)
        return _scale_counts(weights, total_routes, max_per_expert)

    if pattern == "zipf_skewed":
        # Both cross-rank and intra-rank Zipfian skew: rank 0 receives the heaviest load.
        weights = []
        for r in range(world):
            rank_factor = 1.0 / ((r + 1) ** 0.7)
            rank_w = [
                0.0 if j >= max(1, int(local_e * 0.8)) else rank_factor / ((j + 1) ** zipf_alpha)
                for j in range(local_e)
            ]
            rng.shuffle(rank_w)
            weights.extend(rank_w)
        return _scale_counts(weights, total_routes, max_per_expert)

    if pattern == "hot_experts":
        # Bimodal: ~2 hot experts per rank absorb ~65% of routes; ~25% experts empty; rest small partial tiles.
        weights = []
        for r in range(world):
            hot_count = max(1, local_e // 16)
            empty_count = local_e // 4
            rank_w = []
            for j in range(local_e):
                if j < hot_count:
                    rank_w.append(35.0 * (1.0 + 0.3 * (hot_count - j)))
                elif j < hot_count + empty_count:
                    rank_w.append(0.0)
                else:
                    rank_w.append(1.0 + (j % 4) * 0.5)
            rng.shuffle(rank_w)
            weights.extend(rank_w)
        return _scale_counts(weights, total_routes, max_per_expert)

    raise ValueError(f"Unsupported routing pattern: {pattern}")


def _generate_routing(config, pattern, zipf_alpha, seed, device):
    import torch

    if pattern == "balanced":
        torch.manual_seed(seed + config.rank)
        ids = torch.rand((config.max_tokens, config.num_experts), device=device).topk(config.top_k, dim=-1).indices
        ids = ids.to(torch.int32)
        scores = torch.rand((config.max_tokens, config.top_k), device=device, dtype=torch.float32)
        scores /= scores.sum(dim=-1, keepdim=True)
        return ids, scores

    counts = _build_global_expert_counts(config, pattern, zipf_alpha, seed)
    total_tokens = config.world_size * config.max_tokens
    assert sum(counts) <= total_tokens * config.top_k

    # Fill [top_k, total_tokens] in row-major order so each expert is spread evenly across tokens.
    grid = torch.full((config.top_k * total_tokens,), -1, dtype=torch.int32)
    cursor = 0
    for expert, count in enumerate(counts):
        if count > 0:
            grid[cursor : cursor + count] = expert
            cursor += count
    all_ids = grid.view(config.top_k, total_tokens).t().contiguous()

    gen = torch.Generator().manual_seed(seed)
    token_perm = torch.randperm(total_tokens, generator=gen)
    all_ids = all_ids[token_perm].contiguous()

    start = config.rank * config.max_tokens
    ids = all_ids[start : start + config.max_tokens].to(device=device, dtype=torch.int32).contiguous()
    torch.manual_seed(seed + config.rank)
    scores = torch.rand((config.max_tokens, config.top_k), device=device, dtype=torch.float32)
    scores = torch.where(ids >= 0, scores, torch.zeros_like(scores))
    row_sums = scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    scores = torch.where(ids >= 0, scores / row_sums, torch.zeros_like(scores))
    return ids, scores


def _routing_summary(config, ids, tile_n):
    import torch
    import torch.distributed as dist

    host_ids = ids.cpu()
    if config.world_size > 1:
        gathered = [torch.empty_like(host_ids) for _ in range(config.world_size)]
        dist.all_gather(gathered, host_ids)
        all_ids = torch.cat(gathered, dim=0)
    else:
        all_ids = host_ids
    valid = all_ids[all_ids >= 0].long()
    global_counts = (
        torch.bincount(valid, minlength=config.num_experts).tolist() if valid.numel() else [0] * config.num_experts
    )
    first = config.rank * config.local_experts
    local_counts = global_counts[first : first + config.local_experts]
    nonzero = [c for c in local_counts if c > 0]
    received = sum(local_counts)
    blocks = sum((c + tile_n - 1) // tile_n for c in local_counts)
    tail_blocks = sum(1 for c in local_counts if c % tile_n != 0)
    return {
        "received_routes": received,
        "active_local_experts": len(nonzero),
        "zero_local_experts": config.local_experts - len(nonzero),
        "max_expert_routes": max(local_counts, default=0),
        "min_nonzero_expert_routes": min(nonzero, default=0),
        "mean_expert_routes": received / max(1, config.local_experts),
        "token_blocks": blocks,
        "tail_blocks": tail_blocks,
        "padded_slots": blocks * tile_n,
        "padding_overhead_pct": ((blocks * tile_n - received) / max(1, received) * 100.0) if received else 0.0,
        "local_expert_counts": local_counts,
    }


def _weights(config, device):
    import torch
    from .quantization import quantize_mxfp4

    tensors = []
    for m, k in ((2 * config.intermediate, config.hidden), (config.hidden, config.intermediate)):
        values = torch.empty((config.local_experts, m, k // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((config.local_experts, m, k // 32), dtype=torch.uint8, device=device)
        for expert in range(config.local_experts):
            source = torch.randn((m, k), dtype=torch.float32, device=device) / k**0.5
            quantized, scale = quantize_mxfp4(source)
            values[expert].copy_(quantized)
            scales[expert].copy_(scale)
        tensors.extend((values, scales))
    return tensors


def _reference(config, inputs, ids, scores, weights):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from .quantization import dequantize_mxfp4, dequantize_mxfp8, quantize_mxfp8

    def gather(tensor):
        if config.world_size == 1:
            return tensor, 0
        host = tensor.cpu()
        gathered = [None] * config.world_size
        dist.all_gather_object(gathered, host)
        offset = sum(item.shape[0] for item in gathered[: config.rank])
        return torch.cat(gathered).to(inputs.device), offset

    (all_inputs, first_token), (all_ids, _), (all_scores, _) = gather(inputs), gather(ids), gather(scores)
    partial = torch.zeros((all_inputs.shape[0], config.top_k, config.hidden), device=inputs.device, dtype=torch.float32)
    fc1, fc1_scale, fc2, fc2_scale = weights
    for expert in range(config.local_experts):
        token, slot = torch.where(all_ids == config.rank * config.local_experts + expert)
        if not token.numel():
            continue
        first = dequantize_mxfp4(fc1[expert], fc1_scale[expert], dtype=torch.float32)
        second = dequantize_mxfp4(fc2[expert], fc2_scale[expert], dtype=torch.float32)
        expert_input = dequantize_mxfp8(*quantize_mxfp8(all_inputs[token].float()), dtype=torch.float32)
        gate, up = (expert_input @ first.T).chunk(2, dim=-1)
        if config.gate_up_clamp >= 0:
            gate = gate.clamp(max=config.gate_up_clamp)
            up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
        hidden = functional.silu(gate) * up
        hidden = dequantize_mxfp8(*quantize_mxfp8(hidden), dtype=torch.float32)
        result = (hidden @ second.T) * all_scores[token, slot, None]
        partial[token, slot] = result.to(torch.bfloat16).float()
    host_partial = partial.cpu()
    if config.world_size > 1:
        dist.all_reduce(host_partial)
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
    parser.add_argument("--input-mode", choices=("staged", "direct"), default="staged")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-batch", type=int, default=10, help="collectives captured per replay")
    parser.add_argument(
        "--graph-timing",
        choices=("isolated", "steady-state"),
        default="isolated",
        help="steady-state queues an untimed replay before timing to exclude initial rank submission skew",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cache-dir", default=None, help="local native kernel JIT cache")
    parser.add_argument(
        "--w4-profile",
        nargs="?",
        const="",
        default=None,
        metavar="JSON",
        help="JIT the exact W4A8 policy from JSON (omit JSON to use the packaged profile)",
    )
    parser.add_argument(
        "--routing-pattern",
        choices=ROUTING_PATTERNS,
        default="balanced",
        help="token-to-expert routing distribution (balanced, zipf, zipf_skewed, hot_experts)",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.2, help="Zipf exponent for zipf/zipf_skewed patterns")
    parser.add_argument("--tile-k", type=int, default=None, help="override W4A8KernelConfig.tile_k")
    parser.add_argument("--load-stages", type=int, default=None, help="override W4A8KernelConfig.load_stages")
    parser.add_argument("--num-warps", type=int, default=None, help="override W4A8KernelConfig.num_warps")
    parser.add_argument(
        "--transfer-registers", type=int, default=None, help="override W4A8KernelConfig.transfer_registers"
    )
    parser.add_argument("--load-warps", type=int, default=None, help="override W4A8KernelConfig.load_warps")
    parser.add_argument(
        "--split-pipelines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override W4A8KernelConfig.split_pipelines",
    )
    parser.add_argument("--epilogue-tokens", type=int, default=None, help="override W4A8KernelConfig.epilogue_tokens")
    parser.add_argument("--epilogue-warps", type=int, default=None, help="override W4A8KernelConfig.epilogue_warps")
    parser.add_argument(
        "--epilogue-registers", type=int, default=None, help="override W4A8KernelConfig.epilogue_registers"
    )
    parser.add_argument("--dispatch-chunk", type=int, default=None, help="override W4A8KernelConfig.dispatch_chunk")
    parser.add_argument("--dispatch-warps", type=int, default=None, help="override W4A8KernelConfig.dispatch_warps")
    parser.add_argument("--dispatch-stages", type=int, default=None, help="override W4A8KernelConfig.dispatch_stages")
    parser.add_argument(
        "--fixed-token-count",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override W4A8KernelConfig.fixed_token_count",
    )
    parser.add_argument(
        "--expert-major-tasks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override W4A8KernelConfig.expert_major_tasks",
    )
    parser.add_argument(
        "--cache-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override W4A8KernelConfig.cache_weights",
    )
    parser.add_argument(
        "--adaptive-cache-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override W4A8KernelConfig.adaptive_cache_weights",
    )
    parser.add_argument("--cluster-size", type=int, default=None, help="override W4A8KernelConfig.cluster_size")
    parser.add_argument("--check", action="store_true", help="compare against an independent Torch reference")
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap-port", type=int, default=None, help="MSCCL++ port (default MASTER_PORT+1)")
    parser.add_argument("--json-output", default=None, help="optional output path")
    args = parser.parse_args()
    if min(args.tokens, args.graph_batch, args.warmup, args.iterations) < 1:
        parser.error("tokens, graph-batch, warmup, and iterations must be positive")
    if (args.hidden, args.intermediate) not in ((4096, 6656), (8192, 4096), (9216, 4096), (9216, 4608)):
        parser.error("W4A8 H/I must be 4096/6656, 8192/4096, 9216/4096, or 9216/4608")
    if not args.graph and args.graph_timing != "isolated":
        parser.error("--graph-timing steady-state requires --graph")

    import torch
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from .api import MegaMoE, MegaMoEConfig, is_available
    from .jit import W4A8KernelConfig, compile_kernel
    from .w4a8 import resolve_w4a8_kernel_config

    if not is_available():
        parser.error("MSCCL++ lacks native MegaMoE: build with MSCCLPP_BUILD_EXT_MEGAMOE=ON")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not torch.cuda.is_available():
        parser.error("native MegaMoE requires SM100-family CUDA GPUs")
    torch.cuda.set_device(local_rank)
    config = MegaMoEConfig(
        rank=rank,
        world_size=world,
        max_tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.experts,
        top_k=args.top_k,
        sm_margin=args.sm_margin,
        gate_up_clamp=args.gate_up_clamp,
    )
    kernel_config = W4A8KernelConfig()
    if args.w4_profile is not None:
        kernel_config = resolve_w4a8_kernel_config(config, profile_path=args.w4_profile or None)
    policy_overrides = {
        name: getattr(args, name)
        for name in (
            "tile_k",
            "load_stages",
            "num_warps",
            "transfer_registers",
            "load_warps",
            "split_pipelines",
            "epilogue_tokens",
            "epilogue_warps",
            "epilogue_registers",
            "dispatch_chunk",
            "dispatch_warps",
            "dispatch_stages",
            "fixed_token_count",
            "expert_major_tasks",
            "cache_weights",
            "adaptive_cache_weights",
            "cluster_size",
        )
        if getattr(args, name) is not None
    }
    if policy_overrides:
        kernel_config = replace(kernel_config, **policy_overrides)
    kernel = compile_kernel(kernel_config, cache_dir=args.cache_dir)
    torch.manual_seed(args.seed + rank)
    # Gloo is used only for rendezvous, reporting, and the untimed reference.
    # All timed expert communication uses the native MSCCL++ CudaIpc mappings.
    dist.init_process_group("gloo", rank=rank, world_size=world)
    bootstrap = TcpBootstrap.create(rank, world)
    port = args.bootstrap_port or int(os.environ["MASTER_PORT"]) + 1
    bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
    communicator = Communicator(bootstrap)
    device = torch.device("cuda", local_rank)
    weights = _weights(config, device)
    start = time.perf_counter()
    context = MegaMoE(config, communicator, *weights, kernel=kernel)
    initialization_ms = (time.perf_counter() - start) * 1000
    inputs = torch.randn((args.tokens, args.hidden), device=device, dtype=torch.bfloat16)
    ids, scores = _generate_routing(config, args.routing_pattern, args.zipf_alpha, args.seed, device)
    routing_summary = _routing_summary(config, ids, kernel_config.tile_n)
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
            if graph and args.graph_timing == "steady-state":
                graph.replay()
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
            "kernel_id": context.kernel_id,
            "kernel_config": context.effective_kernel_config,
            "routing": routing_summary,
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
            "timing": {
                "method": "CUDA events; slowest rank per sample",
                "graph_timing": args.graph_timing if graph else None,
                "untimed_graph_replays_per_sample": int(bool(graph) and args.graph_timing == "steady-state"),
            },
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
