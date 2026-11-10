# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native routed and optional overlapped-shared MegaMoE benchmark; launch with torchrun."""

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
    "lognormal",
)


def _latency_summary(reports, key, iterations):
    slowest = [max(report[key][sample] for report in reports) for sample in range(iterations)]
    return {
        "median": statistics.median(slowest),
        "mean": statistics.mean(slowest),
        "minimum": min(slowest),
        "maximum": max(slowest),
    }


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


def build_expert_popularity(config, pattern, zipf_alpha, seed, *, lognormal_sigma=0.85):
    """Return unrounded expert popularity for traffic generators."""
    import random

    world = config.world_size
    local_e = config.local_experts
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
        return weights

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
        return weights

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
        return weights

    if pattern == "lognormal":
        # Global i.i.d. LogNormal(0, sigma) expert popularity across all ranks (sigma=0.85 matches c1024-decode-p50).
        if not math.isfinite(lognormal_sigma) or lognormal_sigma < 0:
            raise ValueError("lognormal_sigma must be a finite nonnegative number")
        weights = [math.exp(rng.gauss(0.0, lognormal_sigma)) for _ in range(world * local_e)]
        return weights

    raise ValueError(f"Unsupported routing pattern: {pattern}")


def _build_global_expert_counts(config, pattern, zipf_alpha, seed, *, lognormal_sigma=0.85):
    total_routes = config.world_size * config.max_tokens * config.top_k
    weights = build_expert_popularity(
        config,
        pattern,
        zipf_alpha,
        seed,
        lognormal_sigma=lognormal_sigma,
    )
    return _scale_counts(weights, total_routes, total_routes)


def _generate_routing(config, pattern, zipf_alpha, seed, device, *, lognormal_sigma=0.85):
    import torch

    if pattern == "balanced":
        torch.manual_seed(seed + config.rank)
        ids = torch.rand((config.max_tokens, config.num_experts), device=device).topk(config.top_k, dim=-1).indices
        ids = ids.to(torch.int32)
        scores = torch.rand((config.max_tokens, config.top_k), device=device, dtype=torch.float32)
        scores /= scores.sum(dim=-1, keepdim=True)
        return ids, scores

    counts = _build_global_expert_counts(config, pattern, zipf_alpha, seed, lognormal_sigma=lognormal_sigma)
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


def _trace_kernel_summary(path):
    """Median kernel durations (us) per profiled graph from a Chrome trace; replay 0 absorbs rank skew."""
    with open(path) as handle:
        events = [event for event in json.load(handle)["traceEvents"] if event.get("ph") == "X"]
    replays = sorted(
        (event["ts"], event["name"])
        for event in events
        if event.get("cat") == "user_annotation" and "-replay" in event["name"]
    )
    labelled = []
    shared_streams = set()
    for event in sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"]):
        # GPU work follows its replay's launch and every replay is synchronized before the next one.
        label = None
        for start, name in replays:
            if start <= event["ts"]:
                label = name
        if label is None or label.endswith("-replay0"):
            continue
        name = event["name"]
        stream = (label, event.get("args", {}).get("stream"))
        if "megaMoeW8A16<1" in name or "megaMoeW8A16<2" in name or "megaMoeW8A8Local" in name:
            kind = "shared_kernel"
            shared_streams.add(stream)
        elif "megaMoeW" in name:
            kind = "routed_kernel"
        elif "quantize" in name:
            kind = "quantize"
        else:
            continue
        labelled.append((label.rsplit("-", 1)[0], kind, stream, event["dur"]))
    durations = {}
    for label, kind, stream, duration in labelled:
        # The W8A8 shared expert quantizes its own input on the shared stream.
        if kind == "quantize" and stream in shared_streams:
            kind = "shared_quantize"
        durations.setdefault(label, {}).setdefault(kind, []).append(duration)
    return {
        label: {kind: statistics.median(values) for kind, values in kinds.items()} for label, kinds in durations.items()
    }


def _weights(config, device):
    import torch
    from .quantization import quantize_mxfp4, quantize_mxfp8

    fp8_nv = config.weight_dtype == "fp8_nv"
    mxfp8_weights = config.weight_dtype == "mxfp8" or fp8_nv
    tensors = []
    for m, k in ((2 * config.intermediate, config.hidden), (config.hidden, config.intermediate)):
        values = torch.empty(
            (config.local_experts, m, k if mxfp8_weights else k // 2),
            dtype=torch.float8_e4m3fn if mxfp8_weights else torch.uint8,
            device=device,
        )
        if fp8_nv:
            scales = torch.empty((config.local_experts,), dtype=torch.float32, device=device)
        else:
            scales = torch.empty((config.local_experts, m, k // 32), dtype=torch.uint8, device=device)
        for expert in range(config.local_experts):
            source = torch.randn((m, k), dtype=torch.float32, device=device) / k**0.5
            if fp8_nv:
                # One FP32 scale per matrix, amax / 448, as fp8_nv checkpoints store the shared expert.
                scale = source.abs().amax() / torch.finfo(torch.float8_e4m3fn).max
                quantized = (source / scale).to(torch.float8_e4m3fn)
            else:
                quantized, scale = quantize_mxfp8(source) if mxfp8_weights else quantize_mxfp4(source)
            values[expert].copy_(quantized)
            scales[expert].copy_(scale)
        tensors.extend((values, scales))
    return tensors


def _reference(config, inputs, ids, scores, weights):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from .quantization import (
        dequantize_fp8_static,
        dequantize_mxfp4,
        dequantize_mxfp8,
        quantize_fp8_static,
        quantize_mxfp8,
    )

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
        if config.is_w8a8:
            # Static-FP8 input and hidden with MXFP8 or fp8_nv weights; the shared output scale is applied after
            # the sum.
            if config.weight_dtype == "fp8_nv":
                first = fc1[expert].float() * fc1_scale[expert]
                second = fc2[expert].float() * fc2_scale[expert]
            else:
                first = dequantize_mxfp8(fc1[expert], fc1_scale[expert], dtype=torch.float32)
                second = dequantize_mxfp8(fc2[expert], fc2_scale[expert], dtype=torch.float32)
            expert_input = quantize_fp8_static(all_inputs[token].float(), config.activation_scale).float()
            gate, up = ((expert_input @ first.T) * config.activation_scale).chunk(2, dim=-1)
            if config.gate_up_clamp >= 0:
                gate = gate.clamp(max=config.gate_up_clamp)
                up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
            hidden = quantize_fp8_static(functional.silu(gate) * up, config.activation_scale).float()
            partial[token, slot] = (hidden @ second.T) * all_scores[token, slot, None]
            continue
        if config.is_w8a16:
            first = dequantize_mxfp8(fc1[expert], fc1_scale[expert], dtype=torch.bfloat16).float()
            second = dequantize_mxfp8(fc2[expert], fc2_scale[expert], dtype=torch.bfloat16).float()
            gate, up = (all_inputs[token].float() @ first.T).chunk(2, dim=-1)
            if config.gate_up_clamp >= 0:
                gate = gate.clamp(max=config.gate_up_clamp)
                up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
            if config.combine_route_weights:
                hidden = (functional.silu(gate) * up).to(torch.bfloat16)
                partial[token, slot] = (hidden.float() @ second.T).to(torch.bfloat16).float() * all_scores[
                    token, slot, None
                ]
            else:
                hidden = (functional.silu(gate) * up * all_scores[token, slot, None]).to(torch.bfloat16)
                partial[token, slot] = (hidden.float() @ second.T).to(torch.bfloat16).float()
            continue
        first = dequantize_mxfp4(fc1[expert], fc1_scale[expert], dtype=torch.float32)
        second = dequantize_mxfp4(fc2[expert], fc2_scale[expert], dtype=torch.float32)
        if config.is_static_fp8:
            expert_input = quantize_fp8_static(all_inputs[token].float(), config.activation_scale).float()
            gate, up = ((expert_input @ first.T) * config.activation_scale).chunk(2, dim=-1)
        else:
            expert_input = dequantize_mxfp8(*quantize_mxfp8(all_inputs[token].float()), dtype=torch.float32)
            gate, up = (expert_input @ first.T).chunk(2, dim=-1)
        if config.gate_up_clamp >= 0:
            gate = gate.clamp(max=config.gate_up_clamp)
            up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
        hidden = functional.silu(gate) * up
        if config.is_static_fp8:
            hidden = quantize_fp8_static(hidden, config.activation_scale).float()
            if config.combine_route_weights:
                result = hidden @ second.T
                partial[token, slot] = quantize_fp8_static(result, 1.0).float() * all_scores[token, slot, None]
            else:
                result = (hidden @ second.T) * all_scores[token, slot, None]
                partial[token, slot] = quantize_fp8_static(result, 1.0).float()
        else:
            hidden = dequantize_mxfp8(*quantize_mxfp8(hidden), dtype=torch.float32)
            if config.combine_route_weights:
                result = hidden @ second.T
                partial[token, slot] = result.to(torch.bfloat16).float() * all_scores[token, slot, None]
            else:
                result = (hidden @ second.T) * all_scores[token, slot, None]
                partial[token, slot] = result.to(torch.bfloat16).float()
    host_partial = partial.cpu()
    if config.world_size > 1:
        dist.all_reduce(host_partial)
    local_sum = host_partial[first_token : first_token + inputs.shape[0]].sum(dim=1)
    if config.is_static_fp8:
        local_sum = local_sum * config.activation_scale
    return local_sum.to(inputs.device, torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="tokens per rank")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4352, help="post-SwiGLU intermediate width")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--sm-margin", type=int, default=32)
    parser.add_argument("--gate-up-clamp", type=float, default=-1.0)
    parser.add_argument(
        "--overlap-shared-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlap one local W8A16 or W8A8 shared expert with routed MegaMoE",
    )
    parser.add_argument(
        "--fused-shared-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run one fp8_nv shared expert inside the routed static-FP8 W4A8 kernel",
    )
    parser.add_argument("--shared-hidden", type=int, default=8704)
    parser.add_argument("--shared-intermediate", type=int, default=2048)
    parser.add_argument(
        "--shared-sm-count",
        type=int,
        default=16,
        help="persistent CTAs reserved for the shared expert",
    )
    parser.add_argument("--shared-gate-up-clamp", type=float, default=-1.0)
    parser.add_argument(
        "--shared-activation-dtype",
        choices=("bf16", "fp8_static"),
        default="bf16",
        help="shared-expert activations: bf16 runs the W8A16 kernel, fp8_static the W8A8 kernel "
        "(FP8 weights on the block-scaled W4A8 pipeline; see --shared-weight-dtype)",
    )
    parser.add_argument(
        "--shared-activation-scale",
        type=float,
        default=None,
        help="static FP8 per-tensor scale for --shared-activation-dtype fp8_static (default 1.0)",
    )
    parser.add_argument(
        "--shared-weight-dtype",
        choices=("mxfp8", "fp8_nv"),
        default="mxfp8",
        help="shared-expert FP8 weights: mxfp8 (E8M0 K32 block scales) or fp8_nv (one FP32 scale per FC1/FC2 "
        "matrix, requires --shared-activation-dtype fp8_static)",
    )
    parser.add_argument(
        "--shared-weight-banks",
        type=int,
        default=1,
        help="separately packed equal-valued shared weight copies rotated per call (one aliased workspace); "
        "use >1 so shared weights are not L2-resident across calls as they would be in a multi-layer model",
    )
    parser.add_argument(
        "--shared-wait-for-routed-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="admit shared work only after the routed kernel signals entry",
    )
    parser.add_argument(
        "--allow-sm-oversubscription",
        action="store_true",
        help="diagnostic: allow routed + shared CTAs to exceed the SM count; each CTA fills an SM, so shared CTAs "
        "then only start on SMs the routed kernel has released",
    )
    parser.add_argument(
        "--shared-join",
        choices=("iteration", "batch"),
        default="iteration",
        help="join the shared branch before every routed call (model-like) or only at the end of each graph batch "
        "(diagnostic: routed calls stay rank-aligned, isolating SM/memory contention from cross-rank skew)",
    )
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
        "--kernel-key",
        default=None,
        help="load an already-prepared W4A8 module by content-addressed cache key",
    )
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
        help="token-to-expert routing distribution (balanced, zipf, zipf_skewed, hot_experts, lognormal)",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.2, help="Zipf exponent for zipf/zipf_skewed patterns")
    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=0.85,
        help="log-normal sigma for lognormal routing pattern (default: 0.85, matching c1024-decode-all-ranks-p50)",
    )
    parser.add_argument(
        "--weight-dtype",
        choices=("mxfp4", "mxfp8"),
        default="mxfp4",
        help="weight quantization mode (mxfp4 for W4A8, mxfp8 for W8A16)",
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("mxfp8", "fp8_static", "bf16"),
        default="mxfp8",
        help="activation and combine mode (mxfp8, fp8_static, or bf16 for W8A16)",
    )
    parser.add_argument("--activation-scale", type=float, default=None, help="static FP8 activation scale")
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
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap-port", type=int, default=None, help="MSCCL++ port (default MASTER_PORT+1)")
    parser.add_argument(
        "--combine-route-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply routing weights in combine instead of FC1/FC2",
    )
    parser.add_argument("--json-output", default=None, help="optional output path")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="after timing, profile a few graph replays, write each rank's Chrome trace to this (host-local) "
        "directory, and report per-graph median kernel durations",
    )
    args = parser.parse_args()
    if min(args.tokens, args.graph_batch, args.warmup, args.iterations) < 1:
        parser.error("tokens, graph-batch, warmup, and iterations must be positive")
    is_w8a16 = args.weight_dtype == "mxfp8" or args.activation_dtype == "bf16"
    if not is_w8a16 and (args.hidden, args.intermediate) not in (
        (4096, 4352),
        (4096, 5120),
        (4096, 6656),
        (8192, 4096),
        (9216, 4096),
        (9216, 4608),
    ):
        parser.error("W4A8 H/I must be 4096/4352, 4096/5120, 4096/6656, 8192/4096, 9216/4096, or 9216/4608")
    if not args.graph and args.graph_timing != "isolated":
        parser.error("--graph-timing steady-state requires --graph")
    if not math.isfinite(args.lognormal_sigma) or args.lognormal_sigma < 0:
        parser.error("--lognormal-sigma must be nonnegative and finite")
    if args.overlap_shared_expert and args.fused_shared_expert:
        parser.error("--overlap-shared-expert and --fused-shared-expert are mutually exclusive")
    if (args.overlap_shared_expert or args.fused_shared_expert) and args.shared_weight_banks < 1:
        parser.error("--shared-weight-banks must be positive")
    if args.overlap_shared_expert and (
        min(args.shared_hidden, args.shared_intermediate, args.shared_sm_count, args.shared_weight_banks) < 1
        or args.shared_sm_count % 2
    ):
        parser.error(
            "shared hidden/intermediate/weight-banks must be positive and shared-sm-count must be positive and even"
        )
    if not math.isfinite(args.shared_gate_up_clamp):
        parser.error("--shared-gate-up-clamp must be finite")
    if (
        args.overlap_shared_expert
        and args.shared_activation_scale is not None
        and args.shared_activation_dtype != "fp8_static"
    ):
        parser.error("--shared-activation-scale requires --shared-activation-dtype fp8_static")
    if (
        args.overlap_shared_expert
        and args.shared_weight_dtype == "fp8_nv"
        and args.shared_activation_dtype != "fp8_static"
    ):
        parser.error("--shared-weight-dtype fp8_nv requires --shared-activation-dtype fp8_static")

    import torch
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from .api import MegaMoE, MegaMoEConfig, SharedExpertConfig, is_available
    from .jit import W4A8KernelConfig, compile_kernel, load_cached_kernel
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
    device = torch.device("cuda", local_rank)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    if args.overlap_shared_expert and args.shared_sm_count > sm_count - 2:
        parser.error("--shared-sm-count must leave at least one two-SM cluster available")
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
        weight_dtype=args.weight_dtype,
        activation_dtype=args.activation_dtype,
        activation_scale=args.activation_scale,
        combine_route_weights=args.combine_route_weights,
    )
    if args.fused_shared_expert:
        if config.weight_dtype != "mxfp4" or config.activation_dtype != "fp8_static":
            parser.error("--fused-shared-expert requires routed --weight-dtype mxfp4 and --activation-dtype fp8_static")
        if args.shared_activation_scale is not None and args.shared_activation_scale != config.activation_scale:
            parser.error("fused shared and routed activation scales must match")
    kernel_config = W4A8KernelConfig()
    if not config.is_w8a16 and args.w4_profile is not None:
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
    if args.kernel_key is not None and (args.w4_profile is not None or policy_overrides):
        parser.error("--kernel-key cannot be combined with --w4-profile or kernel policy overrides")
    if args.fused_shared_expert:
        if args.kernel_key is not None or args.w4_profile is not None or policy_overrides:
            parser.error("--fused-shared-expert currently requires the builtin kernel without policy overrides")
        kernel = load_cached_kernel("builtin")
    elif config.is_w8a16:
        if args.kernel_key is not None:
            parser.error("--kernel-key is only supported for W4A8")
        kernel = None
    elif args.kernel_key is not None:
        kernel = load_cached_kernel(args.kernel_key, cache_dir=args.cache_dir)
        kernel_config = kernel.config
    else:
        kernel = compile_kernel(kernel_config, cache_dir=args.cache_dir)
    if args.overlap_shared_expert:
        # Both kernels are cooperative persistent launches. If their CTAs cannot be co-resident,
        # the shared kernel only runs on SMs the routed kernel leaves free and effectively serializes.
        routed_cluster = 2 if config.is_w8a16 else kernel_config.cluster_size
        routed_cta_bound = (sm_count - args.sm_margin) // routed_cluster * routed_cluster
        if routed_cta_bound + args.shared_sm_count > sm_count and not args.allow_sm_oversubscription:
            parser.error(
                f"routed MegaMoE may keep {routed_cta_bound} CTAs resident (--sm-margin {args.sm_margin}) and the "
                f"shared expert needs {args.shared_sm_count}, exceeding {sm_count} SMs; the cooperative launches "
                "could not overlap. Use --sm-margin >= --shared-sm-count"
            )
    torch.manual_seed(args.seed + rank)
    # Gloo is used only for rendezvous, reporting, and the untimed reference.
    # All timed expert communication uses the native MSCCL++ CudaIpc mappings.
    dist.init_process_group("gloo", rank=rank, world_size=world)
    bootstrap = TcpBootstrap.create(rank, world)
    port = args.bootstrap_port or int(os.environ["MASTER_PORT"]) + 1
    bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
    communicator = Communicator(bootstrap)
    weights = _weights(config, device)
    shared_bootstrap = None
    shared_communicator = None
    shared_context = None
    shared_contexts = []
    shared_config = None
    shared_weights = None
    shared_initialization_ms = None
    fused_contexts = []
    if args.fused_shared_expert:
        shared_config = MegaMoEConfig(
            rank=0,
            world_size=1,
            max_tokens=args.tokens,
            hidden=args.shared_hidden,
            intermediate=args.shared_intermediate,
            num_experts=1,
            top_k=1,
            sm_margin=args.sm_margin,
            gate_up_clamp=args.shared_gate_up_clamp,
            combine_route_weights=False,
            weight_dtype="fp8_nv",
            activation_dtype="fp8_static",
            activation_scale=config.activation_scale,
        )
        shared_weights = _weights(shared_config, device)
        fused_config = SharedExpertConfig(
            args.shared_intermediate,
            args.shared_gate_up_clamp,
            hidden=args.shared_hidden,
        )
        start = time.perf_counter()
        for _ in range(args.shared_weight_banks):
            fused_contexts.append(
                MegaMoE(
                    config,
                    communicator,
                    *weights,
                    kernel=kernel,
                    shared_workspace=fused_contexts[0] if fused_contexts else None,
                    shared_config=fused_config,
                    shared_fc1=shared_weights[0],
                    shared_fc1_scale=shared_weights[1],
                    shared_fc2=shared_weights[2],
                    shared_fc2_scale=shared_weights[3],
                )
            )
        context = fused_contexts[0]
        initialization_ms = (time.perf_counter() - start) * 1000
    else:
        start = time.perf_counter()
        context = MegaMoE(config, communicator, *weights, kernel=kernel)
        initialization_ms = (time.perf_counter() - start) * 1000
    if args.overlap_shared_expert:
        shared_config = MegaMoEConfig(
            rank=0,
            world_size=1,
            max_tokens=args.tokens,
            hidden=args.shared_hidden,
            intermediate=args.shared_intermediate,
            num_experts=1,
            top_k=1,
            sm_margin=sm_count - args.shared_sm_count,
            gate_up_clamp=args.shared_gate_up_clamp,
            combine_route_weights=False,
            weight_dtype=args.shared_weight_dtype,
            activation_dtype=args.shared_activation_dtype,
            activation_scale=args.shared_activation_scale,
        )
        shared_bootstrap = TcpBootstrap.create(0, 1)
        shared_bootstrap.initialize(TcpBootstrap.create_unique_id())
        shared_communicator = Communicator(shared_bootstrap)
        shared_weights = _weights(shared_config, device)
        start = time.perf_counter()
        # Extra banks alias the first workspace and only add separately packed copies of the same weight values.
        for _ in range(args.shared_weight_banks):
            shared_contexts.append(
                MegaMoE(
                    shared_config,
                    shared_communicator,
                    *shared_weights,
                    shared_workspace=shared_contexts[0] if shared_contexts else None,
                )
            )
        shared_context = shared_contexts[0]
        shared_initialization_ms = (time.perf_counter() - start) * 1000
        if shared_context.cta_count != args.shared_sm_count:
            raise RuntimeError(
                f"shared CTA cap mismatch: requested {args.shared_sm_count}, got {shared_context.cta_count}"
            )
        if context.cta_count + shared_context.cta_count > sm_count and not args.allow_sm_oversubscription:
            raise RuntimeError(
                f"routed ({context.cta_count}) and shared ({shared_context.cta_count}) CTAs exceed {sm_count} SMs"
            )
    inputs = torch.randn((args.tokens, args.hidden), device=device, dtype=torch.bfloat16)
    ids, scores = _generate_routing(
        config, args.routing_pattern, args.zipf_alpha, args.seed, device, lognormal_sigma=args.lognormal_sigma
    )
    routing_summary = _routing_summary(config, ids, kernel_config.tile_n)
    if args.input_mode == "direct":
        direct = context.input_view(args.tokens)
        direct.copy_(inputs)
        inputs = direct
    output = torch.empty_like(inputs)
    stream = torch.cuda.Stream(device=device)
    shared_inputs = None
    shared_output = None
    shared_stream = None
    if fused_contexts:
        shared_inputs = (
            inputs
            if args.shared_hidden == args.hidden
            else torch.randn(
                (args.tokens, args.shared_hidden),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        shared_output = torch.empty_like(shared_inputs)
    elif shared_context is not None:
        shared_inputs = torch.randn(
            (args.tokens, args.shared_hidden),
            device=device,
            dtype=torch.bfloat16,
        )
        if args.input_mode == "direct":
            direct = shared_context.input_view(args.tokens)
            direct.copy_(shared_inputs)
            shared_inputs = direct
        shared_output = torch.empty_like(shared_inputs)
        shared_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    stream.wait_stream(current_stream)
    if shared_stream is not None:
        shared_stream.wait_stream(current_stream)

    def launch_routed(*, signal_start=False, timing=None):
        if timing is not None:
            timing["routed_begin"].record(stream)
        context(
            inputs,
            ids,
            scores,
            output=output,
            stream=stream,
            signal_start=signal_start,
        )
        if timing is not None:
            timing["routed_end"].record(stream)

    def launch_shared(index, target_stream):
        shared_contexts[index % len(shared_contexts)].forward_shared(
            shared_inputs,
            output=shared_output,
            stream=target_stream,
        )

    def launch_fused(index=0):
        fused_contexts[index % len(fused_contexts)].forward_fused_shared(
            inputs,
            ids,
            scores,
            shared_inputs=shared_inputs,
            output=output,
            shared_output=shared_output,
            stream=stream,
        )

    def launch_overlap(timing=None, index=0, join=True):
        assert shared_context is not None
        assert shared_inputs is not None
        assert shared_output is not None
        assert shared_stream is not None
        shared_stream.wait_stream(stream)
        launch_routed(
            signal_start=args.shared_wait_for_routed_start,
            timing=timing,
        )
        if args.shared_wait_for_routed_start:
            context.wait_until_started(shared_stream)
        if timing is not None:
            timing["shared_begin"].record(shared_stream)
        launch_shared(index, shared_stream)
        if timing is not None:
            timing["shared_end"].record(shared_stream)
        if join:
            stream.wait_stream(shared_stream)

    for _ in range(args.warmup):
        launch_routed()
    stream.synchronize()
    if fused_contexts:
        for index in range(args.warmup):
            launch_fused(index)
        stream.synchronize()
    elif shared_context is not None:
        for index in range(args.warmup):
            launch_overlap(index=index)
        stream.synchronize()
    rtol = args.rtol if args.rtol is not None else (0.125 if config.is_static_fp8 else 0.05)
    atol = (
        args.atol
        if args.atol is not None
        else (max(0.02, 2.5 * config.activation_scale) if config.is_static_fp8 else 0.01)
    )
    # The shared reference models W8A8 static-FP8 rounding too, so both shared variants keep the tight tolerance.
    shared_rtol = args.rtol if args.rtol is not None else 0.03
    shared_atol = args.atol if args.atol is not None else 0.01
    correctness = None
    shared_correctness = None
    if args.check:
        reference = _reference(config, inputs, ids, scores, weights)
        error = (output.float() - reference.float()).abs()
        correctness = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        if fused_contexts or shared_context is not None:
            assert shared_config is not None
            assert shared_inputs is not None
            assert shared_output is not None
            assert shared_weights is not None
            shared_ids = torch.zeros((args.tokens, 1), dtype=torch.int32, device=device)
            shared_scores = torch.ones((args.tokens, 1), dtype=torch.float32, device=device)
            shared_reference = _reference(
                shared_config,
                shared_inputs,
                shared_ids,
                shared_scores,
                shared_weights,
            )
            shared_error = (shared_output.float() - shared_reference.float()).abs()
            shared_correctness = {
                "max_abs_error": shared_error.max().item(),
                "mean_abs_error": shared_error.mean().item(),
            }
            torch.testing.assert_close(shared_output, shared_reference, rtol=shared_rtol, atol=shared_atol)
    del weights
    del shared_weights
    routed_graph = None
    shared_graph = None
    overlap_graph = None
    overlap_timed_graph = None
    overlap_timing = None
    fused_graph = None
    if args.graph:
        routed_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(routed_graph, stream=stream):
            for _ in range(args.graph_batch):
                launch_routed()
        routed_graph.replay()
        torch.cuda.synchronize(device)
        if args.check:
            torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        if fused_contexts:
            fused_graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(fused_graph, stream=stream):
                for index in range(args.graph_batch):
                    launch_fused(index)
            fused_graph.replay()
            torch.cuda.synchronize(device)
            if args.check:
                torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
                torch.testing.assert_close(shared_output, shared_reference, rtol=shared_rtol, atol=shared_atol)
        elif shared_context is not None:
            shared_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(shared_graph, stream=shared_stream):
                for index in range(args.graph_batch):
                    launch_shared(index, shared_stream)
            shared_graph.replay()
            torch.cuda.synchronize(device)
            overlap_timing = [
                {
                    name: torch.cuda.Event(enable_timing=True, external=True)
                    for name in (
                        "routed_begin",
                        "routed_end",
                        "shared_begin",
                        "shared_end",
                    )
                }
                for _ in range(args.graph_batch)
            ]
            # The untimed graph gives the end-to-end cost; branch event nodes add graph latency, so the
            # per-branch breakdown comes from a separately captured graph.
            overlap_graph = torch.cuda.CUDAGraph()
            overlap_timed_graph = torch.cuda.CUDAGraph()
            for graph, timings in (
                (overlap_graph, [None] * args.graph_batch),
                (overlap_timed_graph, overlap_timing),
            ):
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    for index, timing in enumerate(timings):
                        launch_overlap(
                            timing,
                            index,
                            join=args.shared_join == "iteration" or index == len(timings) - 1,
                        )
                graph.replay()
                torch.cuda.synchronize(device)
            if args.check:
                torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
                torch.testing.assert_close(shared_output, shared_reference, rtol=shared_rtol, atol=shared_atol)
    routed_control_samples = []
    repetitions = args.graph_batch if routed_graph else 1
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        dist.barrier()
        with torch.cuda.stream(stream):
            if routed_graph and args.graph_timing == "steady-state":
                routed_graph.replay()
            begin.record()
            routed_graph.replay() if routed_graph else launch_routed()
            end.record()
        end.synchronize()
        routed_control_samples.append(begin.elapsed_time(end) * 1000 / repetitions)
    shared_control_samples = None
    if shared_context is not None:
        # The shared expert alone on its capped CTAs; no peers are involved.
        shared_control_samples = []
        repetitions = args.graph_batch if shared_graph else 1
        for _ in range(args.iterations):
            dist.barrier()
            with torch.cuda.stream(shared_stream):
                if shared_graph and args.graph_timing == "steady-state":
                    shared_graph.replay()
                begin.record()
                shared_graph.replay() if shared_graph else launch_shared(0, shared_stream)
                end.record()
            end.synchronize()
            shared_control_samples.append(begin.elapsed_time(end) * 1000 / repetitions)
    overlap_samples = None
    overlap_timed_samples = None
    routed_overlap_samples = None
    shared_overlap_samples = None
    shared_start_delay_samples = None
    shared_tail_samples = None
    if shared_context is not None:
        overlap_samples = []
        overlap_timed_samples = []
        routed_overlap_samples = []
        shared_overlap_samples = []
        shared_start_delay_samples = []
        shared_tail_samples = []
        eager_timing = None
        if overlap_graph is None:
            eager_timing = {
                name: torch.cuda.Event(enable_timing=True, external=True)
                for name in (
                    "routed_begin",
                    "routed_end",
                    "shared_begin",
                    "shared_end",
                )
            }
            overlap_timing = [eager_timing]
        assert overlap_timing is not None
        repetitions = args.graph_batch if overlap_graph else 1
        for _ in range(args.iterations):
            for graph, timing, samples in (
                (overlap_graph, None, overlap_samples),
                (overlap_timed_graph, eager_timing, overlap_timed_samples),
            ):
                dist.barrier()
                with torch.cuda.stream(stream):
                    if graph and args.graph_timing == "steady-state":
                        graph.replay()
                    begin.record()
                    graph.replay() if graph else launch_overlap(timing)
                    end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / repetitions)
            routed_overlap_samples.append(
                sum(timing["routed_begin"].elapsed_time(timing["routed_end"]) for timing in overlap_timing)
                * 1000
                / repetitions
            )
            shared_overlap_samples.append(
                sum(timing["shared_begin"].elapsed_time(timing["shared_end"]) for timing in overlap_timing)
                * 1000
                / repetitions
            )
            # Branch alignment: routed start -> shared start, and routed end -> shared end
            # (a positive tail is shared work left exposed after the routed branch).
            shared_start_delay_samples.append(
                sum(timing["routed_begin"].elapsed_time(timing["shared_begin"]) for timing in overlap_timing)
                * 1000
                / repetitions
            )
            shared_tail_samples.append(
                sum(timing["routed_end"].elapsed_time(timing["shared_end"]) for timing in overlap_timing)
                * 1000
                / repetitions
            )
    fused_samples = None
    if fused_contexts:
        fused_samples = []
        repetitions = args.graph_batch if fused_graph else 1
        for _ in range(args.iterations):
            dist.barrier()
            with torch.cuda.stream(stream):
                if fused_graph and args.graph_timing == "steady-state":
                    fused_graph.replay()
                begin.record()
                fused_graph.replay() if fused_graph else launch_fused()
                end.record()
            end.synchronize()
            fused_samples.append(begin.elapsed_time(end) * 1000 / repetitions)
    ktrace_dir = os.environ.get("MEGAMOE_KTRACE_DIR")
    if ktrace_dir and routed_graph is not None:
        import ctypes
        import glob as _glob
        import numpy as _np
        import mscclpp as _m

        lib_path = sorted(_glob.glob(os.path.join(os.path.dirname(_m.__file__), "lib", "libmscclpp_megamoe.so*")))[0]
        lib = ctypes.CDLL(lib_path)
        tracks, capacity = 32, 8192
        os.makedirs(ktrace_dir, exist_ok=True)
        for label, graph in [("routed", routed_graph)] + ([("fused", fused_graph)] if fused_graph is not None else []):
            for replay in range(2):
                torch.cuda.synchronize(device)
                dist.barrier()
                assert lib.mscclpp_megamoe_w4_trace_reset() == 0
                with torch.cuda.stream(stream):
                    graph.replay()
                torch.cuda.synchronize(device)
                events = _np.zeros(tracks * capacity * 2, dtype=_np.uint64)
                counts = _np.zeros(tracks, dtype=_np.uint32)
                assert (
                    lib.mscclpp_megamoe_w4_trace_copy(
                        events.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_size_t(events.nbytes),
                        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                        ctypes.c_size_t(counts.nbytes),
                    )
                    == 0
                )
                if rank < 4:
                    _np.savez(
                        os.path.join(ktrace_dir, f"{label}-r{replay}-rank{rank}.npz"), events=events, counts=counts
                    )
    kernel_trace = None
    if args.trace_dir and routed_graph is not None:
        from torch.profiler import ProfilerActivity, profile

        profiled = [("routed", routed_graph)]
        if fused_graph is not None:
            profiled.append(("fused", fused_graph))
        if overlap_graph is not None:
            profiled += [("shared", shared_graph), ("overlap", overlap_graph)]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            for label, graph in profiled:
                # The first replay absorbs profiler-start rank skew; inspect the later ones.
                for replay in range(3):
                    dist.barrier()
                    with (
                        torch.profiler.record_function(f"{label}-replay{replay}"),
                        torch.cuda.stream(shared_stream if label == "shared" else stream),
                    ):
                        graph.replay()
                    torch.cuda.synchronize(device)
        os.makedirs(args.trace_dir, exist_ok=True)
        trace_path = os.path.join(args.trace_dir, f"rank-{rank}.json")
        profiler.export_chrome_trace(trace_path)
        kernel_trace = _trace_kernel_summary(trace_path)
    reports = [None] * world
    primary_samples = (
        fused_samples
        if fused_samples is not None
        else (overlap_samples if overlap_samples is not None else routed_control_samples)
    )
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
            "samples_us": primary_samples,
            "routed_control_samples_us": routed_control_samples,
            "overlap_timed_samples_us": overlap_timed_samples,
            "routed_overlap_samples_us": routed_overlap_samples,
            "shared_overlap_samples_us": shared_overlap_samples,
            "shared_control_samples_us": shared_control_samples,
            "shared_start_delay_samples_us": shared_start_delay_samples,
            "shared_tail_samples_us": shared_tail_samples,
            "fused_samples_us": fused_samples,
            "kernel_trace_us": kernel_trace,
            "correctness": correctness,
            "shared": (
                None
                if shared_context is None and not fused_contexts
                else {
                    "mode": "fused" if fused_contexts else "two_kernel_overlap",
                    "cta_count": None if fused_contexts else shared_context.cta_count,
                    "weight_banks": len(fused_contexts) if fused_contexts else len(shared_contexts),
                    "weight_dtype": shared_config.weight_dtype,
                    "activation_dtype": shared_config.activation_dtype,
                    "activation_scale": shared_config.activation_scale,
                    "workspace_bytes": context.workspace_bytes if fused_contexts else shared_context.workspace_bytes,
                    "initialization_ms": shared_initialization_ms,
                    "kernel_id": context.kernel_id if fused_contexts else shared_context.kernel_id,
                    "kernel_config": (
                        context.effective_kernel_config if fused_contexts else shared_context.effective_kernel_config
                    ),
                    "correctness": shared_correctness,
                }
            ),
        },
    )
    if rank == 0:
        result = {
            "implementation": "mscclpp-native-cuda-megamoe",
            "scope": (
                "routed_and_fused_shared_expert"
                if fused_contexts
                else ("routed_and_overlapped_shared_expert" if shared_context is not None else "routed_experts_only")
            ),
            "excluded": (
                ["squash", "unsquash", "router", "residual", "routed_shared_add"]
                if shared_context is not None or fused_contexts
                else ["shared_expert", "squash", "unsquash", "router", "residual"]
            ),
            "configuration": {**vars(args), "world_size": world},
            "timing": {
                "method": "CUDA events; slowest rank per sample",
                "graph_timing": args.graph_timing if routed_graph else None,
                "untimed_graph_replays_per_sample": int(bool(routed_graph) and args.graph_timing == "steady-state"),
                "branch_events": "separate overlap graph" if shared_context is not None else None,
            },
            "hardware": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "latency_us_max_across_ranks": _latency_summary(
                reports,
                "samples_us",
                args.iterations,
            ),
            "routed_control_latency_us_max_across_ranks": _latency_summary(
                reports,
                "routed_control_samples_us",
                args.iterations,
            ),
            "ranks": reports,
        }
        if kernel_trace is not None:
            result["kernel_trace_us_max_across_ranks"] = {
                label: {kind: max(report["kernel_trace_us"][label][kind] for report in reports) for kind in kinds}
                for label, kinds in kernel_trace.items()
            }
        if shared_context is not None:
            result["overlap_timed_latency_us_max_across_ranks"] = _latency_summary(
                reports,
                "overlap_timed_samples_us",
                args.iterations,
            )
            result["routed_overlap_latency_us_max_across_ranks"] = _latency_summary(
                reports,
                "routed_overlap_samples_us",
                args.iterations,
            )
            result["shared_overlap_latency_us_max_across_ranks"] = _latency_summary(
                reports,
                "shared_overlap_samples_us",
                args.iterations,
            )
            result["shared_control_latency_us_max_across_ranks"] = _latency_summary(
                reports,
                "shared_control_samples_us",
                args.iterations,
            )
            result["shared_start_delay_us_max_across_ranks"] = _latency_summary(
                reports,
                "shared_start_delay_samples_us",
                args.iterations,
            )
            result["shared_tail_after_routed_us_max_across_ranks"] = _latency_summary(
                reports,
                "shared_tail_samples_us",
                args.iterations,
            )
            result["sm_partition"] = {
                "sm_count": sm_count,
                "routed_ctas": context.cta_count,
                "shared_ctas": shared_context.cta_count,
            }
        if fused_contexts:
            result["fused_latency_us_max_across_ranks"] = _latency_summary(
                reports,
                "fused_samples_us",
                args.iterations,
            )
            result["sm_partition"] = {
                "sm_count": sm_count,
                "fused_ctas": context.cta_count,
                "reserved_shared_ctas": 0,
            }
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.json_output:
            with open(args.json_output, "w") as output_file:
                output_file.write(text + "\n")
    # Release only after every GPU has completed every peer access.
    torch.cuda.synchronize(device)
    bootstrap.barrier()
    if shared_bootstrap is not None:
        shared_bootstrap.barrier()
    del routed_graph, shared_graph, overlap_graph, overlap_timed_graph
    del context, shared_context, shared_contexts, inputs, shared_inputs
    del shared_communicator, shared_bootstrap
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
