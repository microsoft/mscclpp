# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routed-only FlashInfer MegaMoE benchmark; launch with torchrun (see README)."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="tokens per rank")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4352, help="post-SwiGLU intermediate width")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--gate-up-clamp", type=float, default=-1.0)
    parser.add_argument("--e5m2", action="store_true")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-batch", type=int, default=10, help="collectives captured per replay")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--in-kernel-fc2-reduce", action="store_true")
    knobs = parser.add_mutually_exclusive_group()
    knobs.add_argument("--autotune", action="store_true", help="run FlashInfer's collective knob tuner at warmup")
    knobs.add_argument("--knobs-json", help="inline JSON object or path to a JSON file with pinned kernel knobs")
    parser.add_argument(
        "--flashinfer-root",
        default=os.environ.get("FLASHINFER_ROOT"),
        help="FlashInfer source checkout; defaults to FLASHINFER_ROOT or sibling ../flashinfer",
    )
    parser.add_argument(
        "--flashinfer-cache-root",
        default=os.environ.get("FLASHINFER_BENCHMARK_CACHE_ROOT"),
        help="persistent cache root; each global rank uses a separate subdirectory",
    )
    parser.add_argument("--check", action="store_true", help="compare against an independent Torch BF16 reference")
    parser.add_argument(
        "--reference-relative-l2",
        type=float,
        default=None,
        help="correctness threshold; defaults to 0.05 for E4M3 and 0.12 for E5M2",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", default=None, help="optional output path")
    args = parser.parse_args(argv)
    if min(args.tokens, args.graph_batch, args.warmup, args.iterations) < 1:
        parser.error("tokens, graph-batch, warmup, and iterations must be positive")
    if args.hidden < 32 or args.hidden % 32:
        parser.error("hidden must be a positive multiple of 32")
    if args.intermediate < 64 or args.intermediate % 64:
        parser.error("intermediate must be a positive multiple of 64")
    if args.experts < 1 or not 1 <= args.top_k <= min(32, args.experts):
        parser.error("experts must be positive and top-k must be in [1, min(32, experts)]")
    if not math.isfinite(args.gate_up_clamp):
        parser.error("gate-up-clamp must be finite; negative disables clamping")
    if args.reference_relative_l2 is not None and (
        not math.isfinite(args.reference_relative_l2) or args.reference_relative_l2 <= 0
    ):
        parser.error("reference-relative-l2 must be finite and positive")
    return args


def _default_flashinfer_root():
    try:
        candidate = Path(__file__).resolve().parents[5] / "flashinfer"
    except IndexError:
        return None
    return candidate if (candidate / "flashinfer/__init__.py").is_file() else None


def _activate_flashinfer_root(value):
    root = Path(value).expanduser().resolve() if value else _default_flashinfer_root()
    if root is not None:
        if not (root / "flashinfer/__init__.py").is_file():
            raise FileNotFoundError(f"FlashInfer source package is missing under {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    return root


def _load_knobs(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    text = path.read_text() if path.is_file() else value
    knobs = json.loads(text)
    if not isinstance(knobs, dict):
        raise ValueError("knobs JSON must be an object")
    return {key: tuple(setting) if isinstance(setting, list) else setting for key, setting in knobs.items()}


def _source_info(root, module):
    info = {
        "version": getattr(module, "__version__", "unknown"),
        "module": str(Path(module.__file__).resolve()),
    }
    source_commit = os.environ.get("FLASHINFER_SOURCE_COMMIT", "").strip()
    if source_commit:
        info["commit"] = source_commit
    if root is not None:
        info["root"] = str(root)
        if source_commit:
            return info
        try:
            info["commit"] = subprocess.run(
                ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            info["commit"] = "unknown"
    return info


def _runtime_versions():
    packages = (
        "apache-tvm-ffi",
        "nvidia-cutlass-dsl",
        "nvshmem4py-cu13",
        "nvidia-nccl-cu13",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _weights(rank, local_experts, hidden, intermediate, device, seed=13):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    w13 = torch.randn(
        local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w13.mul_(hidden**-0.5)
    w2 = torch.randn(
        local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(intermediate**-0.5)
    return w13, w2


def _reference(rank, world, hidden, intermediate, top_k, clamp, inputs, ids, scores, weights):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    def gather(tensor):
        gathered = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered)

    all_inputs, all_ids, all_scores = gather(inputs), gather(ids), gather(scores)
    partial = torch.zeros(
        all_inputs.shape[0],
        top_k,
        hidden,
        device=inputs.device,
        dtype=torch.float32,
    )
    w13, w2 = weights
    local_experts = w13.shape[0]
    for expert in range(local_experts):
        token, slot = torch.where(all_ids == rank * local_experts + expert)
        if not token.numel():
            continue
        gate, up = (all_inputs[token].float() @ w13[expert].float().T).chunk(2, dim=-1)
        if clamp is not None:
            gate = gate.clamp(max=clamp)
            up = up.clamp(-clamp, clamp)
        activated = (functional.silu(gate) * up * all_scores[token, slot, None]).to(torch.bfloat16)
        partial[token, slot] = (activated.float() @ w2[expert].float().T).to(torch.bfloat16).float()
    dist.all_reduce(partial)
    first_token = rank * inputs.shape[0]
    return partial[first_token : first_token + inputs.shape[0]].sum(dim=1).to(torch.bfloat16)


def _error_stats(actual, expected):
    import torch

    difference = actual.float() - expected.float()
    expected_norm = torch.linalg.vector_norm(expected.float())
    return {
        "max_abs_error": difference.abs().max().item(),
        "mean_abs_error": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / expected_norm.clamp_min(1e-12)).item(),
    }


def _phase(rank, name):
    if rank == 0:
        print(f"[flashinfer-megamoe] {name}", flush=True)


def _configure_single_node_transport(world, local_world):
    if world != local_world:
        return
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_NET", "Socket")
    os.environ.setdefault("NCCL_MNNVL_ENABLE", "1")
    os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")


def main(argv=None):
    args = _parse_args(argv)
    if args.reference_relative_l2 is None:
        args.reference_relative_l2 = 0.12 if args.e5m2 else 0.05
    flashinfer_root = _activate_flashinfer_root(args.flashinfer_root)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    _configure_single_node_transport(world, local_world)
    if args.flashinfer_cache_root:
        cache = Path(args.flashinfer_cache_root).expanduser().resolve() / f"rank-{rank}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(cache)
        os.environ["XDG_CACHE_HOME"] = str(cache / ".cache")

    _phase(rank, "import latest FlashInfer checkout")
    import torch
    import torch.distributed as dist
    import flashinfer
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if args.experts % world:
        raise ValueError(f"experts ({args.experts}) must be divisible by world size ({world})")
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("FlashInfer MegaMoE requires NVIDIA CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("this FlashInfer MegaMoE benchmark supports GB200 (SM100) only")
    _phase(rank, "initialize torch.distributed")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world)

    _phase(rank, "create inputs and BF16 weights")
    torch.manual_seed(args.seed + rank)
    local_experts = args.experts // world
    weights = _weights(rank, local_experts, args.hidden, args.intermediate, device)
    inputs = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    ids = torch.rand(args.tokens, args.experts, device=device).topk(args.top_k, dim=-1).indices.to(torch.int64)
    scores = torch.rand(args.tokens, args.top_k, dtype=torch.float32, device=device)
    scores /= scores.sum(dim=-1, keepdim=True)
    tensors = MoEEpTensors(hidden_states=inputs, topk_ids=ids, topk_weights=scores)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    clamp = None if args.gate_up_clamp < 0 else args.gate_up_clamp
    requested_knobs = "auto" if args.autotune else _load_knobs(args.knobs_json)
    kernel_config = Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        kind="bf16_mxfp8_e5m2" if args.e5m2 else "bf16_mxfp8_e4m3",
        gate_up_clamp=clamp,
        fast_math=args.fast_math,
        enable_in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        knobs=requested_knobs,
    )
    bootstrap = BootstrapConfig(
        world_size=world,
        rank=rank,
        stream=stream.cuda_stream,
        device=local_rank,
        process_group=dist.group.WORLD,
    )

    layer = graph = output = None
    try:
        initialization_start = time.perf_counter()
        _phase(rank, "construct layer: runtime bootstrap and MXFP8 preprocessing")
        construction_start = time.perf_counter()
        with torch.cuda.stream(stream):
            layer = MoEEpLayer(
                bootstrap=bootstrap,
                fleet_params=FleetParams(
                    num_experts=args.experts,
                    max_tokens_per_rank=args.tokens,
                    token_hidden_size=args.hidden,
                ),
                weights=MoEWeightPack(w13=weights[0], w2=weights[1]),
                backend=MegaConfig(
                    megakernel=kernel_config,
                    quantize_input=True,
                    preprocess_weights=True,
                ),
            )
            if not isinstance(layer, MoEEpMegaLayer):
                raise RuntimeError("FlashInfer did not construct a MegaMoE layer")
            if not layer.supports_output_view:
                raise RuntimeError("FlashInfer mixed MegaMoE must support workspace output views")
            construction_ms = (time.perf_counter() - construction_start) * 1000
            _phase(rank, "collective warmup: workspace and CuTeDSL compile")
            warmup_start = time.perf_counter()
            layer.warmup(tensors)
        stream.synchronize()
        dist.barrier()
        warmup_initialization_ms = (time.perf_counter() - warmup_start) * 1000
        initialization_ms = (time.perf_counter() - initialization_start) * 1000
        _phase(rank, "steady-state warmup")

        def launch():
            return layer.forward(tensors, return_workspace_view=True)

        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                output = launch()
        stream.synchronize()
        correctness = None
        if args.check:
            _phase(rank, "Torch correctness reference")
            observed = output.clone()
            reference = _reference(
                rank,
                world,
                args.hidden,
                args.intermediate,
                args.top_k,
                clamp,
                inputs,
                ids,
                scores,
                weights,
            )
            torch.cuda.synchronize(device)
            correctness = _error_stats(observed, reference)
            if not torch.isfinite(observed).all() or correctness["relative_l2"] > args.reference_relative_l2:
                raise AssertionError(
                    f"FlashInfer relative L2 {correctness['relative_l2']:.6f} exceeds "
                    f"{args.reference_relative_l2:.6f}"
                )
        del weights

        if args.graph:
            _phase(rank, "capture CUDA graph")
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.graph_batch):
                    output = launch()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            dist.barrier()
            if args.check:
                graph_correctness = _error_stats(output, reference)
                if graph_correctness["relative_l2"] > args.reference_relative_l2:
                    raise AssertionError(
                        f"FlashInfer graph relative L2 {graph_correctness['relative_l2']:.6f} exceeds "
                        f"{args.reference_relative_l2:.6f}"
                    )

        samples = []
        _phase(rank, "timed iterations")
        repetitions = args.graph_batch if graph else 1
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations):
            dist.barrier()
            with torch.cuda.stream(stream):
                begin.record()
                if graph:
                    graph.replay()
                else:
                    output = launch()
                end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / repetitions)

        reports = [None] * world
        dist.all_gather_object(
            reports,
            {
                "rank": rank,
                "initialization_ms": initialization_ms,
                "construction_ms": construction_ms,
                "warmup_initialization_ms": warmup_initialization_ms,
                "samples_us": samples,
                "correctness": correctness,
            },
        )
        if rank == 0:
            slowest = [max(report["samples_us"][sample] for report in reports) for sample in range(args.iterations)]
            result = {
                "implementation": "flashinfer-sm100-bf16-mxfp8-cutedsl-megamoe",
                "scope": "routed_experts_only",
                "excluded": ["router", "projections", "shared_expert", "postnorm", "residual"],
                "configuration": {
                    **vars(args),
                    "world_size": world,
                    "kind": kernel_config.kind,
                    "gate_up_clamp": clamp,
                    "knobs": requested_knobs,
                },
                "hardware": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flashinfer": _source_info(flashinfer_root, flashinfer),
                "runtime_versions": _runtime_versions(),
                "communication_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "NCCL_IB_DISABLE",
                        "NCCL_NET",
                        "NCCL_MNNVL_ENABLE",
                        "NVSHMEM_REMOTE_TRANSPORT",
                    )
                },
                "backend_config": asdict(kernel_config),
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
                output_path = Path(args.json_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text + "\n")
    finally:
        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.barrier()
        if layer is not None:
            layer.destroy()
        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
