#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified NCCL-EP, MSCCL++ EP, and DeepEP Python benchmark.

Each backend uses the same tensors, routing, paired dispatch/combine event loop,
and cross-rank statistics. Kineto optionally reports kernel-only phase times.
Input tensors and synthetic BF16 expert output are prepared outside the timed loop.
NCCL-EP runs first, matching the ordering in the feature/ep benchmark.

Launch with mpirun and mpi4py. MSCCL++ requires CUDA
SM90+ and a single IPC domain. Latency graph replay requires at least two paired
operations per graph; throughput supports graph replay with reusable preparation.
See python/mscclpp/ep/README.md for dependencies,
launch examples, layouts, and timing limitations.
"""

from __future__ import annotations

import argparse
import importlib

import torch

from .ep_bench_common import (
    MPI,
    init_mpi,
    make_inputs,
    _init_torch_nccl,
    _mpi_stats,
    sum_matching_kernel_us,
)
from .ep_bench_mscclpp import setup_mscclpp, parse_kineto_kernels as mscclpp_parse_kineto
from .ep_bench_nccl import setup_nccl, parse_kineto_kernels as nccl_parse_kineto
from .ep_bench_deepep import setup_deepep, parse_kineto_kernels as deepep_parse_kineto


# ----------------------------------------------------------------------------
# CLI — shared workload and backend/mode selection.
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified EP benchmark across Python backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["mscclpp", "nccl", "deepep", "all"],
        default="all",
        help="which backend(s) to benchmark (all runs nccl, mscclpp, deepep; missing optional packages are reported)",
    )
    p.add_argument(
        "--mode",
        choices=("latency", "throughput"),
        default="latency",
        help="algorithm family for backends with an explicit mode (MSCCL++ and NCCL-EP). "
        "DeepEP V2.5 uses one EPBuffer path and is tuned with --num-sms.",
    )
    p.add_argument(
        "--num-sms",
        type=int,
        default=0,
        help="Communication SM/block budget. DeepEP interprets it as SMs; MSCCL++ maps it to the total "
        "communication block count. 0 uses backend defaults.",
    )
    p.add_argument("-t", "--num-tokens", type=int, default=128, help="tokens per rank")
    p.add_argument("-d", "--hidden", type=int, default=7168, help="hidden dimension")
    p.add_argument("-k", "--num-topk", type=int, default=8, help="top-k experts per token")
    p.add_argument("-e", "--num-experts", type=int, default=256, help="global number of experts")
    p.add_argument("-w", "--num-warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("-i", "--num-iters", type=int, default=50, help="timed iterations")
    p.add_argument("--seed", type=int, default=0xB3C4, help="per-rank RNG seed base")
    p.add_argument(
        "--dispatch-dtype",
        choices=("bf16", "fp8_e4m3", "fp8_e4m3_static", "mxfp8_e4m3"),
        default="bf16",
        help="dispatch wire format. MSCCL++ latency supports bf16, fp8_e4m3 (block-scaled on expert_major "
        "or static FP8 on rank_major), fp8_e4m3_static (rank_major), and mxfp8_e4m3 (rank_major); "
        "MSCCL++ throughput supports bf16 and fp8_e4m3. DeepEP EPBuffer supports all four "
        "(pre-quantized input). NCCL-EP supports bf16 and fp8_e4m3 (latency expert_major or throughput).",
    )
    p.add_argument(
        "--combine-dtype",
        choices=("bf16", "fp8_e4m3_static"),
        default="bf16",
        help="MSCCL++ latency rank_major combine input format (fp8_e4m3_static requires static FP8 dispatch).",
    )
    p.add_argument(
        "--fp8-scale",
        type=float,
        default=1.0,
        help="Static FP8 multiplier scale for MSCCL++ and DeepEP.",
    )
    p.add_argument(
        "--fp8-input-mode",
        choices=("quantize", "prequantized"),
        default="quantize",
        help="MSCCL++ latency rank_major static FP8 dispatch input mode (fused BF16 quantize or prequantized E4M3).",
    )
    p.add_argument(
        "--combine-mode",
        "--optimized-combine-mode",
        choices=("rank_local_reduce", "direct_send"),
        default="rank_local_reduce",
        help="combine reduction mode for MSCCL++ latency and DeepEP EPBuffer "
        "(direct_send / allow_multiple_reduction=0 vs rank_local_reduce / allow_multiple_reduction=1).",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture dispatch and combine as one CUDA graph and replay it in the timed loop.",
    )
    p.add_argument(
        "--iters-per-graph",
        dest="iters_per_graph",
        type=int,
        default=50,
        help="with --cuda-graph, number of dispatch->combine iterations captured INSIDE one CUDA "
        "graph (replayed as a unit; default 50). >1 amortizes launch overhead and keeps the "
        "spin-waiting dispatch/combine kernels from being inflated by per-replay launch skew; "
        "reported times are per iteration. Automatically treated as 1 without --cuda-graph.",
    )
    p.add_argument(
        "--ep-layout",
        choices=["token_major", "rank_major", "expert_major"],
        default=None,
        help="received-token dispatch layout. When omitted, each backend uses its own default "
        "layout (NCCL-EP latency=expert_major, NCCL-EP throughput=token_major, "
        "MSCCL++ latency=expert_major, MSCCL++ throughput=token_major, "
        "DeepEP=rank_major). "
        "The explicit value is applied where supported: NCCL-EP latency and DeepEP accept rank_major/expert_major; "
        "NCCL-EP throughput accepts token_major/expert_major. "
        "MSCCL++ latency accepts rank_major/expert_major and throughput accepts rank_major/token_major. "
        "Unsupported layouts are rejected.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="mscclpp: run a one-time combine correctness check before timing.",
    )
    args = p.parse_args()
    if args.num_tokens <= 0 or args.num_experts <= 0:
        raise SystemExit("--num-tokens and --num-experts must be positive")
    if args.num_topk <= 0 or args.num_topk > args.num_experts:
        raise SystemExit("--num-topk must be in [1, num-experts]")
    if args.hidden <= 0:
        raise SystemExit("--hidden must be positive")
    if args.num_warmup < 0 or args.num_iters <= 0:
        raise SystemExit("--num-warmup must be non-negative and --num-iters must be positive")
    if args.num_sms < 0:
        raise SystemExit("--num-sms must be non-negative")
    if args.iters_per_graph <= 0:
        raise SystemExit("--iters-per-graph must be positive")
    if not args.cuda_graph:
        # Grouping only applies to graph capture; treat as 1 for eager runs so the
        # non-1 default does not error a plain (non-graph) benchmark.
        args.iters_per_graph = 1
    if args.dispatch_dtype in ("fp8_e4m3_static", "mxfp8_e4m3"):
        if args.backend in ("nccl", "all"):
            raise SystemExit(
                f"--dispatch-dtype {args.dispatch_dtype} is supported only by mscclpp and deepep "
                "(NCCL-EP supports only bf16 and fp8_e4m3)"
            )
        if args.backend == "mscclpp" and (args.mode != "latency" or args.ep_layout != "rank_major"):
            raise SystemExit(
                f"MSCCL++ --dispatch-dtype {args.dispatch_dtype} requires --mode latency --ep-layout rank_major"
            )
        if args.backend == "deepep" and args.ep_layout == "expert_major":
            raise SystemExit(
                f"DeepEP --dispatch-dtype {args.dispatch_dtype} requires rank_major layout (omit --ep-layout or pass rank_major)"
            )
    if (
        args.dispatch_dtype == "fp8_e4m3"
        and args.backend == "nccl"
        and args.mode == "latency"
        and args.ep_layout == "rank_major"
    ):
        raise SystemExit("NCCL-EP latency rank_major does not support --dispatch-dtype fp8_e4m3")
    if args.combine_dtype != "bf16":
        if args.backend != "mscclpp" or args.mode != "latency" or args.ep_layout != "rank_major":
            raise SystemExit(
                "--combine-dtype fp8_e4m3_static requires --backend mscclpp --mode latency --ep-layout rank_major"
            )
        if args.dispatch_dtype not in ("fp8_e4m3", "fp8_e4m3_static"):
            raise SystemExit("--combine-dtype fp8_e4m3_static requires static FP8 dispatch")
    if not (args.fp8_scale > 0.0 and torch.isfinite(torch.tensor(args.fp8_scale))):
        raise SystemExit("--fp8-scale must be finite and positive")
    return args


# ============================================================================
# Shared paired benchmark + summary (mirrors NCCL-EP ep_bench).
# ============================================================================
def _flush_l2_cache():
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


def torch_profiler_kernel_us(
    dispatch_fn,
    combine_fn,
    num_tests,
    flush_l2=True,
    barrier=None,
    parse_kernels=None,
    graph=None,
):
    """DeepEP bench_kineto-style kernel timing: torch.profiler (CUDA activity)
    over the paired dispatch->combine loop, with a per-iteration L2 flush and a
    cuda._sleep(~10ms) + cross-rank GPU barrier to absorb host launch skew. Returns
    the average per-kernel GPU time (us) for the dispatch and combine kernels,
    matched by name substring in the profiler key_averages() table."""
    import torch.profiler as _tp

    if parse_kernels is None:
        parse_kernels = lambda ka: (
            sum_matching_kernel_us(ka, ("dispatch",)),
            sum_matching_kernel_us(ka, ("combine",)),
        )

    def _do_barrier():
        torch.cuda._sleep(int(2e7))  # ~10 ms GPU spin to absorb host launch skew
        if barrier is not None:
            barrier()  # GPU-side barrier (aligns ranks on-device)

    if graph is None:
        combine_fn(dispatch_fn())
    else:
        graph.replay()
    torch.cuda.synchronize()
    schedule = _tp.schedule(wait=0, warmup=1, active=1, repeat=1)
    with _tp.profile(activities=[_tp.ProfilerActivity.CUDA], schedule=schedule, acc_events=True) as prof:
        for _ in range(2):
            for _ in range(num_tests):
                if flush_l2:
                    _flush_l2_cache()
                _do_barrier()
                if graph is None:
                    combine_fn(dispatch_fn())
                else:
                    graph.replay()
            torch.cuda.synchronize()
            prof.step()

    ka = prof.key_averages()
    return parse_kernels(ka)


# ============================================================================
# CUDA-graph capture (owned by the harness, not the backends). Every EP backend
# replays dispatch+combine as ONE combined graph, so this single helper captures
# the paired op from any backend; the backend only supplies its capture-safe ops.
# ============================================================================
def _capture_paired_graph(dispatch_op, combine_op, graph_group_size=1):
    """Capture paired dispatch->combine operations and return their CUDA graph.
    One replay runs both phases. dispatch_op/combine_op are the backend's capture-safe
    ops (no CPU sync, no host collective) that go inside the graph.
    ``graph_group_size`` dispatch->combine iterations are captured inside the single
    graph so one replay runs them all -- this amortizes launch overhead and keeps the
    spin-waiting dispatch/combine kernels from being inflated by per-replay launch
    skew (reported times are divided back to per-iteration by the caller). Capture is
    performed on the same non-default stream used for setup and eager calls;
    capture failures propagate instead of reusing potentially invalid state."""
    dispatch_op()
    combine_op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(graph_group_size):
            dispatch_op()
            combine_op()

    return graph


def run_backend(
    name,
    args,
    comm,
    rank,
    num_ranks,
    inputs,
    dispatch_fn,
    combine_fn,
    nccl_barrier=None,
    bench_barrier=None,
    parse_kernels=None,
    graph_group_size=1,
    graph=None,
):
    _, _, _, num_valid_selections = inputs
    hidden = args.hidden
    warmup, iters = args.num_warmup, args.num_iters
    raw_disp_dtype = args.dispatch_dtype
    raw_comb_dtype = args.combine_dtype
    disp_elt = 2 if raw_disp_dtype == "bf16" else 1
    comb_elt = 1 if raw_comb_dtype == "fp8_e4m3_static" else 2
    disp_bytes = num_valid_selections * hidden * disp_elt  # dispatch wire format
    comb_bytes = num_valid_selections * hidden * comb_elt  # combine wire format

    stream = torch.cuda.current_stream()

    # --- Warmup (paired). ---
    for _ in range(warmup):
        if graph is None:
            combine_fn(dispatch_fn())
        else:
            graph.replay()
    if warmup:
        stream.synchronize()
        comm.barrier()

    # Kernel-only timing: DeepEP bench_kineto-style torch.profiler pass with an
    # L2 flush and a GPU-side barrier per iteration to align ranks on-device.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    if graph is None:
        dispatch_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        combine_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # --- Timed loop (paired); no per-iter sync/barrier -- kernels pipeline back-to-back. ---
    for i in range(iters):
        starts[i].record(stream)
        if graph is None:
            dout = dispatch_fn()
            dispatch_ends[i].record(stream)
            combine_starts[i].record(stream)
            combine_fn(dout)
        else:
            graph.replay()
        ends[i].record(stream)

    torch.cuda.synchronize()

    comm.barrier()
    ck_disp, ck_comb = torch_profiler_kernel_us(
        dispatch_fn,
        combine_fn,
        iters,
        barrier=bench_barrier or nccl_barrier,
        parse_kernels=parse_kernels,
        graph=graph,
    )
    inproc_ok = ck_disp > 0.0 and ck_comb > 0.0

    # --- Per-iter times (ms->us), trim the first (warmup outlier). ---
    # graph_group_size is the number of dispatch->combine iterations captured in one
    # graph (the --iters-per-graph arg; 1 when this backend is not graph-captured).
    # One replay ran them all, so divide the per-replay time
    # back to per-iteration. Kernel-only kineto is already per-iteration (its
    # per-launch average divides by the kernel count, which scales with the group).
    group = graph_group_size if graph is not None else 1
    total_us = [starts[i].elapsed_time(ends[i]) * 1e3 / group for i in range(iters)]
    timings = [("Total (D+C)", total_us, disp_bytes + comb_bytes)]
    is_static_fp8_map = raw_disp_dtype == "fp8_e4m3" and (
        (name == "mscclpp" and args.mode == "latency" and args.ep_layout == "rank_major")
        or (name == "deepep" and args.ep_layout != "expert_major")
    )
    dispatch_dtype = {
        "bf16": "BF16",
        "fp8_e4m3": "FP8 E4M3 Static" if is_static_fp8_map else "FP8 E4M3",
        "fp8_e4m3_static": "FP8 E4M3 Static",
        "mxfp8_e4m3": "MXFP8 E4M3",
    }.get(raw_disp_dtype, raw_disp_dtype.upper())
    combine_dtype = "FP8 E4M3 Static" if comb_elt == 1 else "BF16"
    if graph is None:
        dispatch_us = [starts[i].elapsed_time(dispatch_ends[i]) * 1e3 for i in range(iters)]
        combine_us = [max(combine_starts[i].elapsed_time(ends[i]) * 1e3, 1e-3) for i in range(iters)]
        timings = [
            (f"Dispatch ({dispatch_dtype})", dispatch_us, disp_bytes),
            (f"Combine ({combine_dtype})", combine_us, comb_bytes),
            *timings,
        ]

    if rank == 0:
        print(f"\n=== Summary [{name}] ({args.mode.title()}, across {num_ranks} ranks) ===")
        print("\n--- Host-observed performance ---")
        if graph is not None:
            print("Paired CUDA graph: event timing reports only total D+C; phase times come from kineto.")

    for label, times, nbytes in timings:
        if iters > 1:
            times = times[1:]
        avg = sum(times) / len(times)
        g_avg, g_min, g_max = _mpi_stats(comm, avg, min(times), max(times), num_ranks)
        throughputs = comm.gather((nbytes / 1e9) / (avg * 1e-6), root=0)
        if rank == 0:
            low_rank = min(range(num_ranks), key=lambda r: throughputs[r])
            high_rank = max(range(num_ranks), key=lambda r: throughputs[r])
            print(f"{label}: avg={g_avg:.2f} us, min={g_min:.2f} us, max={g_max:.2f} us")
            print(
                f"                  throughput: avg={(nbytes / 1e9) / (g_avg * 1e-6):.2f} GB/s, "
                f"min={throughputs[low_rank]:.2f} GB/s (rank {low_rank}), "
                f"max={throughputs[high_rank]:.2f} GB/s (rank {high_rank})"
            )

    # --- Kernel-only (torch kineto) cross-rank reduction. The LL dispatch
    # kernel ends in a cross-rank recv spin-wait, so a lagging rank's device time
    # includes wait skew; the cross-rank MIN (the rank that did not wait) is the
    # representative kernel floor. Combine has little recv-spin and is stable. ---
    kernel_ok = comm.allreduce(1 if inproc_ok else 0, op=MPI.MIN)
    gk_d_avg, gk_d_min, gk_d_max = _mpi_stats(comm, ck_disp, ck_disp, ck_disp, num_ranks)
    gk_c_avg, gk_c_min, gk_c_max = _mpi_stats(comm, ck_comb, ck_comb, ck_comb, num_ranks)

    if rank == 0:
        _kt_hdr = "torch kineto (per-iter barrier + L2 flush)"
        print(f"\n--- Kernel-only performance ({_kt_hdr}) ---")
        if kernel_ok:
            # Report BOTH min and avg for dispatch and combine. The LL dispatch
            # kernel ends in a cross-rank recv spin-wait, so its avg/max carry
            # wait skew on lagging ranks; the cross-rank MIN is the representative
            # kernel floor. Combine has little recv-spin, so its min ~ avg.
            print(
                f"Dispatch:    avg={gk_d_avg:.2f} us, min={gk_d_min:.2f} us (representative), "
                f"max={gk_d_max:.2f} us [avg/max carry recv-spin skew on lagging ranks]"
            )
            print(
                f"                  throughput: avg={(disp_bytes / 1e9) / (gk_d_avg * 1e-6):.2f} GB/s, "
                f"@min={(disp_bytes / 1e9) / (gk_d_min * 1e-6):.2f} GB/s"
            )
            print(f"Combine:     avg={gk_c_avg:.2f} us, min={gk_c_min:.2f} us, max={gk_c_max:.2f} us")
            print(f"                  throughput: avg={(comb_bytes / 1e9) / (gk_c_avg * 1e-6):.2f} GB/s")
            print(
                f"Total (D+C): avg={gk_d_avg + gk_c_avg:.2f} us (dispatch avg + combine avg), "
                f"floor={gk_d_min + gk_c_avg:.2f} us (dispatch min + combine avg)"
            )
        else:
            print("  NOTE: kineto captured 0 EP kernels (collector unavailable).")

        print(
            f"\nLogical routed bytes: dispatch={disp_bytes / 1e6:.2f} MB ({dispatch_dtype}), "
            f"combine={comb_bytes / 1e6:.2f} MB ({combine_dtype}), selections={num_valid_selections}"
        )


# ----------------------------------------------------------------------------
# Backend registry.
# ----------------------------------------------------------------------------
_SETUP = {
    "mscclpp": setup_mscclpp,
    "nccl": setup_nccl,
    "deepep": setup_deepep,
}
# Per-backend kineto kernel-name parse, owned by each backend module.
_PARSE_KINETO = {
    "mscclpp": mscclpp_parse_kineto,
    "nccl": nccl_parse_kineto,
    "deepep": deepep_parse_kineto,
}


def main() -> None:
    args = parse_args()
    comm, rank, num_ranks = init_mpi()
    if args.num_experts % num_ranks:
        raise ValueError("num_experts must be divisible by num_ranks")
    # Single routing input reused for every captured iteration. --iters-per-graph
    # replays the SAME paired dispatch->combine N times inside one graph to amortize
    # launch overhead; it does not need N distinct routings.
    inputs = make_inputs(
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        rank,
        args.seed,
    )

    backends = ["nccl", "mscclpp", "deepep"] if args.backend == "all" else [args.backend]
    benchmark_stream = torch.cuda.Stream()
    benchmark_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(benchmark_stream):
            _run_backends(args, comm, rank, num_ranks, inputs, backends)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _run_backends(args, comm, rank, num_ranks, inputs, backends):
    available_backends = []
    # DeepEP validates loaded libnccl paths at import; import it before NCCL-EP
    # loads libnccl_ep.so, which some DeepEP versions mistake for another runtime.
    for name in reversed(backends):
        package = {"nccl": "nccl.ep", "mscclpp": "mscclpp.ep", "deepep": "deep_ep"}[name]
        root_package = package.split(".", 1)[0]
        try:
            importlib.import_module(package)
            available = 1
        except ModuleNotFoundError as exc:
            if exc.name not in (package, root_package):
                raise
            available = 0
        if comm.allreduce(available, op=MPI.MIN) == 0:
            message = f"backend '{name}' requires the '{package}' Python package on every rank"
            if args.backend != "all":
                raise ModuleNotFoundError(message, name=package)
            if rank == 0:
                print(f"\n[skip] {message}", flush=True)
            continue
        available_backends.append(name)
    if not available_backends:
        raise RuntimeError("No EP backend was available")

    nccl_barrier = _init_torch_nccl(comm, rank, num_ranks)
    if rank == 0:
        print("[cfg] kineto barrier: torch NCCL all_reduce (GPU-side)", flush=True)

    for name in backends:
        if name not in available_backends:
            continue
        ops = _SETUP[name](args, comm, rank, num_ranks, inputs)
        dispatch_fn = ops["dispatch"]
        combine_fn = ops["combine"]
        teardown = ops["teardown"]
        backend_barrier = ops.get("barrier")
        graph = None
        spec = ops.get("graph")
        effective_group_size = 1
        try:
            if spec is not None:
                graph = _capture_paired_graph(
                    spec["dispatch"],
                    spec["combine"],
                    graph_group_size=args.iters_per_graph,
                )
                effective_group_size = args.iters_per_graph
                if rank == 0:
                    print(
                        f"[cfg] {name} cuda_graph captured "
                        f"(single graph; dispatch+combine; iters_per_graph={effective_group_size})",
                        flush=True,
                    )
            run_backend(
                name,
                args,
                comm,
                rank,
                num_ranks,
                inputs,
                dispatch_fn,
                combine_fn,
                nccl_barrier=nccl_barrier,
                bench_barrier=backend_barrier,
                parse_kernels=_PARSE_KINETO[name],
                graph_group_size=effective_group_size,
                graph=graph,
            )
        finally:
            torch.cuda.synchronize()
            comm.barrier()
            # Drop the operation closures and the captured graph BEFORE teardown frees
            # the buffers/handles the graph captured.
            dispatch_fn = combine_fn = None
            graph = None
            spec = None
            backend_barrier = None
            ops.clear()
            teardown()
            teardown = None
            comm.barrier()


if __name__ == "__main__":
    main()
