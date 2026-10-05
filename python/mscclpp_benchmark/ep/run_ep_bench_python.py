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
import importlib.util
import os

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
        "DeepEP V2 uses one ElasticBuffer path and is tuned with --num-sms.",
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
        choices=("bf16", "fp8_e4m3"),
        default="bf16",
        help="MSCCL++ latency dispatch wire format. NCCL-EP itself supports FP8, but its path in this "
        "benchmark is wired BF16-only (the harness does not plumb NCCL-EP's dispatch scales yet).",
    )
    p.add_argument(
        "--combine-mode",
        "--optimized-combine-mode",
        choices=("rank_local_reduce", "direct_send"),
        default="rank_local_reduce",
        help="MSCCL++ latency combine mode (direct_send is bit-exact; rank_local_reduce is faster).",
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
        "layout (NCCL-EP=expert_major, MSCCL++ latency=expert_major, MSCCL++ throughput=token_major, "
        "DeepEP=rank_major). "
        "The explicit value is applied where supported: NCCL-EP latency and DeepEP accept rank_major/expert_major; "
        "NCCL-EP throughput accepts expert_major only in this benchmark. "
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
    if args.dispatch_dtype == "fp8_e4m3" and args.backend != "mscclpp":
        raise SystemExit(
            "--dispatch-dtype fp8_e4m3 is only wired for the mscclpp backend in this benchmark "
            "(NCCL-EP supports FP8 but its path here is BF16-only); use --backend mscclpp"
        )
    if args.dispatch_dtype == "fp8_e4m3" and args.mode != "latency":
        raise SystemExit("MSCCL++ throughput mode currently supports BF16 dispatch only")
    return args


# ============================================================================
# Shared paired benchmark + summary (mirrors NCCL-EP ep_bench).
# ============================================================================
def _flush_l2_cache():
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


def torch_profiler_kernel_us(
    dispatch_fn,
    combine_fn,
    comm,
    num_tests,
    flush_l2=True,
    use_barrier=True,
    barrier=None,
    mid_barrier=None,
    parse_kernels=None,
    graph=None,
):
    """DeepEP bench_kineto-style kernel timing: torch.profiler (CUDA activity)
    over the paired dispatch->combine loop (the default), with a per-iteration L2 flush and a
    cuda._sleep(~10ms) + cross-rank barrier to absorb host launch skew. Returns
    the average per-kernel GPU time (us) for the dispatch and combine kernels,
    matched by name substring in the profiler key_averages() table.

    EP_KINETO_BARRIER_COMBINE=1 inserts a SECOND GPU-side barrier between eager
    dispatch and combine so the combine kernel also enters GPU-aligned across
    ranks. This is the same treatment the FlashInfer harness applies (barrier
    before BOTH phases) to collapse the combine recv-spin skew -- without it the
    single pre-dispatch barrier aligns dispatch but combine drifts again because
    it is a separate launch whose in-kernel arrival-wait absorbs the skew.
    The mid barrier uses a PLAIN NCCL all_reduce (``mid_barrier``), not the
    backend's native barrier.

    NOTE: for DeepEP this is SINGLE-NODE ONLY. Inserting ANY collective (even a
    plain NCCL all_reduce) between DeepEP's dispatch and combine corrupts the
    ElasticBuffer's pending symmetric-memory state and crashes on the multi-node
    scale-out path (Cuda 719 in DeepEP symmetric.hpp). mscclpp / NCCL-EP have
    independent dispatch/combine and tolerate it at any scale.

    EP_KINETO_SEPARATE=1 measures dispatch and combine in TWO separate
    profiled passes -- each a single op per iteration with the barrier immediately
    before it -- exactly like DeepEP's own bench_kineto (which is called once per
    op). This aligns BOTH kernels at entry without ever placing a barrier between
    a paired dispatch->combine (so it is safe for DeepEP multi-node), collapsing
    the combine recv-spin skew. This opt-in mode is limited to backends that can
    repeat each phase independently; MSCCL++ always uses paired calls because
    latency handles are single-use. EP_KINETO_SEPARATE=0 is required
    whenever a backend captures dispatch+combine in ONE CUDA graph: a
    single replay runs both phases, so the separate pass can no longer isolate
    combine and the per-phase split must come from the paired pass instead."""
    import torch.profiler as _tp

    use_mid = graph is None and os.environ.get("EP_KINETO_BARRIER_COMBINE", "0") == "1" and mid_barrier is not None
    separate = graph is None and os.environ.get("EP_KINETO_SEPARATE", "0") == "1"
    # Backend-specific kineto parse: maps a key_averages() table to
    # (dispatch_us, combine_us) using that library's kernel names (see
    # ep_bench_<lib>.parse_kineto_kernels). Fall back to the generic phase-word
    # split when none is supplied.
    if parse_kernels is None:
        parse_kernels = lambda ka: (
            sum_matching_kernel_us(ka, ("dispatch",)),
            sum_matching_kernel_us(ka, ("combine",)),
        )

    def _do_barrier():
        if not use_barrier:
            return
        torch.cuda._sleep(int(2e7))  # ~10 ms GPU spin to absorb host launch skew
        if barrier is not None:
            barrier()  # GPU-side barrier (aligns ranks on-device)
        else:
            comm.barrier()  # Host-only alignment.

    if separate:
        # ---- Two separate passes, each: [flush; barrier; single op] ----
        # Generic two-pass timing method (adopted from DeepEP's bench_kineto,
        # which profiles one op per call): dispatch and combine are each timed in
        # their own profiled loop with the barrier immediately before the op, so
        # both kernels enter GPU-aligned across ranks and the combine recv-spin
        # skew collapses. It applies to every backend -- the dispatch_fn/combine_fn
        # closures are backend-supplied and this loop has no per-library logic.
        # Because no barrier is ever placed BETWEEN a paired dispatch->combine, it
        # is also safe for DeepEP multi-node (a mid-pair collective would corrupt
        # its symmetric-memory state; see the class docstring).
        def _run_pass(op_fn):
            op_fn()  # warm / auto-tune
            torch.cuda.synchronize()
            schedule = _tp.schedule(wait=0, warmup=1, active=1, repeat=1)
            with _tp.profile(activities=[_tp.ProfilerActivity.CUDA], schedule=schedule, acc_events=True) as prof:
                for _ in range(2):
                    for _ in range(num_tests):
                        if flush_l2:
                            _flush_l2_cache()
                        _do_barrier()
                        op_fn()
                    torch.cuda.synchronize()
                    prof.step()
            return prof.key_averages()

        # Dispatch pass.
        ka_d = _run_pass(dispatch_fn)
        # Combine pass: prime one dispatch to obtain a valid combine input, then
        # replay combine alone. The primed dout carries whatever state the backend
        # needs (DeepEP replays its fixed primed handle; mscclpp / NCCL-EP consume
        # this dout each iteration) -- all via the backend-supplied combine_fn.
        dout = dispatch_fn()
        torch.cuda.synchronize()
        ka_c = _run_pass(lambda: combine_fn(dout))
        # Dispatch time from the dispatch pass, combine time from the combine pass.
        return parse_kernels(ka_d)[0], parse_kernels(ka_c)[1]

    # ---- Paired single-pass loop (EP_KINETO_SEPARATE=0) ----
    # Times dispatch and combine in ONE profiled pass over the paired
    # dispatch->combine loop. REQUIRED for single-graph CUDA-graph backends (one
    # replay runs both phases, so the separate pass cannot isolate combine).
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
                    dout = dispatch_fn()
                    if use_mid:
                        mid_barrier()
                    combine_fn(dout)
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
def _capture_paired_graph(dispatch_op, combine_op, prime=True, pre_capture=None, graph_group_size=1):
    """Capture paired dispatch->combine operations and return their CUDA graph.
    One replay runs both phases. dispatch_op/combine_op are the backend's capture-safe
    ops (no CPU sync, no host collective) that go inside the graph.
    ``graph_group_size`` dispatch->combine iterations are captured inside the single
    graph so one replay runs them all -- this amortizes launch overhead and keeps the
    spin-waiting dispatch/combine kernels from being inflated by per-replay launch
    skew (reported times are divided back to per-iteration by the caller). Capture is
    performed on the same non-default stream used for setup and eager calls;
    capture failures propagate instead of reusing potentially invalid state."""
    if prime:
        if pre_capture is not None:
            pre_capture()
        dispatch_op()
        combine_op()
        torch.cuda.synchronize()
    if pre_capture is not None:
        pre_capture()
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
    base_inputs = inputs[0] if isinstance(inputs, list) else inputs
    _, _, _, num_valid_selections = base_inputs
    hidden = args.hidden
    warmup, iters = args.num_warmup, args.num_iters
    disp_elt = 1 if getattr(args, "dispatch_dtype", "bf16") == "fp8_e4m3" else 2
    disp_bytes = num_valid_selections * hidden * disp_elt  # dispatch wire format
    comb_bytes = num_valid_selections * hidden * 2  # BF16 combine output (per ep_bench)

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

    # Kernel-only timing (EP_KERNEL_TIMER=kineto, the default): DeepEP bench_kineto-style
    # torch.profiler pass with an L2 flush and a GPU-side torch NCCL all_reduce barrier
    # per iteration (EP_KINETO_BARRIER=nccl) to align ranks on-device -- skew-free avg.
    use_kineto = os.environ.get("EP_KERNEL_TIMER", "kineto") == "kineto"
    have_kernel = use_kineto

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

    ck_disp = ck_comb = 0.0
    inproc_ok = False
    if use_kineto:
        comm.barrier()
        ck_disp, ck_comb = torch_profiler_kernel_us(
            dispatch_fn,
            combine_fn,
            comm,
            iters,
            barrier=(bench_barrier or nccl_barrier),
            mid_barrier=nccl_barrier,
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
    dispatch_dtype = "FP8 E4M3" if disp_elt == 1 else "BF16"
    if graph is None:
        dispatch_us = [starts[i].elapsed_time(dispatch_ends[i]) * 1e3 for i in range(iters)]
        combine_us = [max(combine_starts[i].elapsed_time(ends[i]) * 1e3, 1e-3) for i in range(iters)]
        timings = [
            (f"Dispatch ({dispatch_dtype})", dispatch_us, disp_bytes),
            ("Combine (BF16)", combine_us, comb_bytes),
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
    kernel_ok = 0
    gk_d_avg = gk_d_min = gk_d_max = 0.0
    gk_c_avg = gk_c_min = gk_c_max = 0.0
    if have_kernel:
        kernel_ok = comm.allreduce(1 if inproc_ok else 0, op=MPI.MIN)
        gk_d_avg, gk_d_min, gk_d_max = _mpi_stats(comm, ck_disp, ck_disp, ck_disp, num_ranks)
        gk_c_avg, gk_c_min, gk_c_max = _mpi_stats(comm, ck_comb, ck_comb, ck_comb, num_ranks)

    if rank == 0:
        if have_kernel:
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
            f"combine={comb_bytes / 1e6:.2f} MB (BF16), selections={num_valid_selections}"
        )


# ----------------------------------------------------------------------------
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
    # Debug aid: EP_FAULTHANDLER_SECS>0 dumps every thread's Python traceback if the
    # process is still alive after N seconds (surfaces the exact hang location under
    # an mpirun timeout). Off unless the env var is set.
    _fh_secs = float(os.environ.get("EP_FAULTHANDLER_SECS", "0") or "0")
    if _fh_secs > 0:
        import faulthandler

        faulthandler.dump_traceback_later(_fh_secs, repeat=True)
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

    # Snapshot the user's EP_KINETO_SEPARATE so we can restore it per backend below
    # (cuda-graph capture toggles it for some backends only -- see the loop).
    _user_kineto_separate = os.environ.get("EP_KINETO_SEPARATE")

    backends = ["nccl", "mscclpp", "deepep"] if args.backend == "all" else [args.backend]
    benchmark_stream = torch.cuda.Stream()
    benchmark_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(benchmark_stream):
            _run_backends(args, comm, rank, num_ranks, inputs, backends)
    finally:
        if _user_kineto_separate is None:
            os.environ.pop("EP_KINETO_SEPARATE", None)
        else:
            os.environ["EP_KINETO_SEPARATE"] = _user_kineto_separate
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _run_backends(args, comm, rank, num_ranks, inputs, backends):
    available_backends = []
    # DeepEP validates loaded libnccl paths at import; import it before NCCL-EP
    # loads libnccl_ep.so, which some DeepEP versions mistake for another runtime.
    for name in reversed(backends):
        package = {"nccl": "nccl", "mscclpp": "mscclpp", "deepep": "deep_ep"}[name]
        available = int(importlib.util.find_spec(package) is not None)
        if comm.allreduce(available, op=MPI.MIN) == 0:
            message = f"backend '{name}' requires the '{package}' Python package on every rank"
            if args.backend != "all":
                raise ImportError(message)
            if rank == 0:
                print(f"\n[skip] {message}", flush=True)
            continue
        importlib.import_module({"nccl": "nccl.ep", "mscclpp": "mscclpp.ep", "deepep": "deep_ep"}[name])
        available_backends.append(name)
    if not available_backends:
        raise RuntimeError("No EP backend was available")

    nccl_barrier = None
    if (
        os.environ.get("EP_KERNEL_TIMER", "kineto") == "kineto"
        and os.environ.get("EP_KINETO_BARRIER", "nccl") == "nccl"
    ):
        nccl_barrier = _init_torch_nccl(comm, rank, num_ranks)
        if rank == 0:
            print("[cfg] kineto barrier: torch NCCL all_reduce (GPU-side)", flush=True)

    user_separate = os.environ.get("EP_KINETO_SEPARATE")
    for name in backends:
        if name not in available_backends:
            continue
        if user_separate is None:
            os.environ.pop("EP_KINETO_SEPARATE", None)
        else:
            os.environ["EP_KINETO_SEPARATE"] = user_separate
        ops = _SETUP[name](args, comm, rank, num_ranks, inputs)
        dispatch_fn = ops["dispatch"]
        combine_fn = ops["combine"]
        teardown = ops["teardown"]
        backend_barrier = ops.get("barrier")
        if ops.get("paired_profile"):
            os.environ["EP_KINETO_SEPARATE"] = "0"
            if user_separate == "1" and rank == 0:
                print("[cfg] mscclpp profiling stays paired: latency handles cannot be combined twice", flush=True)
        graph = None
        spec = ops.get("graph")
        effective_group_size = 1
        try:
            if spec is not None:
                graph = _capture_paired_graph(
                    spec["dispatch"],
                    spec["combine"],
                    pre_capture=spec.get("pre_capture"),
                    graph_group_size=args.iters_per_graph,
                )
                effective_group_size = args.iters_per_graph
                os.environ["EP_KINETO_SEPARATE"] = "0"
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
