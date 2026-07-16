#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified steady-state low-latency EP benchmark for MSCCL++.

Why this exists
---------------
``ep_bench`` is the reference NCCL-EP micro-benchmark. This script uses the same
workload, byte accounting, CUDA-event timing, and summary format while replacing
the NCCL-EP collective API with MSCCL++ ``MoECommunicator`` calls.

Iterations are queued as dispatch→combine pairs on one CUDA stream and
synchronized once, matching ``test_low_latency_multirank.py --bench`` and
measuring steady-state device latency without Python rank-launch skew.

* **Paired** dispatch→combine ordering on the same CUDA stream.
* **Per-iteration CUDA events** recorded around each dispatch/combine operation.
* **Skip the first timed iteration** (warmup outlier) — matches ``ep_bench``'s
  ``calc_stats`` which trims ``times[0]`` when ``num_iters > 1``.
* **Byte accounting** uses expert selections for expert-major output and unique
  ``(token, destination rank)`` rows for token-major output.
* **Cross-rank reduction** identical to ``printLowLatencyResults``: latency
  ``avg = mean``, ``min = MIN``, ``max = MAX``; per-rank throughput min/max are
  tagged with the owning rank (``MPI_MINLOC`` / ``MPI_MAXLOC`` analog).
* **Output** mirrors ``ep_bench``'s ``=== Summary (Low Latency, across N ranks) ===``
  block so the two runs can be diffed directly.

CLI mirrors ``ep_bench``'s LL-relevant flags (long + short):
    -t/--num-tokens   tokens per rank            (ep_bench LL default 128)
    -d/--hidden       hidden dim                 (7168)
    -k/--num-topk     top-k experts per token    (8)
    -e/--num-experts  global experts             (256)
    -w/--num-warmup   warmup iterations          (10)
    -i/--num-iters    timed iterations           (50)

Fidelity note
-------------
``ep_bench`` is C++/MPI; MSCCL++ EP's LL API is Python/torch. Synchronized
per-call pacing makes early ranks wait inside the LL receive-spin for later
Python ranks, so this benchmark reports steady-state queued latency. Host launch
latency still sits outside the CUDA events. For a pure kernel number, run under
``nsys``/CUPTI as with ``ep_bench``'s ``--- Kernel-only ---`` section.

Launch
------
Manual per-rank env (DSM hostnames break torchrun rendezvous on these nodes):
    RANK=.. LOCAL_RANK=.. WORLD_SIZE=.. MASTER_ADDR=.. MASTER_PORT=.. \
        python ep_bench_ll.py -t 128 -d 7168 -k 8 -e 256 -w 10 -i 50
Single node (4/8 GPU):
    torchrun --standalone --nproc_per_node=4 ep_bench_ll.py -e 128
"""

from __future__ import annotations

import argparse
import os
import random

# Quiet ProcessGroupNCCL's heartbeat monitor before importing torch.distributed
# (same rationale as test_low_latency_multirank.py).
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------
# CLI — mirrors ep_bench.cu's getopt flags for the LL path.
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MSCCL++ EP steady-state low-latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Env fallbacks keep the existing MSCCLPP_EP_BENCH_* launchers working.
    p.add_argument(
        "-a",
        "--algorithm",
        default="ll",
        choices=["ll", "low-latency"],
        help="algorithm mode (only LL is implemented here)",
    )
    p.add_argument(
        "-t",
        "--num-tokens",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_TOKENS", "128")),
        help="tokens per rank (ep_bench LL max_tokens_per_rank)",
    )
    p.add_argument(
        "-d",
        "--hidden",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_HIDDEN", "7168")),
        choices=(4096, 6656, 7168, 8192, 9216),
        help="hidden dimension",
    )
    p.add_argument(
        "-k",
        "--num-topk",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_TOPK", "8")),
        choices=range(1, 10),
        help="top-k experts per token",
    )
    p.add_argument(
        "-e",
        "--num-experts",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_EXPERTS", "256")),
        help="global number of experts",
    )
    p.add_argument(
        "-w",
        "--num-warmup",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_WARMUP", "10")),
        help="warmup iterations",
    )
    p.add_argument(
        "-i",
        "--num-iters",
        type=int,
        default=int(os.environ.get("MSCCLPP_EP_BENCH_ITERS", "50")),
        help="timed iterations",
    )
    p.add_argument(
        "--dispatch-dtype",
        choices=("bf16", "fp8_e4m3"),
        default="bf16",
        help="low-latency dispatch payload format",
    )
    p.add_argument(
        "--combine-mode",
        choices=("rank_local_reduce", "direct_send"),
        default="rank_local_reduce",
        help="low-latency combine algorithm",
    )
    p.add_argument(
        "--output-layout",
        choices=("expert_major", "token_major"),
        default="expert_major",
        help="low-latency dispatch output layout",
    )
    p.add_argument("--num-blocks", type=int, default=130, help="total low-latency dispatch blocks")
    p.add_argument(
        "--no-kernel-timing",
        dest="kernel_timing",
        action="store_false",
        help="disable the CUPTI/torch.profiler kernel-only measurement pass "
        "(on by default, mirrors ep_bench's CUPTI KernelTimer)",
    )
    p.add_argument(
        "--cupti-region",
        action="store_true",
        help="bracket ONLY the timed loop with cudaProfilerStart/Stop (for nsys "
        "--capture-range=cudaProfilerApi) so an external CUPTI collector times "
        "exactly the post-warmup dispatch/combine kernels, like ep_bench's "
        "KernelTimer.start()-after-warmup. Skips the in-process torch.profiler "
        "pass; kernel numbers come from nsys.",
    )
    p.add_argument(
        "--cupti-inproc",
        action="store_true",
        help="use the in-process CUPTI collector (libcupti_kernel_timer.so, a faithful "
        "port of ep_bench's KernelTimer): CUPTI Activity API records per-kernel GPU "
        "time over the post-warmup timed loop, near-zero host perturbation, and works "
        "multinode without nsys. Matches mangled dispatch/combine kernel names and "
        "replaces the torch.profiler pass.",
    )
    p.add_argument("--seed", type=int, default=0xB3C4, help="per-rank RNG seed base")
    args = p.parse_args()
    if args.hidden not in (4096, 6656, 7168, 8192, 9216):
        p.error("--hidden must be one of 4096, 6656, 7168, 8192, 9216")
    if not 1 <= args.num_topk <= 9:
        p.error("--num-topk must be in [1, 9]")
    if args.num_tokens <= 0 or args.num_experts <= 0:
        p.error("--num-tokens and --num-experts must be positive")
    if args.num_topk > args.num_experts:
        p.error("--num-topk must not exceed --num-experts")
    if args.num_warmup < 0 or args.num_iters <= 0:
        p.error("--num-warmup must be non-negative and --num-iters must be positive")
    return args


def init_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, local_rank, dist.new_group(list(range(world_size)))


def _reduce_scalar(value: float, op, group) -> float:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=op, group=group)
    return t.item()


def _gather_scalars(value: float, num_ranks: int, group) -> list:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    out = [torch.zeros_like(t) for _ in range(num_ranks)]
    dist.all_gather(out, t, group=group)
    return [float(x.item()) for x in out]


def _profile_paired_kernels(dispatch_fn, combine_fn, iters: int, stream, group, rank: int):
    """Kernel-only dispatch/combine device time (us/iter) via torch.profiler.

    Mirrors ep_bench's CUPTI ``KernelTimer``: it profiles the SAME paired
    ``dispatch -> sync -> combine -> sync -> barrier`` loop used for the
    host-observed measurement. Profiling the *paired* loop (rather than isolated
    dispatch-only / combine-only loops) is essential: the LL dispatch kernel
    ends with a cross-rank receive spin-wait, and without the per-iter barrier
    the ranks drift out of lockstep so that spin balloons to milliseconds on the
    laggards. The barrier keeps every rank aligned at each iteration boundary, so
    the recv-wait stays bounded -- exactly why ep_bench times the paired loop.

    Kernels are bucketed by name substring ``dispatch`` / ``combine`` (the mscclpp
    LL kernels demangle to ``mscclpp::ep::low_latency::dispatch<...>`` /
    ``::combine<...>``), matching ep_bench's ``get_avg_us("dispatch"/"combine")``.
    All other device activity (the pacing barrier's NCCL kernel, memcpy/memset)
    is ignored.
    """
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            dout = dispatch_fn()
            stream.synchronize()
            combine_fn(dout)
            stream.synchronize()
            dist.barrier(group=group)
        torch.cuda.synchronize()

    disp_us = 0.0
    comb_us = 0.0
    dbg = []
    for e in prof.key_averages():
        dev_us = getattr(e, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(e, "self_cuda_time_total", 0.0)
        if not dev_us or dev_us <= 0:
            continue
        low = str(e.key).lower()
        if "memcpy" in low or "memset" in low:
            continue  # CUPTI KernelTimer counts KERNEL activities only
        if "dispatch" in low:
            disp_us += dev_us
        elif "combine" in low:
            comb_us += dev_us
        dbg.append((dev_us, str(e.key)))

    if os.environ.get("MSCCLPP_EP_KDEBUG", "0") == "1" and rank == 0:
        dbg.sort(reverse=True)
        print(f"[kdebug] top device activities (self device us/iter over {iters} iters):", flush=True)
        for us, name in dbg[:10]:
            print(f"    {us / iters:8.2f} us/iter  {name[:90]}", flush=True)

    return disp_us / iters, comb_us / iters


class _InProcCupti:
    """In-process CUPTI kernel timer, a faithful analog of ep_bench's KernelTimer.

    Loads ``libcupti_kernel_timer.so`` (built from cupti_kernel_timer.cpp, sitting
    next to this file) via ctypes and drives the CUPTI Activity API directly:
    ``start()`` after warmup, ``stop()`` after the timed loop, then
    ``avg_us("dispatch"/"combine")`` buckets recorded kernels by mangled-name
    substring -- exactly ep_bench's methodology, with near-zero host perturbation
    (out-of-band buffer callbacks), so the LL dispatch recv-spin is measured
    cleanly rather than being serialized by an in-process tracer.
    """

    def __init__(self):
        import ctypes
        import os as _os

        so = _os.environ.get(
            "MSCCLPP_EP_CUPTI_TIMER_LIB",
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "build", "libcupti_kernel_timer.so"),
        )
        self.lib = ctypes.CDLL(so)
        self.lib.kt_start.restype = ctypes.c_int
        self.lib.kt_stop.restype = ctypes.c_int
        self.lib.kt_get_avg_us.restype = ctypes.c_double
        self.lib.kt_get_avg_us.argtypes = [ctypes.c_char_p]
        self.lib.kt_get_count.restype = ctypes.c_long
        self.lib.kt_get_count.argtypes = [ctypes.c_char_p]

    def start(self) -> int:
        return int(self.lib.kt_start())

    def stop(self) -> int:
        return int(self.lib.kt_stop())

    def avg_us(self, substr: str) -> float:
        return float(self.lib.kt_get_avg_us(substr.encode()))

    def count(self, substr: str) -> int:
        return int(self.lib.kt_get_count(substr.encode()))


def main() -> None:
    args = parse_args()
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    ep_group = CommGroup(torch_group=group)

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    warmup = args.num_warmup
    iters = args.num_iters
    assert num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    num_local_experts = num_experts // num_ranks
    dispatch_data_type = {
        "bf16": ep.DispatchDataType.BF16,
        "fp8_e4m3": ep.DispatchDataType.FP8_E4M3,
    }[args.dispatch_dtype]
    combine_mode = {
        "rank_local_reduce": ep.CombineMode.RANK_LOCAL_REDUCE,
        "direct_send": ep.CombineMode.DIRECT_SEND,
    }[args.combine_mode]
    output_layout = {
        "expert_major": ep.DispatchLayout.EXPERT_MAJOR,
        "token_major": ep.DispatchLayout.TOKEN_MAJOR,
    }[args.output_layout]
    if output_layout == ep.DispatchLayout.TOKEN_MAJOR and combine_mode != ep.CombineMode.RANK_LOCAL_REDUCE:
        raise ValueError("token-major output requires rank_local_reduce combine")
    dispatch_quant = (
        None if dispatch_data_type == ep.DispatchDataType.BF16 else ep.QuantConfig(format=dispatch_data_type)
    )
    dispatch_dtype = torch.bfloat16 if dispatch_quant is None else torch.float8_e4m3fn
    dispatch_label = "BF16" if dispatch_quant is None else "FP8_E4M3"

    # bf16 precision anchor (same convention as test_low_latency_multirank.py).
    rank_offset = 128
    assert num_ranks - rank_offset < 257, "too many ranks for bf16 precision anchor"

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # ---- Inputs (mirror ep_bench setupLowLatencyTensors: BF16 tokens + routing).
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()

    # ep_bench byte accounting: num_valid_selections = count(topk_idx >= 0). We
    # keep every selection valid (a full LL load), so this equals num_tokens*top_k.
    if output_layout == ep.DispatchLayout.EXPERT_MAJOR:
        num_dispatch_rows = int((topk_idx >= 0).sum().item())
    else:
        destination_mask = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device="cuda")
        for topk_slot in range(num_topk):
            expert = topk_idx[:, topk_slot]
            valid = expert >= 0
            destination_mask[valid, expert[valid] // num_local_experts] = True
        num_dispatch_rows = int(destination_mask.sum().item())
    dispatch_bytes_per_token = hidden * 2 if dispatch_quant is None else hidden + hidden // 128 * 4
    disp_bytes = num_dispatch_rows * dispatch_bytes_per_token
    comb_bytes = num_dispatch_rows * hidden * 2

    if rank == 0:
        print(
            f"[cfg] algorithm=LOW_LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} top_k={num_topk} warmup={warmup} iters={iters} "
            f"dispatch_dtype={args.dispatch_dtype} combine_mode={args.combine_mode} "
            f"output_layout={args.output_layout} "
            f"pacing=batched_steady_state",
            flush=True,
        )

    # High-level MoE communicator (feature/ep). LOW_LATENCY mode selects the LL
    # backend; dispatch/combine run the full (send+recv) op inline on the stream.
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        low_latency_num_blocks=args.num_blocks,
        low_latency_combine_mode=combine_mode,
        output_layout=output_layout,
        quant=dispatch_quant,
    )
    assert moe_comm.is_available()
    if rank == 0:
        print(f"[cfg] MoECommunicator is_internode={moe_comm.is_internode()}", flush=True)

    # ---- Hoist dispatch/combine output tensors out of the timed loop (ep_bench
    # preallocates all EP tensors before benchmarking; matching that keeps the
    # timed region kernel-bound rather than allocator-bound). The communicator
    # owns its src_info/layout_range/count buffers internally; we only supply the
    # dispatch output buffer and the combine output tensor.
    output_shape = (
        (num_local_experts, num_ranks * num_tokens, hidden)
        if output_layout == ep.DispatchLayout.EXPERT_MAJOR
        else (num_ranks * num_tokens, hidden)
    )
    output_buffer = torch.empty(output_shape, dtype=dispatch_dtype, device="cuda")
    expert_output = (
        None
        if dispatch_quant is None and output_layout == ep.DispatchLayout.EXPERT_MAJOR
        else torch.zeros(output_shape, dtype=torch.bfloat16, device="cuda")
    )
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def dispatch_fn():
        # MoECommunicator.dispatch runs the full (send+recv) LL dispatch inline on
        # the stream and returns (DispatchOutput, DispatchHandle) -- the analog of
        # ncclEpDispatch + ncclEpComplete.
        return moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)

    def combine_fn(dout):
        dispatch_out, handle = dout
        moe_comm.combine(dispatch_out.tokens if expert_output is None else expert_output, handle, out=out)

    stream = torch.cuda.current_stream()

    # Warm up with the same pacing as the timed loop. Same-stream ordering keeps
    # dispatch output alive until combine consumes it without host synchronization.
    for _ in range(warmup):
        dout = dispatch_fn()
        combine_fn(dout)

    # Drain warmup work and align ranks once before recording timed events.
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # CUPTI/nsys region: capture only the post-warmup timed kernels. An external
    # nsys run with --capture-range=cudaProfilerApi records exactly the
    # dispatch/combine kernels between these two calls.
    _cupti = bool(getattr(args, "cupti_region", False))
    if _cupti:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        torch.cuda.cudart().cudaProfilerStart()

    # In-process CUPTI collector (ep_bench KernelTimer analog). start() after
    # warmup, stop() after the timed loop -- same window as the CUDA events.
    _inproc = None
    inproc_requested = bool(getattr(args, "cupti_inproc", False))
    local_inproc_ready = False
    if inproc_requested:
        try:
            _inproc = _InProcCupti()
            local_inproc_ready = True
        except Exception as exc:
            if rank == 0:
                print(f"[warn] in-proc CUPTI unavailable ({exc}); host-observed only", flush=True)
            _inproc = None
        ready = torch.tensor(int(local_inproc_ready), dtype=torch.int32, device="cuda")
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
        if ready.item() == 0:
            if rank == 0:
                print("[warn] in-proc CUPTI unavailable on at least one rank; disabling globally", flush=True)
            _inproc = None
        else:
            torch.cuda.synchronize()
            dist.barrier(group=group)
            try:
                _rc = _inproc.start()
            except Exception:
                _rc = -1
            started = torch.tensor(int(_rc == 0), dtype=torch.int32, device="cuda")
            dist.all_reduce(started, op=dist.ReduceOp.MIN, group=group)
            if started.item() == 0:
                if _rc == 0:
                    _inproc.stop()
                if rank == 0:
                    print("[warn] in-proc CUPTI failed to start on at least one rank; disabling globally", flush=True)
                _inproc = None
            dist.barrier(group=group)

    d_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        d_start[i].record(stream)
        dout = dispatch_fn()
        d_end[i].record(stream)
        c_start[i].record(stream)
        combine_fn(dout)
        c_end[i].record(stream)

    torch.cuda.synchronize()
    if _cupti:
        torch.cuda.cudart().cudaProfilerStop()
    ck_disp_us = ck_comb_us = 0.0
    inproc_ok = False
    if _inproc is not None:
        _inproc.stop()
        dist.barrier(group=group)
        ck_disp_us = _inproc.avg_us("dispatch")
        ck_comb_us = _inproc.avg_us("combine")
        n_disp = _inproc.count("dispatch")
        n_comb = _inproc.count("combine")
        inproc_ok = ck_disp_us > 0 and ck_comb_us > 0
        if os.environ.get("MSCCLPP_EP_KDEBUG", "0") == "1" and rank == 0:
            print(
                f"[kdebug inproc] dispatch: {ck_disp_us:.1f}us x{n_disp}  " f"combine: {ck_comb_us:.1f}us x{n_comb}",
                flush=True,
            )

    # ---- Collect per-iter times (ms->us) and trim the first (warmup outlier). --
    disp_us = [d_start[i].elapsed_time(d_end[i]) * 1e3 for i in range(iters)]
    comb_us = [c_start[i].elapsed_time(c_end[i]) * 1e3 for i in range(iters)]
    tot_us = [d_start[i].elapsed_time(c_end[i]) * 1e3 for i in range(iters)]
    if iters > 1:
        disp_us, comb_us, tot_us = disp_us[1:], comb_us[1:], tot_us[1:]

    def stats(times):
        return sum(times) / len(times), min(times), max(times)

    d_avg, d_min, d_max = stats(disp_us)
    c_avg, c_min, c_max = stats(comb_us)
    t_avg, t_min, t_max = stats(tot_us)

    # per-rank throughput (GB/s) uses this rank's own byte count / its avg time.
    d_tp = (disp_bytes / 1e9) / (d_avg * 1e-6)
    c_tp = (comb_bytes / 1e9) / (c_avg * 1e-6)
    t_tp = ((disp_bytes + comb_bytes) / 1e9) / (t_avg * 1e-6)

    # ---- Cross-rank reduction (mirror printLowLatencyResults). ----
    g_d_avg = _reduce_scalar(d_avg, dist.ReduceOp.SUM, group) / num_ranks
    g_d_min = _reduce_scalar(d_min, dist.ReduceOp.MIN, group)
    g_d_max = _reduce_scalar(d_max, dist.ReduceOp.MAX, group)
    g_c_avg = _reduce_scalar(c_avg, dist.ReduceOp.SUM, group) / num_ranks
    g_c_min = _reduce_scalar(c_min, dist.ReduceOp.MIN, group)
    g_c_max = _reduce_scalar(c_max, dist.ReduceOp.MAX, group)
    g_t_avg = _reduce_scalar(t_avg, dist.ReduceOp.SUM, group) / num_ranks
    g_t_min = _reduce_scalar(t_min, dist.ReduceOp.MIN, group)
    g_t_max = _reduce_scalar(t_max, dist.ReduceOp.MAX, group)

    # ---- Kernel-only pass (torch.profiler / Kineto-CUPTI) — ep_bench parity. ----
    # Measures device-side kernel time (strips host launch latency). Dispatch and
    # combine are profiled in isolation so no kernel-name matching is required.
    kernel_ok = False
    g_dk_avg = g_dk_min = g_dk_max = 0.0
    g_ck_avg = g_ck_min = g_ck_max = 0.0
    if args.kernel_timing and not _cupti and not bool(getattr(args, "cupti_inproc", False)):
        try:
            dk_us, ck_us = _profile_paired_kernels(dispatch_fn, combine_fn, iters, stream, group, rank)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            g_dk_avg = _reduce_scalar(dk_us, dist.ReduceOp.SUM, group) / num_ranks
            g_dk_min = _reduce_scalar(dk_us, dist.ReduceOp.MIN, group)
            g_dk_max = _reduce_scalar(dk_us, dist.ReduceOp.MAX, group)
            g_ck_avg = _reduce_scalar(ck_us, dist.ReduceOp.SUM, group) / num_ranks
            g_ck_min = _reduce_scalar(ck_us, dist.ReduceOp.MIN, group)
            g_ck_max = _reduce_scalar(ck_us, dist.ReduceOp.MAX, group)
            kernel_ok = g_dk_avg > 0 and g_ck_avg > 0
        except Exception as exc:  # profiler unavailable / hiccup: keep host numbers valid
            if rank == 0:
                print(f"[warn] kernel-only pass failed ({exc}); reporting host-observed only", flush=True)

    # ---- In-process CUPTI reduction (ep_bench KernelTimer analog). ----
    g_ik_d_avg = g_ik_d_min = g_ik_d_max = 0.0
    g_ik_c_avg = g_ik_c_min = g_ik_c_max = 0.0
    g_inproc_ok = 0
    if bool(getattr(args, "cupti_inproc", False)):
        g_ik_d_avg = _reduce_scalar(ck_disp_us, dist.ReduceOp.SUM, group) / num_ranks
        g_ik_d_min = _reduce_scalar(ck_disp_us if inproc_ok else 1e18, dist.ReduceOp.MIN, group)
        g_ik_d_max = _reduce_scalar(ck_disp_us, dist.ReduceOp.MAX, group)
        g_ik_c_avg = _reduce_scalar(ck_comb_us, dist.ReduceOp.SUM, group) / num_ranks
        g_ik_c_min = _reduce_scalar(ck_comb_us if inproc_ok else 1e18, dist.ReduceOp.MIN, group)
        g_ik_c_max = _reduce_scalar(ck_comb_us, dist.ReduceOp.MAX, group)
        g_inproc_ok = int(_reduce_scalar(1.0 if inproc_ok else 0.0, dist.ReduceOp.MIN, group))

    d_tp_all = _gather_scalars(d_tp, num_ranks, group)
    c_tp_all = _gather_scalars(c_tp, num_ranks, group)
    t_tp_all = _gather_scalars(t_tp, num_ranks, group)

    if rank == 0:
        # avg throughput uses rank-0 byte count / global avg time (as ep_bench does).
        avg_d_tp = (disp_bytes / 1e9) / (g_d_avg * 1e-6)
        avg_c_tp = (comb_bytes / 1e9) / (g_c_avg * 1e-6)
        avg_t_tp = ((disp_bytes + comb_bytes) / 1e9) / (g_t_avg * 1e-6)

        def minmax_rank(vals):
            lo = min(range(num_ranks), key=lambda r: vals[r])
            hi = max(range(num_ranks), key=lambda r: vals[r])
            return vals[lo], lo, vals[hi], hi

        d_lo, d_lo_r, d_hi, d_hi_r = minmax_rank(d_tp_all)
        c_lo, c_lo_r, c_hi, c_hi_r = minmax_rank(c_tp_all)
        t_lo, t_lo_r, t_hi, t_hi_r = minmax_rank(t_tp_all)

        print(f"\n=== Summary (Low Latency, across {num_ranks} ranks) ===")
        print("\n--- Host-observed performance ---")
        print("Pacing: batched steady state")
        print(f"Dispatch ({dispatch_label}):  avg={g_d_avg:.2f} us, min={g_d_min:.2f} us, max={g_d_max:.2f} us")
        print(
            f"                  throughput: avg={avg_d_tp:.2f} GB/s, "
            f"min={d_lo:.2f} GB/s (rank {d_lo_r}), max={d_hi:.2f} GB/s (rank {d_hi_r})"
        )
        print(f"Combine (BF16):   avg={g_c_avg:.2f} us, min={g_c_min:.2f} us, max={g_c_max:.2f} us")
        print(
            f"                  throughput: avg={avg_c_tp:.2f} GB/s, "
            f"min={c_lo:.2f} GB/s (rank {c_lo_r}), max={c_hi:.2f} GB/s (rank {c_hi_r})"
        )
        print(f"Total (D+C):      avg={g_t_avg:.2f} us, min={g_t_min:.2f} us, max={g_t_max:.2f} us")
        print(
            f"                  throughput: avg={avg_t_tp:.2f} GB/s, "
            f"min={t_lo:.2f} GB/s (rank {t_lo_r}), max={t_hi:.2f} GB/s (rank {t_hi_r})"
        )

        print("\n--- Kernel-only performance (device kernel time via torch.profiler/CUPTI) ---")
        if kernel_ok:
            # The LL dispatch kernel ends with a cross-rank receive spin-wait, so
            # its device time includes wait skew. torch.profiler's host tracing
            # overhead makes one rank lag, inflating that rank's dispatch device
            # time into the ms range; the cross-rank MIN (the rank that did not
            # wait) is the representative kernel floor and matches ep_bench's
            # low-perturbation CUPTI number. Combine has little recv-spin and is
            # stable across ranks. throughput uses the representative (min) time.
            print(
                f"Dispatch:    min={g_dk_min:.2f} us (representative)  "
                f"[avg={g_dk_avg:.2f}, max={g_dk_max:.2f} us -- inflated by profiler recv-spin skew]"
            )
            print(f"                  throughput @min: {(disp_bytes / 1e9) / (g_dk_min * 1e-6):.2f} GB/s")
            print(
                f"Combine:     min={g_ck_min:.2f} us (representative)  "
                f"[avg={g_ck_avg:.2f}, max={g_ck_max:.2f} us -- inflated by profiler rank skew]"
            )
            print(f"                  throughput @min: {(comb_bytes / 1e9) / (g_ck_min * 1e-6):.2f} GB/s")
            print(
                "  NOTE: for an authoritative low-perturbation kernel-only number, run under "
                "nsys (as ep_bench's CUPTI path does); torch.profiler perturbs the LL recv-spin."
            )
        else:
            print("  NOTE: kernel-only pass disabled or unavailable.")

        if bool(getattr(args, "cupti_inproc", False)):
            print("\n--- Kernel-only performance (in-process CUPTI Activity API, ep_bench KernelTimer analog) ---")
            if g_inproc_ok:
                # The LL dispatch kernel ends with a cross-rank receive spin-wait,
                # so a lagging rank's device time includes wait skew (same effect
                # as nsys's max outlier). The cross-rank MIN (the rank that did not
                # wait) is the representative kernel floor; it matches the nsys
                # CUPTI number and ep_bench's low-perturbation figure. Combine has
                # little recv-spin and is stable across ranks.
                print(
                    f"Dispatch:    min={g_ik_d_min:.2f} us (representative)  "
                    f"[avg={g_ik_d_avg:.2f}, max={g_ik_d_max:.2f} us -- recv-spin skew on lagging ranks]"
                )
                print(f"                  throughput @min: {(disp_bytes / 1e9) / (g_ik_d_min * 1e-6):.2f} GB/s")
                print(
                    f"Combine:     min={g_ik_c_min:.2f} us (representative)  "
                    f"[avg={g_ik_c_avg:.2f}, max={g_ik_c_max:.2f} us -- rank skew on lagging ranks]"
                )
                print(f"                  throughput @min: {(comb_bytes / 1e9) / (g_ik_c_min * 1e-6):.2f} GB/s")
            else:
                print("  NOTE: in-process CUPTI collector unavailable (see [warn] above).")

        print(
            f"\nByte counts: dispatch={disp_bytes / 1e6:.2f} MB ({dispatch_label}), "
            f"combine={comb_bytes / 1e6:.2f} MB (BF16), rows={num_dispatch_rows}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
