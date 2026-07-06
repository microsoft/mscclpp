#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified low-latency EP benchmark for MSCCL++ EP — an apples-to-apples port of
NCCL-EP's ``contrib/nccl_ep/ep_bench.cu`` low-latency (LL) flow, with the NCCL-EP
API (``ncclEpDispatch`` / ``ncclEpCombine``) replaced by MSCCL++ EP's
``Buffer.low_latency_dispatch`` / ``Buffer.low_latency_combine``.

Why this exists
---------------
``ep_bench`` is the reference NCCL-EP micro-benchmark. To compare MSCCL++ EP
against it fairly we must measure *the same thing the same way*. This script is a
line-for-line reimplementation of ``ep_bench``'s LL measurement methodology, only
swapping the collective API underneath:

* **Paired** dispatch→sync→combine→sync→barrier per iteration (``runPairedBenchmark``).
* **Per-iteration CUDA events** recorded on the stream *around each kernel launch*;
  the ``cudaStreamSynchronize`` and ``MPI_Barrier`` (here ``dist.barrier``) happen
  **outside** the timed region, exactly as in ``ep_bench``.
* **Skip the first timed iteration** (warmup outlier) — matches ``ep_bench``'s
  ``calc_stats`` which trims ``times[0]`` when ``num_iters > 1``.
* **Byte accounting** identical to ``calculateLowLatencyBytes``:
  ``bytes = num_valid_selections * hidden * 2`` (BF16) for *both* dispatch and
  combine, where ``num_valid_selections = count(topk_idx >= 0)``.
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
``ep_bench`` is C++/MPI; MSCCL++ EP's LL API is Python/torch, so this harness is
Python. The *measurement* is identical: both bracket the same dispatch/combine
kernels with CUDA events and report GPU-side host-observed time. The only
difference is host-side launch latency, which sits *outside* the recorded events
for the async kernels and is the same definitional gap ``ep_bench`` has (larger
in Python, but not counted in the kernel elapsed time). For a pure kernel number,
run under ``nsys``/CUPTI as with ``ep_bench``'s ``--- Kernel-only ---`` section.

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
        description="MSCCL++ EP low-latency benchmark (ep_bench parity)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Env fallbacks keep the existing MSCCLPP_EP_BENCH_* launchers working.
    p.add_argument("-a", "--algorithm", default="ll", choices=["ll", "low-latency"],
                   help="algorithm mode (only LL is implemented here)")
    p.add_argument("-t", "--num-tokens", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_TOKENS", "128")),
                   help="tokens per rank (ep_bench LL max_tokens_per_rank)")
    p.add_argument("-d", "--hidden", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_HIDDEN", "7168")),
                   help="hidden dimension")
    p.add_argument("-k", "--num-topk", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_TOPK", "8")),
                   help="top-k experts per token")
    p.add_argument("-e", "--num-experts", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_EXPERTS", "256")),
                   help="global number of experts")
    p.add_argument("-w", "--num-warmup", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_WARMUP", "10")),
                   help="warmup iterations")
    p.add_argument("-i", "--num-iters", type=int,
                   default=int(os.environ.get("MSCCLPP_EP_BENCH_ITERS", "50")),
                   help="timed iterations")
    p.add_argument("--seed", type=int, default=0xB3C4, help="per-rank RNG seed base")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp.ext import ep

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    warmup = args.num_warmup
    iters = args.num_iters
    assert num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    num_local_experts = num_experts // num_ranks

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
    num_valid_selections = int((topk_idx >= 0).sum().item())
    disp_bytes = num_valid_selections * hidden * 2  # BF16
    comb_bytes = num_valid_selections * hidden * 2  # BF16 (symmetric, per ep_bench)

    num_rdma_bytes = ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if rank == 0:
        print(
            f"[cfg] algorithm=LOW_LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} top_k={num_topk} warmup={warmup} iters={iters} "
            f"num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    buf = ep.Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=max(1, num_local_experts),
    )
    assert buf.is_available()

    # ---- Hoist dispatch/combine output tensors out of the timed loop (ep_bench
    # preallocates all EP tensors before benchmarking; matching that keeps the
    # timed region kernel-bound rather than allocator-bound).
    recv_x = torch.empty((num_local_experts, num_ranks * num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    recv_src_info = torch.empty((num_local_experts, num_ranks * num_tokens), dtype=torch.int32, device="cuda")
    recv_layout_range = torch.empty((num_local_experts, num_ranks), dtype=torch.int64, device="cuda")
    recv_count = torch.empty((num_local_experts,), dtype=torch.int32, device="cuda")
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def dispatch_fn():
        # return_recv_hook=False => full (send+recv) dispatch runs inline on the
        # stream, so the CUDA-event bracket captures the whole op (the analog of
        # ncclEpDispatch + ncclEpComplete).
        return buf.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            False, False, False,  # use_fp8, async, return_recv_hook
            recv_x, None, recv_src_info, recv_layout_range, recv_count,
        )

    def combine_fn(dout):
        buf.low_latency_combine(
            dout[0], topk_idx, topk_weights, dout[3], dout[4],
            num_tokens, num_experts,
            False, False, False,  # zero_copy, async, return_recv_hook
            out,
        )

    stream = torch.cuda.current_stream()

    # ---- runPairedBenchmark: warmup (paired), then per-iter timed (paired). ----
    for _ in range(warmup):
        dout = dispatch_fn()
        stream.synchronize()
        combine_fn(dout)
        stream.synchronize()
        dist.barrier(group=group)

    d_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        d_start[i].record(stream)
        dout = dispatch_fn()
        d_end[i].record(stream)          # record before sync
        stream.synchronize()             # sync outside timing
        c_start[i].record(stream)        # record after sync, before combine
        combine_fn(dout)
        c_end[i].record(stream)          # record before sync
        stream.synchronize()             # sync outside timing
        dist.barrier(group=group)        # keep ranks in lockstep, outside timing

    torch.cuda.synchronize()

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
        print(f"Dispatch (BF16):  avg={g_d_avg:.2f} us, min={g_d_min:.2f} us, max={g_d_max:.2f} us")
        print(f"                  throughput: avg={avg_d_tp:.2f} GB/s, "
              f"min={d_lo:.2f} GB/s (rank {d_lo_r}), max={d_hi:.2f} GB/s (rank {d_hi_r})")
        print(f"Combine (BF16):   avg={g_c_avg:.2f} us, min={g_c_min:.2f} us, max={g_c_max:.2f} us")
        print(f"                  throughput: avg={avg_c_tp:.2f} GB/s, "
              f"min={c_lo:.2f} GB/s (rank {c_lo_r}), max={c_hi:.2f} GB/s (rank {c_hi_r})")
        print(f"Total (D+C):      avg={g_t_avg:.2f} us, min={g_t_min:.2f} us, max={g_t_max:.2f} us")
        print(f"                  throughput: avg={avg_t_tp:.2f} GB/s, "
              f"min={t_lo:.2f} GB/s (rank {t_lo_r}), max={t_hi:.2f} GB/s (rank {t_hi_r})")
        print(f"\nByte counts: dispatch={disp_bytes / 1e6:.2f} MB (BF16), "
              f"combine={comb_bytes / 1e6:.2f} MB (BF16), selections={num_valid_selections}")


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
