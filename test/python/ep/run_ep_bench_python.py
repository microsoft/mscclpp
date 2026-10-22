#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified in-process latency EP benchmark that drives both libraries'
Python APIs directly, inside the *same* measurement flow:

* **mscclpp EP**   — ``mscclpp.ep.MoECommunicator.dispatch`` / ``.combine``.
* **NVIDIA NCCL-EP** — ``nccl.ep.Group`` / ``nccl.ep.Handle.dispatch`` / ``.combine``
  (the ``nccl4py`` Pythonic bindings for ``libnccl_ep.so``).

This script calls each backend's Python API in one process through a shared
paired-benchmark loop, so both are timed with the same methodology:

* **Paired** ``dispatch -> combine`` per iteration, with no per-iteration
  ``stream.synchronize()`` or cross-rank barrier inside the timed loop -- the
  dispatch and combine kernels pipeline back-to-back on the stream.
* **Per-iteration CUDA events** recorded on the stream around each launch; the final
  ``torch.cuda.synchronize()`` and any barriers are *outside* the timed loop.
* **Skip the first timed iteration** (warmup outlier), matching ``ep_bench``.
* **Byte accounting** identical to ``calculateLowLatencyBytes``:
  ``bytes = num_valid_selections * hidden * 2`` (BF16) for both dispatch and combine.
* **Cross-rank reduction** identical to ``printLowLatencyResults``.
* **Output** mirrors the ``=== Summary (Low Latency, across N ranks) ===`` block so a
  run can be diffed directly against NCCL-EP ``ep_bench``.

Bootstrap: MPI (mpi4py + mpirun)
--------------------------------
Both backends share an MPI ``COMM_WORLD`` bootstrap (the same mechanism the C++
NCCL-EP ``ep_test.py`` uses, *not* torch.distributed:

* mscclpp wraps the MPI communicator with ``CommGroup(mpi_comm=MPI.COMM_WORLD)``.
* NCCL-EP builds a ``nccl.core.Communicator`` from a unique id broadcast over MPI.
* Cross-rank reductions/barriers use ``mpi4py`` (``comm.allreduce`` / ``comm.Barrier``).

torch is still used for CUDA tensors and event timing; only its distributed NCCL
backend is avoided.

Backend ordering (``--backend all``)
------------------------------------
NCCL-EP is benchmarked **before** mscclpp. Initializing mscclpp's LL
``MoECommunicator`` in a process perturbs CUDA state such that a subsequent NCCL-EP
cooperative-launch dispatch fails with ``cudaErrorInvalidValue``; the reverse order
is clean. Both backends still use independent communicators, and each runs its own
warmup, so the ordering does not affect the reported numbers.

Launch environment
------------------
NCCL-EP JIT-compiles its LL kernels with ``nvcc`` at first use and dynamically links
``libnccl_ep.so`` / ``libnccl.so``, which must match NCCL major/minor. mscclpp's LL
runtime supports both CUDA-IPC (NVLink) and RDMA/IB transports. When every peer is
reachable over CUDA IPC -- i.e. on the same node or within the same NVLink/MNNVL
domain -- the LL path runs entirely over CUDA IPC and derives its NVLink/IPC domain
from the bootstrap (ranks-per-node / ranks-per-IPC-domain); in that case NO HCA list
(``MSCCLPP_HCA_DEVICES``) or fabric-IPC env is required for the mscclpp backend.
Cross-domain peers use the RDMA/IB path, which does need the active HCA list. A
working single-node 4-GPU (CUDA-IPC) launch::

    NCCL_BUILD=/opt/microsoft/mrc/ep/nccl/build
    mpirun -np 4 --bind-to none \
        -x PATH -x CUDA_HOME=/usr/local/cuda \
        -x LD_LIBRARY_PATH=$NCCL_BUILD/lib:$LD_LIBRARY_PATH \
        -x LD_PRELOAD=$NCCL_BUILD/lib/libnccl.so.2.30.7 \
        -x NCCL_EP_JIT_SOURCE_DIR=/opt/microsoft/mrc/ep/nccl/contrib/nccl_ep \
        -x NCCL_EP_JIT_BUILD_INCLUDE_DIR=$NCCL_BUILD/include \
        -x NCCL_IB_DISABLE=1 -x NCCL_MNNVL_ENABLE=0 -x NCCL_NET_PLUGIN=none \
        python run_ep_bench_python.py --backend all -e 128

Multi-node (same NVLink/MNNVL fabric): launch with HPCX Open MPI 4 (rebuild mpi4py
against it) and ``-x LD_PRELOAD=<hpcx>/ompi/lib/libmpi.so.40:<in-tree libnccl.so>``,
set ``-x NCCL_MNNVL_ENABLE=1`` so NCCL-EP uses the cross-node NVLink clique, and
pass a hostfile with ``--map-by ppr:<gpus>:node``. Within a shared NVLink/MNNVL
domain the mscclpp backend needs no extra transport env (CUDA-IPC path); for peers
outside a shared IPC domain, provide ``MSCCLPP_HCA_DEVICES`` for the RDMA/IB path.

A working 2-node, 8-GPU launch (both nodes on one NVLink/MNNVL fabric)::

    NCCL_BUILD=/opt/microsoft/mrc/ep/nccl/build
    HPCXLIB=/opt/hpcx-.../ompi/lib          # HPCX Open MPI 4 (mpi4py built against it)
    PRELOAD_NCCL=$(ls -1 $NCCL_BUILD/lib/libnccl.so.*.* | sort -V | tail -1)
    printf '<ip_address1> slots=4\n<ip_address2> slots=4\n' > /tmp/hostfile

    mpirun -np 8 --hostfile /tmp/hostfile --map-by ppr:4:node --bind-to none \
        -mca plm_rsh_args '-o StrictHostKeyChecking=no' \
        -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include <subnet/prefix> \
        -mca oob_tcp_if_include <subnet/prefix> -mca coll_hcoll_enable 0 \
        -mca coll_ucc_enable 0 -mca mtl ^ofi -mca osc ^ucx \
        -x PATH -x CUDA_HOME=/usr/local/cuda \
        -x LD_LIBRARY_PATH=$HPCXLIB:$NCCL_BUILD/lib:$LD_LIBRARY_PATH \
        -x LD_PRELOAD="$HPCXLIB/libmpi.so.40 $PRELOAD_NCCL" \
        -x NCCL_EP_JIT_SOURCE_DIR=/opt/microsoft/mrc/ep/nccl/contrib/nccl_ep \
        -x NCCL_EP_JIT_BUILD_INCLUDE_DIR=$NCCL_BUILD/include \
        -x UCX_TLS=tcp,self,cuda_copy -x UCX_NET_DEVICES=<iface> \
        -x NCCL_SOCKET_IFNAME=<iface> -x MSCCLPP_SOCKET_IFNAME=<iface> \
        -x NCCL_IB_DISABLE=1 -x NCCL_MNNVL_ENABLE=1 -x NCCL_NET_PLUGIN=none \
        python test/python/ep/run_ep_bench_python.py \
            --backend all -e 128 -t 128 -d 7168 -k 8 -w 10 -i 50

Here ``LD_PRELOAD`` forces HPCX Open MPI 4 (``libmpi.so.40``, matching the mpi4py
rebuild) ahead of any conda Open MPI 5, and the in-tree ``libnccl`` ahead of an
older environment one. ``NCCL_MNNVL_ENABLE=1`` lets NCCL-EP use the cross-node
NVLink clique; the mscclpp backend runs the same NVLink/MNNVL fabric over CUDA IPC.

``LD_PRELOAD`` of the in-tree ``libnccl.so`` is required whenever the environment's
default ``libnccl`` is older than the one ``libnccl_ep.so`` was built against.
See ``src/ext/ep/README.md`` ("Unified in-process benchmark") for a ready-to-run
launch command that sets all of this.
"""

from __future__ import annotations

import argparse
import os

import torch

from ep_bench_common import (
    MPI,
    init_mpi,
    make_inputs,
    _init_torch_nccl,
    _mpi_stats,
    sum_matching_kernel_us,
)
from ep_bench_mscclpp import setup_mscclpp, parse_kineto_kernels as mscclpp_parse_kineto
from ep_bench_nccl import setup_nccl, parse_kineto_kernels as nccl_parse_kineto
from ep_bench_deepep import setup_deepep, parse_kineto_kernels as deepep_parse_kineto
from ep_bench_flashinfer import setup_flashinfer, parse_kineto_kernels as flashinfer_parse_kineto


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
        choices=["mscclpp", "nccl", "deepep", "flashinfer", "all"],
        default="all",
        help="which backend(s) to benchmark in this run (all runs nccl, mscclpp, deepep, flashinfer)",
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
        choices=["token_major", "rank_major", "expert_major", "rank_major_topk_expanded"],
        default=None,
        help="received-token dispatch layout. When omitted, each backend uses its own default "
        "layout (NCCL-EP=expert_major, MSCCL++ latency=expert_major, MSCCL++ throughput=token_major, "
        "DeepEP=rank_major, FlashInfer=rank_major). "
        "The explicit value is applied where supported: NCCL-EP and DeepEP accept rank_major/expert_major; "
        "MSCCL++ latency accepts expert_major/rank_major/rank_major_topk_expanded and throughput accepts "
        "rank_major/token_major. "
        "FlashInfer is rank-major only; unsupported requests are "
        "noted and the backend's default layout is kept.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="mscclpp: run a one-time combine correctness check before timing.",
    )
    p.add_argument(
        "--mscclpp-cross-rack-topk-half",
        action="store_true",
        help=(
            "mscclpp-only diagnostic routing for 8-rank/4-GPU-per-node cross-rack runs: "
            "for top-k=8, route four selections per token to the other node and mark four "
            "invalid, eliminating same-node peer and local-self dispatch traffic."
        ),
    )
    p.add_argument(
        "--mscclpp-same-rack-topk",
        action="store_true",
        help=(
            "mscclpp-only diagnostic routing for 8-rank/4-GPU-per-node runs: route every "
            "top-k selection to experts on the local node/rack, eliminating cross-rack dispatch traffic."
        ),
    )
    p.add_argument(
        "--mscclpp-cross-rack-token-half-all-remote",
        action="store_true",
        help=(
            "mscclpp-only diagnostic routing for 8-rank/4-GPU-per-node cross-rack runs: "
            "for -t 128 -k 8, route the first 64 tokens to all four ranks on the other rack "
            "and mark the remaining 64 tokens invalid."
        ),
    )
    p.add_argument(
        "--mscclpp-rack0-gpu0-one-to-rack1",
        action="store_true",
        help=(
            "mscclpp-only diagnostic routing for 8-rank/4-GPU-per-node cross-rack runs: "
            "only rank 0 (rack0/GPU0) sends tokens, keeping random top-k experts across all ranks; "
            "all top-k entries on every other rank are invalid."
        ),
    )
    p.add_argument(
        "--mscclpp-rack0-gpu0-rack1-only",
        action="store_true",
        help=(
            "like --mscclpp-rack0-gpu0-one-to-rack1, but rank 0 also invalidates its top-k entries "
            "whose expert is on rack0 (ranks 0..3, incl. itself), so it sends only cross-rack "
            "traffic to ranks 4..7."
        ),
    )
    p.add_argument(
        "--mscclpp-rank0-all-to-rack1",
        action="store_true",
        help=(
            "mscclpp-only diagnostic routing for 8-rank/4-GPU-per-node cross-rack runs: only rank 0 sends, "
            "and every token goes to all four rack1 ranks (4..7), so each destination gets the same rows "
            "(1 GPU -> 4 GPUs). --mscclpp-rank0-active-tokens limits how many tokens are routed."
        ),
    )
    p.add_argument(
        "--mscclpp-rank0-active-tokens",
        type=int,
        default=None,
        help="with --mscclpp-rank0-all-to-rack1: route only the first N tokens (0 = empty dispatch).",
    )
    p.add_argument(
        "--mscclpp-rank0-dst-ranks",
        default=None,
        help=(
            "with --mscclpp-rack0-gpu0-rack1-only: comma-separated rack1 destination ranks rank 0 "
            "may send to (default 4,5,6,7); top-k entries to other ranks become invalid."
        ),
    )
    p.add_argument(
        "--print-topk-rank-slots",
        action="store_true",
        help=(
            "print top-k routing balance before benchmarking: raw valid top-k lane counts and "
            "deduplicated token->destination-rank slot counts. Can also be enabled with "
            "EP_PRINT_TOPK_RANK_SLOTS=1."
        ),
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
    if args.dispatch_dtype == "fp8_e4m3" and args.backend in ("nccl", "all"):
        raise SystemExit(
            "--dispatch-dtype fp8_e4m3 is only wired for the mscclpp backend in this benchmark "
            "(NCCL-EP supports FP8 but its path here is BF16-only); use --backend mscclpp"
        )
    if args.dispatch_dtype == "fp8_e4m3" and args.mode != "latency":
        raise SystemExit("MSCCL++ throughput mode currently supports BF16 dispatch only")
    route_flags = [
        args.mscclpp_cross_rack_topk_half,
        args.mscclpp_same_rack_topk,
        args.mscclpp_cross_rack_token_half_all_remote,
        args.mscclpp_rack0_gpu0_one_to_rack1,
        args.mscclpp_rack0_gpu0_rack1_only,
        args.mscclpp_rank0_all_to_rack1,
    ]
    if sum(bool(flag) for flag in route_flags) > 1:
        raise SystemExit("choose only one mscclpp route diagnostic option")
    if any(route_flags) and args.backend != "mscclpp":
        raise SystemExit("mscclpp route diagnostics require --backend mscclpp")
    if args.mscclpp_cross_rack_topk_half and args.num_topk != 8:
        raise SystemExit("--mscclpp-cross-rack-topk-half requires --num-topk 8")
    if args.mscclpp_cross_rack_token_half_all_remote and (args.num_tokens != 128 or args.num_topk != 8):
        raise SystemExit("--mscclpp-cross-rack-token-half-all-remote requires -t 128 -k 8")
    if args.mscclpp_rack0_gpu0_one_to_rack1 and args.num_topk < 4:
        raise SystemExit("--mscclpp-rack0-gpu0-one-to-rack1 requires --num-topk >= 4")
    if args.mscclpp_rank0_dst_ranks is not None:
        if not args.mscclpp_rack0_gpu0_rack1_only:
            raise SystemExit("--mscclpp-rank0-dst-ranks requires --mscclpp-rack0-gpu0-rack1-only")
        os.environ["MSCCLPP_EP_ROUTE_RANK0_DST_RANKS"] = args.mscclpp_rank0_dst_ranks
    if args.mscclpp_rank0_active_tokens is not None:
        if not args.mscclpp_rank0_all_to_rack1:
            raise SystemExit("--mscclpp-rank0-active-tokens requires --mscclpp-rank0-all-to-rack1")
        os.environ["MSCCLPP_EP_ROUTE_RANK0_ACTIVE_TOKENS"] = str(args.mscclpp_rank0_active_tokens)
    if args.mscclpp_rank0_all_to_rack1 and args.num_topk < 4:
        raise SystemExit("--mscclpp-rank0-all-to-rack1 requires --num-topk >= 4")
    if args.mscclpp_rack0_gpu0_rack1_only and args.num_topk < 4:
        raise SystemExit("--mscclpp-rack0-gpu0-rack1-only requires --num-topk >= 4")
    return args


def _topk_remote_ranks(src_rank: int, num_ranks: int, local_world_size: int) -> list[int]:
    if local_world_size > 0 and num_ranks % local_world_size == 0:
        src_node = src_rank // local_world_size
        return [rank for rank in range(num_ranks) if rank // local_world_size != src_node]
    return [rank for rank in range(num_ranks) if rank != src_rank]


def _print_topk_rank_slot_diagnostics(args, comm, rank, num_ranks, topk_idx, local_world_size: int) -> None:
    if not (args.print_topk_rank_slots or os.environ.get("EP_PRINT_TOPK_RANK_SLOTS", "0") == "1"):
        return
    if args.num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    experts_per_rank = args.num_experts // num_ranks
    topk_cpu = topk_idx.detach().cpu()
    valid = topk_cpu >= 0
    dst_rank = torch.full_like(topk_cpu, -1)
    dst_rank[valid] = torch.div(topk_cpu[valid], experts_per_rank, rounding_mode="floor")

    slot_counts = []
    lane_counts = []
    for dst in range(num_ranks):
        hits = dst_rank == dst
        slot_counts.append(int(hits.any(dim=1).sum().item()))
        lane_counts.append(int(hits.sum().item()))

    remote_ranks = _topk_remote_ranks(rank, num_ranks, local_world_size)
    remote_slots = sum(slot_counts[dst] for dst in remote_ranks)
    num_tokens = int(topk_cpu.size(0))
    summary = {
        "rank": rank,
        "num_tokens": num_tokens,
        "num_topk": int(topk_cpu.size(1)),
        "valid_lanes": int(valid.sum().item()),
        "remote_ranks": remote_ranks,
        "remote_slots": remote_slots,
        "remote_slots_per_token": float(remote_slots) / float(num_tokens) if num_tokens else 0.0,
        "slot_counts": slot_counts,
        "lane_counts": lane_counts,
    }

    if os.environ.get("EP_PRINT_TOPK_VALUES", "0") == "1":
        rows = int(os.environ.get("EP_PRINT_TOPK_ROWS", str(num_tokens)) or "0")
        if rows < 0:
            rows = num_tokens
        summary["topk_idx_rows"] = topk_cpu[:rows].tolist()
        summary["topk_dst_rank_rows"] = dst_rank[:rows].tolist()

    summaries = comm.gather(summary, root=0)
    if rank != 0:
        return

    summaries = sorted(summaries, key=lambda item: item["rank"])
    if local_world_size > 0 and num_ranks >= 2 * local_world_size:
        focus_ranks = list(range(local_world_size, 2 * local_world_size))
    else:
        focus_ranks = list(range(num_ranks))

    focus_label = ",".join(str(dst) for dst in focus_ranks)
    print(
        "\n--- Top-k routing / rank-major slot diagnostic ---\n"
        f"experts_per_rank={experts_per_rank}, focus_ranks={focus_label}\n"
        "slots = deduplicated token->destination-rank payload rows; "
        "lanes = raw valid top-k entries before per-token destination-rank dedup.",
        flush=True,
    )
    print(
        "src_rank | remote_ranks | remote_slots | remote_slots/token | "
        f"focus_slots[{focus_label}] | focus_lanes[{focus_label}] | valid_lanes",
        flush=True,
    )
    for item in summaries:
        focus_slots = ",".join(str(item["slot_counts"][dst]) for dst in focus_ranks)
        focus_lanes = ",".join(str(item["lane_counts"][dst]) for dst in focus_ranks)
        remote_label = ",".join(str(dst) for dst in item["remote_ranks"])
        print(
            f"{item['rank']:>8} | {remote_label:<12} | {item['remote_slots']:>12} | "
            f"{item['remote_slots_per_token']:>18.3f} | {focus_slots:<18} | "
            f"{focus_lanes:<18} | {item['valid_lanes']}",
            flush=True,
        )

    if local_world_size > 0 and num_ranks >= 2 * local_world_size:
        print(f"\n[topk-slots] source ranks 0..{local_world_size - 1} -> destination ranks {focus_label}")
        print("src\\dst | " + " | ".join(f"{dst:>5}" for dst in focus_ranks) + " | total | avg/token", flush=True)
        for item in summaries[:local_world_size]:
            total = sum(item["slot_counts"][dst] for dst in focus_ranks)
            avg = float(total) / float(item["num_tokens"]) if item["num_tokens"] else 0.0
            counts = " | ".join(f"{item['slot_counts'][dst]:>5}" for dst in focus_ranks)
            print(f"{item['rank']:>7} | {counts} | {total:>5} | {avg:>9.3f}", flush=True)

    for item in summaries:
        if "topk_idx_rows" not in item:
            continue
        print(f"\n[topk-values][rank {item['rank']}] topk_idx rows: {item['topk_idx_rows']}", flush=True)
        print(
            f"[topk-values][rank {item['rank']}] destination-rank rows: {item['topk_dst_rank_rows']}",
            flush=True,
        )


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
    group_size=1,
    rank=0,
):
    """DeepEP bench_kineto-style kernel timing: torch.profiler (CUDA activity)
    over the paired dispatch->combine loop, with a per-iteration L2 flush and a
    cuda._sleep(~10ms) + cross-rank barrier to absorb host launch skew. Returns
    the average per-kernel GPU time (us) for the dispatch and combine kernels,
    matched by name substring in the profiler key_averages() table.

    EP_KINETO_BARRIER_COMBINE=1 inserts a SECOND GPU-side barrier between
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

    EP_KINETO_SEPARATE (default 1) measures dispatch and combine in TWO separate
    profiled passes -- each a single op per iteration with the barrier immediately
    before it -- exactly like DeepEP's own bench_kineto (which is called once per
    op). This aligns BOTH kernels at entry without ever placing a barrier between
    a paired dispatch->combine (so it is safe for DeepEP multi-node), collapsing
    the combine recv-spin skew. It is the right mode for EAGER runs. Set
    EP_KINETO_SEPARATE=0 for the paired single-pass loop, which is REQUIRED (not
    legacy) whenever a backend captures dispatch+combine in ONE CUDA graph: a
    single replay runs both phases, so the separate pass can no longer isolate
    combine and the per-phase split must come from the paired pass instead."""
    import torch.profiler as _tp

    use_mid = os.environ.get("EP_KINETO_BARRIER_COMBINE", "0") == "1" and mid_barrier is not None
    separate = os.environ.get("EP_KINETO_SEPARATE", "1") == "1"
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
            comm.Barrier()  # MPI host barrier (host-only alignment)

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
    _d = dispatch_fn()
    combine_fn(_d)
    torch.cuda.synchronize()
    schedule = _tp.schedule(wait=0, warmup=1, active=1, repeat=1)
    with _tp.profile(activities=[_tp.ProfilerActivity.CUDA], schedule=schedule, acc_events=True) as prof:
        for _ in range(2):
            for _ in range(num_tests):
                if flush_l2:
                    _flush_l2_cache()
                _do_barrier()
                dout = dispatch_fn()
                if use_mid:
                    mid_barrier()  # align combine entry across ranks (FlashInfer-style)
                combine_fn(dout)
            torch.cuda.synchronize()
            prof.step()

    ka = prof.key_averages()
    _dump_kineto_per_kernel(prof, group_size, rank)
    return parse_kernels(ka)


def _dump_kineto_per_kernel(prof, group_size, rank):
    """EP_KINETO_DUMP_DIR=<dir>: write every profiled dispatch/combine kernel
    (active step only) to <dir>/kineto_kernels_rank<r>.csv, ordered by GPU start
    time and indexed by (replay, pos) where pos is the iteration inside one
    graph replay of ``group_size`` dispatch->combine pairs."""
    out_dir = os.environ.get("EP_KINETO_DUMP_DIR")
    if not out_dir:
        return
    try:
        evs = prof.profiler.kineto_results.events()
    except Exception as exc:  # noqa: BLE001
        print(f"[kineto-dump] rank={rank} no kineto_results: {exc}", flush=True)
        return
    rows = {"dispatch": [], "combine": []}
    for e in evs:
        if str(e.device_type()) != "DeviceType.CUDA":
            continue
        base = e.name().split("<", 1)[0].lower()
        for ph in ("dispatch", "combine"):
            if ph in base:
                rows[ph].append((e.start_ns(), e.duration_ns(), e.name()))
                break
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kineto_kernels_rank{rank}.csv")
    g = max(1, group_size)
    with open(path, "w") as f:
        f.write("rank,phase,idx,replay,pos,start_ns,dur_us,gap_from_prev_same_phase_end_us,name\n")
        for ph, lst in rows.items():
            lst.sort()
            prev_end = None
            for i, (st, du, nm) in enumerate(lst):
                gap = "" if prev_end is None else f"{(st - prev_end) / 1e3:.3f}"
                prev_end = st + du
                f.write(f"{rank},{ph},{i},{i // g},{i % g},{st},{du / 1e3:.3f},{gap},\"{nm[:80]}\"\n")
    print(
        f"[kineto-dump] rank={rank} dispatch_kernels={len(rows['dispatch'])} "
        f"combine_kernels={len(rows['combine'])} group={g} -> {path}",
        flush=True,
    )


# ============================================================================
# CUDA-graph capture (owned by the harness, not the backends). Every EP backend
# replays dispatch+combine as ONE combined graph, so this single helper captures
# the paired op from any backend; the backend only supplies its capture-safe ops.
# ============================================================================
def _capture_paired_graph(dispatch_op, combine_op, prime=True, pre_capture=None, on_fail=None, graph_group_size=1):
    """Capture a paired dispatch->combine into ONE torch.cuda.CUDAGraph and return
    (replay_dispatch, replay_combine, graph). One replay runs BOTH phases, so
    replay_combine is a no-op. dispatch_op/combine_op are the backend's capture-safe
    ops (no CPU sync, no host collective) that go inside the graph.
    ``graph_group_size`` dispatch->combine iterations are captured inside the single
    graph so one replay runs them all -- this amortizes launch overhead and keeps the
    spin-waiting dispatch/combine kernels from being inflated by per-replay launch
    skew (reported times are divided back to per-iteration by the caller). Capture is
    best-effort: on any exception on_fail() is invoked (to let the backend reset its
    state) and None is returned so the caller keeps the eager path."""
    try:
        if prime:
            if pre_capture is not None:
                pre_capture()
            dispatch_op()
            combine_op()
            torch.cuda.synchronize()
        if pre_capture is not None:
            pre_capture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_group_size):
                dispatch_op()
                combine_op()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - capturability is an external-library boundary
        if on_fail is not None:
            on_fail()
        return None

    def replay_dispatch():
        graph.replay()
        return None

    def replay_combine(_dout):
        pass  # both phases already ran inside the single-graph replay

    return replay_dispatch, replay_combine, graph


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
):
    base_inputs = inputs[0] if isinstance(inputs, list) else inputs
    _, _, _, num_valid_selections = base_inputs
    hidden = args.hidden
    warmup, iters = args.num_warmup, args.num_iters
    disp_elt = 1 if getattr(args, "dispatch_dtype", "bf16") == "fp8_e4m3" else 2
    disp_bytes = num_valid_selections * hidden * disp_elt  # dispatch wire format
    comb_bytes = num_valid_selections * hidden * 2  # BF16 combine output (per ep_bench)

    stream = torch.cuda.current_stream()
    _sync_each_iter_env = os.environ.get("EP_SYNC_EACH_ITER")
    sync_each_iter = (
        _sync_each_iter_env == "1"
        if _sync_each_iter_env is not None
        else name == "mscclpp"
        and args.mode == "latency"
        and args.ep_layout in ("rank_major", "rank_major_topk_expanded")
        and os.environ.get("MSCCLPP_EP_ENABLE_GPUNETIO", "0") == "1"
    )
    _debug_pair_env = os.environ.get("EP_DEBUG_PAIR")
    debug_pair = _debug_pair_env == "1" if _debug_pair_env is not None else sync_each_iter

    # --- Warmup (paired). ---
    for warmup_idx in range(warmup):
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] warmup {warmup_idx} pair start", flush=True)
        dout = dispatch_fn()
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] warmup {warmup_idx} dispatch returned", flush=True)
        combine_fn(dout)
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] warmup {warmup_idx} combine returned", flush=True)
        stream.synchronize()
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] warmup {warmup_idx} stream synced", flush=True)
        comm.Barrier()
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] warmup {warmup_idx} barrier done", flush=True)

    # Kernel-only timing (EP_KERNEL_TIMER=kineto, the default): DeepEP bench_kineto-style
    # torch.profiler pass with an L2 flush and a GPU-side torch NCCL all_reduce barrier
    # per iteration (EP_KINETO_BARRIER=nccl) to align ranks on-device -- skew-free avg.
    use_kineto = os.environ.get("EP_KERNEL_TIMER", "kineto") == "kineto"
    have_kernel = use_kineto

    d_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # --- Timed loop (paired); no per-iter sync/barrier by default -- kernels pipeline back-to-back. ---
    for i in range(iters):
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] iter {i} pair start", flush=True)
        d_start[i].record(stream)
        dout = dispatch_fn()
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] iter {i} dispatch returned", flush=True)
        d_end[i].record(stream)
        c_start[i].record(stream)
        combine_fn(dout)
        if debug_pair:
            print(f"[pairdbg][{name}][rank {rank}] iter {i} combine returned", flush=True)
        c_end[i].record(stream)
        if sync_each_iter:
            comm.Barrier()
            stream.synchronize()
            if debug_pair:
                print(f"[pairdbg][{name}][rank {rank}] iter {i} stream synced", flush=True)
            comm.Barrier()
            if debug_pair:
                print(f"[pairdbg][{name}][rank {rank}] iter {i} barrier done", flush=True)

    if debug_pair:
        print(f"[pairdbg][{name}][rank {rank}] final stream sync start", flush=True)
    torch.cuda.synchronize()
    if debug_pair:
        print(f"[pairdbg][{name}][rank {rank}] final stream sync done", flush=True)

    ck_disp = ck_comb = 0.0
    inproc_ok = False
    if use_kineto:
        comm.Barrier()
        ck_disp, ck_comb = torch_profiler_kernel_us(
            dispatch_fn,
            combine_fn,
            comm,
            iters,
            barrier=(bench_barrier or nccl_barrier),
            mid_barrier=nccl_barrier,
            parse_kernels=parse_kernels,
            group_size=graph_group_size,
            rank=rank,
        )
        inproc_ok = ck_disp > 0.0 and ck_comb > 0.0

    # --- Per-iter times (ms->us), trim the first (warmup outlier). ---
    # graph_group_size is the number of dispatch->combine iterations captured in one
    # graph (the --iters-per-graph arg; 1 when this backend is not graph-captured).
    # One replay (one dispatch_fn() call) ran them all, so divide the per-replay time
    # back to per-iteration. Kernel-only kineto is already per-iteration (its
    # per-launch average divides by the kernel count, which scales with the group).
    group = max(1, graph_group_size)
    disp_us = [d_start[i].elapsed_time(d_end[i]) * 1e3 / group for i in range(iters)]
    comb_us = [max(c_start[i].elapsed_time(c_end[i]) * 1e3 / group, 1e-3) for i in range(iters)]
    tot_us = [d_start[i].elapsed_time(c_end[i]) * 1e3 / group for i in range(iters)]
    if iters > 1:
        disp_us, comb_us, tot_us = disp_us[1:], comb_us[1:], tot_us[1:]

    def stats(times):
        return sum(times) / len(times), min(times), max(times)

    d_avg, d_min, d_max = stats(disp_us)
    c_avg, c_min, c_max = stats(comb_us)
    t_avg, t_min, t_max = stats(tot_us)

    d_tp = (disp_bytes / 1e9) / (d_avg * 1e-6)
    c_tp = (comb_bytes / 1e9) / (c_avg * 1e-6)
    t_tp = ((disp_bytes + comb_bytes) / 1e9) / (t_avg * 1e-6)

    # --- Cross-rank reduction (mirror printLowLatencyResults). ---
    g_d_avg, g_d_min, g_d_max = _mpi_stats(comm, d_avg, d_min, d_max, num_ranks)
    g_c_avg, g_c_min, g_c_max = _mpi_stats(comm, c_avg, c_min, c_max, num_ranks)
    g_t_avg, g_t_min, g_t_max = _mpi_stats(comm, t_avg, t_min, t_max, num_ranks)

    d_tp_all = comm.gather(d_tp, root=0)
    c_tp_all = comm.gather(c_tp, root=0)
    t_tp_all = comm.gather(t_tp, root=0)

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

        print(f"\n=== Summary [{name}] ({args.mode.title()}, across {num_ranks} ranks) ===")
        print("\n--- Host-observed performance ---")
        print(f"Dispatch (BF16):  avg={g_d_avg:.2f} us, min={g_d_min:.2f} us, max={g_d_max:.2f} us")
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
            f"\nByte counts: dispatch={disp_bytes / 1e6:.2f} MB (BF16), "
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
    "flashinfer": setup_flashinfer,
}
# Per-backend kineto kernel-name parse, owned by each backend module.
_PARSE_KINETO = {
    "mscclpp": mscclpp_parse_kineto,
    "nccl": nccl_parse_kineto,
    "deepep": deepep_parse_kineto,
    "flashinfer": flashinfer_parse_kineto,
}


def main() -> None:
    args = parse_args()
    if args.ep_layout == "rank_major_topk_expanded" and (args.backend != "mscclpp" or args.mode != "latency"):
        raise ValueError("rank_major_topk_expanded requires --backend mscclpp --mode latency")
    comm, rank, num_ranks, local_rank = init_mpi()
    # Debug aid: EP_FAULTHANDLER_SECS>0 dumps every thread's Python traceback if the
    # process is still alive after N seconds (surfaces the exact hang location under
    # an mpirun timeout). Off unless the env var is set.
    _fh_secs = float(os.environ.get("EP_FAULTHANDLER_SECS", "0") or "0")
    if _fh_secs > 0:
        import faulthandler

        faulthandler.dump_traceback_later(_fh_secs, repeat=True)
    assert args.num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    local_world_size = int(os.environ.get("MSCCLPP_EP_LOCAL_WORLD_SIZE", "0") or "0")
    route_mode = None
    if args.mscclpp_cross_rack_topk_half:
        route_mode = "mscclpp_cross_rack_topk_half"
    elif args.mscclpp_same_rack_topk:
        route_mode = "mscclpp_same_rack_topk"
    elif args.mscclpp_cross_rack_token_half_all_remote:
        route_mode = "mscclpp_cross_rack_token_half_all_remote"
    elif args.mscclpp_rack0_gpu0_one_to_rack1:
        route_mode = "mscclpp_rack0_gpu0_one_to_rack1"
    elif args.mscclpp_rack0_gpu0_rack1_only:
        route_mode = "mscclpp_rack0_gpu0_rack1_only"
    elif args.mscclpp_rank0_all_to_rack1:
        route_mode = "mscclpp_rank0_all_to_rack1"
    if route_mode is not None:
        if num_ranks != 8 or local_world_size != 4:
            raise ValueError("mscclpp route diagnostics require 8 ranks and MSCCLPP_EP_LOCAL_WORLD_SIZE=4")
        if rank == 0:
            if route_mode == "mscclpp_cross_rack_topk_half":
                print(
                    "[cfg] route_mode=mscclpp_cross_rack_topk_half remote_topk=4 invalid_topk=4 "
                    "remote experts selected from random scores; same-node peer/local-self dispatch disabled",
                    flush=True,
                )
            elif route_mode == "mscclpp_same_rack_topk":
                print(
                    "[cfg] route_mode=mscclpp_same_rack_topk local_topk=all "
                    "local-rack experts selected from random scores; cross-rack dispatch disabled",
                    flush=True,
                )
            elif route_mode == "mscclpp_rack0_gpu0_one_to_rack1":
                print(
                    "[cfg] route_mode=mscclpp_rack0_gpu0_one_to_rack1 src_rank=0 "
                    "random_topk_all_ranks; all other ranks invalid",
                    flush=True,
                )
            elif route_mode == "mscclpp_rank0_all_to_rack1":
                print(
                    "[cfg] route_mode=mscclpp_rank0_all_to_rack1 src_rank=0 dst_ranks=4,5,6,7 "
                    f"active_tokens={os.environ.get('MSCCLPP_EP_ROUTE_RANK0_ACTIVE_TOKENS', args.num_tokens)}; "
                    "all other ranks invalid",
                    flush=True,
                )
            elif route_mode == "mscclpp_rack0_gpu0_rack1_only":
                print(
                    "[cfg] route_mode=mscclpp_rack0_gpu0_rack1_only src_rank=0 "
                    "random_topk with rack0 (ranks 0-3) entries invalid; all other ranks invalid; "
                    f"dst_ranks={os.environ.get('MSCCLPP_EP_ROUTE_RANK0_DST_RANKS', '4,5,6,7')}",
                    flush=True,
                )
            else:
                print(
                    "[cfg] route_mode=mscclpp_cross_rack_token_half_all_remote remote_tokens=64 "
                    "remote_ranks=all4 invalid_tokens=64; same-rack/local dispatch disabled",
                    flush=True,
                )
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
        route_mode=route_mode,
        num_ranks=num_ranks,
        local_world_size=local_world_size,
    )
    _print_topk_rank_slot_diagnostics(args, comm, rank, num_ranks, inputs[1], local_world_size)
    comm.Barrier()

    # Snapshot the user's EP_KINETO_SEPARATE so we can restore it per backend below
    # (cuda-graph capture toggles it for some backends only -- see the loop).
    _user_kineto_separate = os.environ.get("EP_KINETO_SEPARATE")

    nccl_barrier = None
    if (
        os.environ.get("EP_KERNEL_TIMER", "kineto") == "kineto"
        and os.environ.get("EP_KINETO_BARRIER", "nccl") == "nccl"
    ):
        try:
            nccl_barrier = _init_torch_nccl(comm, rank, num_ranks, local_rank)
            if rank == 0:
                print("[cfg] kineto barrier: torch NCCL all_reduce (GPU-side)", flush=True)
        except Exception as exc:
            if rank == 0:
                print(
                    f"[warn] torch NCCL barrier init failed ({type(exc).__name__}: {exc}); using MPI host barrier",
                    flush=True,
                )
            nccl_barrier = None

    # nccl is benchmarked before mscclpp (see module docstring: mscclpp LL init
    # perturbs CUDA state that breaks a later NCCL-EP cooperative launch).
    if args.backend == "all":
        backends = ["nccl", "mscclpp", "deepep", "flashinfer"]
    else:
        backends = [args.backend]

    for name in backends:
        # Restore the user's EP_KINETO_SEPARATE each iteration so a prior backend's
        # override (its own, or the single-graph force below) does not leak across
        # --backend all. A backend whose kineto pass has a hard requirement (e.g.
        # FlashInfer's stateful API cannot time combine alone) sets it inside its
        # own setup; the graph force below is layered on top.
        if _user_kineto_separate is None:
            os.environ.pop("EP_KINETO_SEPARATE", None)
        else:
            os.environ["EP_KINETO_SEPARATE"] = _user_kineto_separate

        try:
            ops = _SETUP[name](args, comm, rank, num_ranks, inputs)
        except Exception as exc:
            if args.backend != "all":
                raise
            if rank == 0:
                print(f"\n[skip] backend '{name}' setup failed: {type(exc).__name__}: {exc}", flush=True)
            comm.Barrier()
            continue

        dispatch_fn = ops["dispatch"]
        combine_fn = ops["combine"]
        teardown = ops["teardown"]
        backend_barrier = ops.get("barrier")

        # CUDA-graph capture is owned HERE, not in the backends: a backend that
        # supports it hands back capture-safe dispatch/combine ops and an optional
        # on-capture-failure reset. Capturing dispatch+combine as ONE graph means a
        # single replay runs both phases, so the skew-free separate kineto pass can
        # no longer isolate combine -- force the PAIRED single pass
        # (EP_KINETO_SEPARATE=0) so per-phase kernel time is still attributed by
        # kernel name. If capture fails, keep the eager ops.
        graph = None
        spec = ops.get("graph")
        effective_group_size = 1
        if spec is not None:
            graphed = _capture_paired_graph(
                spec["dispatch"],
                spec["combine"],
                pre_capture=spec.get("pre_capture"),
                on_fail=spec.get("on_fail"),
                graph_group_size=max(1, args.iters_per_graph),
            )
            local_ok = graphed is not None
            all_ok = comm.allreduce(1 if local_ok else 0, op=MPI.MIN)
            if not all_ok:
                graphed = None  # keep every rank on the eager path
            comm.Barrier()
            if graphed is not None:
                dispatch_fn, combine_fn, graph = graphed
                effective_group_size = max(1, args.iters_per_graph)
                os.environ["EP_KINETO_SEPARATE"] = "0"
                if rank == 0:
                    print(
                        f"[cfg] {name} cuda_graph captured "
                        f"(single graph; dispatch+combine; iters_per_graph={effective_group_size})",
                        flush=True,
                    )
        try:
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
            )
        finally:
            torch.cuda.synchronize()
            # Drop the replay closures and the captured graph BEFORE teardown frees
            # the buffers/handles the graph captured.
            dispatch_fn = combine_fn = None
            graph = None
            teardown()
            comm.Barrier()
    if os.environ.get("EP_BOOTSTRAP") == "torch":
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
