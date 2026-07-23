#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified in-process low-latency (LL) EP benchmark that drives BOTH backends'
Python APIs directly, inside the *same* measurement flow:

* **mscclpp EP**   — ``mscclpp.ep.MoECommunicator.dispatch`` / ``.combine``.
* **NVIDIA NCCL-EP** — ``nccl.ep.Group`` / ``nccl.ep.Handle.dispatch`` / ``.combine``
  (the ``nccl4py`` Pythonic bindings for ``libnccl_ep.so``).

This is a Python port of ``mscclpp_ep_bench.cu`` that additionally understands the
NCCL-EP Python API. Unlike ``run_ep_bench.py`` (which shells out to a separate
per-backend process for each measurement), this script calls each backend's Python
API *in one process* through a single shared paired-benchmark loop, so the two are
timed with byte-for-byte the same methodology:

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
  run can be diffed directly against ``ep_bench`` / ``mscclpp_ep_bench.cu``.

Bootstrap: MPI (mpi4py + mpirun)
--------------------------------
Both backends share an MPI ``COMM_WORLD`` bootstrap (the same mechanism the C++
``mscclpp_ep_bench`` and NCCL-EP ``ep_test.py`` use), *not* torch.distributed:

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
import ctypes
import glob
import os
import subprocess
import time

import torch
from mpi4py import MPI

from ep_bench_common import (
    init_mpi,
    make_inputs,
    _init_torch_nccl,
    _mpi_stats,
)
from ep_bench_mscclpp import setup_mscclpp
from ep_bench_nccl import setup_nccl
from ep_bench_deepep import setup_deepep
from ep_bench_flashinfer import setup_flashinfer


# ----------------------------------------------------------------------------
# CLI — mirrors ep_bench.cu's getopt flags for the LL path, plus --backend.
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified low-latency EP benchmark (mscclpp + NCCL-EP Python APIs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["mscclpp", "nccl", "deepep", "flashinfer", "all"],
        default="all",
        help="which backend(s) to benchmark in this run (all runs nccl, mscclpp, deepep, flashinfer)",
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
        help="mscclpp LL dispatch wire format. NCCL-EP path is bf16 only.",
    )
    p.add_argument(
        "--combine-mode",
        "--optimized-combine-mode",
        choices=("rank_local_reduce", "direct_send"),
        default="rank_local_reduce",
        help="mscclpp LL combine mode (direct_send is bit-exact; rank_local_reduce is faster).",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture dispatch and combine as CUDA graphs and replay them in the timed loop "
        "(mscclpp, nccl, deepep cached path; flashinfer captures kernels with the MPI barrier kept outside).",
    )
    p.add_argument(
        "--ep-layout",
        choices=["native", "rank_major", "expert_major"],
        default="native",
        help="received-token dispatch layout. 'native' means no cross-backend normalization: each "
        "backend keeps the layout its own default code path produces "
        "(nccl=expert_major, mscclpp=expert_major, deepep=rank_major, flashinfer=rank_major), which is "
        "the apples-to-apples default for comparing each backend's out-of-the-box behavior. "
        "'rank_major'/'expert_major' force a specific layout where supported: nccl "
        "(Layout.RANK_MAJOR/EXPERT_MAJOR) and deepep (rank_major=plain, expert_major=do_expand). "
        "mscclpp LL is expert-major only and flashinfer is rank-major only; an unsupported request is "
        "noted and the native layout is kept.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="mscclpp: run a one-time combine correctness check before timing.",
    )
    p.add_argument(
        "--kernel-timing",
        action="store_true",
        help="build/use the in-process CUPTI Activity collector (libcupti_kernel_timer.so) "
        "for the kernel-only block. Only needed when EP_KERNEL_TIMER=cupti; the default "
        "kernel timer is torch kineto (EP_KERNEL_TIMER=kineto) with a GPU-side torch NCCL "
        "barrier (EP_KINETO_BARRIER=nccl), which needs no CUPTI build.",
    )
    # NCCL-EP JIT knobs: nccl.ep runtime-compiles (JITs) its device kernels on first use and
    # must be told where the device sources (device/*.cuh) and NCCL public headers live. The
    # defaults match the standard in-tree install, so these are normally unused; they are exposed
    # only so a non-standard NCCL-EP install location can be overridden without editing the script.
    p.add_argument(
        "--nccl-jit-source-dir",
        default=os.environ.get("NCCL_EP_JIT_SOURCE_DIR", "/opt/microsoft/mrc/ep/nccl/contrib/nccl_ep"),
        help="NCCL_EP_JIT_SOURCE_DIR (dir containing device/*.cuh for the runtime JIT)",
    )
    p.add_argument(
        "--nccl-jit-include-dir",
        default=os.environ.get("NCCL_EP_JIT_BUILD_INCLUDE_DIR", "/opt/microsoft/mrc/ep/nccl/build/include"),
        help="NCCL_EP_JIT_BUILD_INCLUDE_DIR (NCCL public headers for the runtime JIT)",
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
    if args.dispatch_dtype == "fp8_e4m3" and args.backend in ("nccl", "all"):
        raise SystemExit("--dispatch-dtype fp8_e4m3 is only supported by the mscclpp backend; use --backend mscclpp")
    return args


# ============================================================================
# Shared paired benchmark + summary (mirrors mscclpp_ep_bench.cu / ep_bench).
# ============================================================================
def _flush_l2_cache():
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


def _kineto_kernel_us(
    dispatch_fn, combine_fn, comm, num_tests, flush_l2=True, use_barrier=True, barrier=None, mid_barrier=None
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
    the combine recv-spin skew. Set EP_KINETO_SEPARATE=0 for the legacy paired
    loop."""
    import torch.profiler as _tp

    use_mid = os.environ.get("EP_KINETO_BARRIER_COMBINE", "0") == "1" and mid_barrier is not None
    separate = os.environ.get("EP_KINETO_SEPARATE", "1") == "1"

    def _do_barrier():
        if not use_barrier:
            return
        torch.cuda._sleep(int(2e7))  # ~10 ms GPU spin to absorb host launch skew
        if barrier is not None:
            barrier()  # GPU-side barrier (aligns ranks on-device)
        else:
            comm.Barrier()  # MPI host barrier (host-only alignment)

    def _parse(ka, substr):
        # Sum each DISTINCT matching kernel's average-per-launch. Single-kernel
        # backends (mscclpp, NCCL-EP) yield that kernel's avg; two-kernel DeepEP
        # (*_impl + *_epilogue) yields their per-iteration SUM (scope-matched).
        # Case-insensitive so FlashInfer's moeA2ADispatchKernel / moeA2ACombineKernel
        # (capitalized) match the "dispatch"/"combine" buckets too.
        total_us = 0.0
        matched = False
        sub = substr.lower()
        for e in ka:
            if sub in e.key.lower() and int(e.count) > 0:
                total_us += float(e.self_device_time_total) / int(e.count)
                matched = True
        return total_us if matched else 0.0

    if separate:
        # ---- Two separate passes, each: [flush; barrier; single op] ----
        # This mirrors DeepEP bench_kineto called once per op, aligning each
        # kernel's entry across ranks. No barrier ever sits between a paired
        # dispatch->combine, so it is safe for DeepEP multi-node.
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
        # replay combine alone (DeepEP uses its fixed primed handle; mscclpp /
        # NCCL-EP consume this dout each iteration).
        dout = dispatch_fn()
        torch.cuda.synchronize()
        ka_c = _run_pass(lambda: combine_fn(dout))
        return _parse(ka_d, "dispatch"), _parse(ka_c, "combine")

    # ---- Legacy paired loop (EP_KINETO_SEPARATE=0) ----
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
    return _parse(ka, "dispatch"), _parse(ka, "combine")


def run_backend(
    name,
    args,
    comm,
    rank,
    num_ranks,
    inputs,
    dispatch_fn,
    combine_fn,
    cupti=None,
    nccl_barrier=None,
    bench_barrier=None,
):
    _, _, _, num_valid_selections = inputs
    hidden = args.hidden
    warmup, iters = args.num_warmup, args.num_iters
    disp_elt = 1 if getattr(args, "dispatch_dtype", "bf16") == "fp8_e4m3" else 2
    disp_bytes = num_valid_selections * hidden * disp_elt  # dispatch wire format
    comb_bytes = num_valid_selections * hidden * 2  # BF16 combine output (per ep_bench)

    stream = torch.cuda.current_stream()

    # --- Warmup (paired). ---
    for _ in range(warmup):
        dout = dispatch_fn()
        stream.synchronize()
        combine_fn(dout)
        stream.synchronize()
        comm.Barrier()

    # Kernel-only timing. Default (EP_KERNEL_TIMER=kineto): DeepEP bench_kineto-style
    # torch.profiler pass with an L2 flush and a GPU-side torch NCCL all_reduce barrier
    # per iteration (EP_KINETO_BARRIER=nccl) to align ranks on-device -- skew-free avg.
    # EP_KERNEL_TIMER=cupti falls back to the in-process CUPTI collector (start after
    # warmup, stop after the timed loop -- same window as the host CUDA events).
    use_kineto = os.environ.get("EP_KERNEL_TIMER", "kineto") == "kineto"
    have_kernel = (cupti is not None) or use_kineto
    inproc_rc = -1
    if cupti is not None and not use_kineto:
        torch.cuda.synchronize()
        comm.Barrier()
        inproc_rc = cupti.start()

    d_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # --- Timed loop (paired); no per-iter sync/barrier -- kernels pipeline back-to-back. ---
    for i in range(iters):
        d_start[i].record(stream)
        dout = dispatch_fn()
        d_end[i].record(stream)
        c_start[i].record(stream)
        combine_fn(dout)
        c_end[i].record(stream)

    torch.cuda.synchronize()

    ck_disp = ck_comb = 0.0
    inproc_ok = False
    if use_kineto:
        comm.Barrier()
        ck_disp, ck_comb = _kineto_kernel_us(
            dispatch_fn, combine_fn, comm, iters, barrier=(bench_barrier or nccl_barrier), mid_barrier=nccl_barrier
        )
        inproc_ok = ck_disp > 0.0 and ck_comb > 0.0
    elif cupti is not None and inproc_rc == 0:
        cupti.stop()
        comm.Barrier()
        ck_disp = cupti.avg_us("dispatch")
        ck_comb = cupti.avg_us("combine")
        inproc_ok = ck_disp > 0.0 and ck_comb > 0.0

    # --- Per-iter times (ms->us), trim the first (warmup outlier). ---
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

    # --- Kernel-only (in-process CUPTI) cross-rank reduction. The LL dispatch
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

        print(f"\n=== Summary [{name}] (Low Latency, across {num_ranks} ranks) ===")
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
            _kt_hdr = "torch kineto (per-iter barrier + L2 flush)" if use_kineto else "in-process CUPTI Activity API"
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
                print("  NOTE: in-process CUPTI captured 0 LL kernels (collector unavailable).")

        print(
            f"\nByte counts: dispatch={disp_bytes / 1e6:.2f} MB (BF16), "
            f"combine={comb_bytes / 1e6:.2f} MB (BF16), selections={num_valid_selections}"
        )


# ----------------------------------------------------------------------------
# In-process CUPTI kernel timer (pure device time), shared by both backends.
# ----------------------------------------------------------------------------
def _cuda_inc_lib():
    home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    inc = os.path.join(home, "include")
    cands = glob.glob(os.path.join(home, "targets", "*", "lib", "libcupti.so")) + [
        os.path.join(home, "lib64", "libcupti.so"),
        os.path.join(home, "lib", "libcupti.so"),
    ]
    lib = next((os.path.dirname(c) for c in cands if os.path.exists(c)), os.path.join(home, "lib64"))
    return inc, lib


def _ensure_cupti_lib(comm, local_rank):
    """Build libcupti_kernel_timer.so next to this file if missing. Only one rank
    per node attempts to compile, and an O_EXCL lock file guarantees that even
    across nodes sharing the filesystem only a single compiler runs at a time
    (the losers wait for the winner's build); everyone then barriers."""
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "libcupti_kernel_timer.so")
    src = os.path.join(here, "cupti_kernel_timer.cpp")
    lock = so + ".lock"
    if not os.path.exists(so) and local_rank == 0 and os.path.exists(src):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            fd = None
        if fd is not None:
            try:
                if not os.path.exists(so):
                    inc, lib = _cuda_inc_lib()
                    subprocess.run(
                        ["g++", "-O2", "-fPIC", "-shared", src, "-o", so, f"-I{inc}", f"-L{lib}", "-lcupti"],
                        check=True,
                    )
            finally:
                os.close(fd)
                try:
                    os.unlink(lock)
                except FileNotFoundError:
                    pass
        else:
            # Another builder holds the lock; wait for the .so to appear.
            for _ in range(600):
                if os.path.exists(so):
                    break
                time.sleep(0.5)
    comm.Barrier()
    return so if os.path.exists(so) else None


class _InProcCupti:
    """ctypes wrapper over libcupti_kernel_timer.so -- a faithful analog of
    ep_bench's CUPTI KernelTimer. start()/stop() bracket the timed loop; the
    Activity API records per-kernel GPU time (CUPTI_ACTIVITY_KIND_KERNEL, which --
    unlike CONCURRENT_KERNEL -- captures the cooperative-launch LL kernels), then
    avg_us('dispatch'/'combine') buckets by mangled-name substring. Both backends'
    LL kernels are named ``dispatch`` / ``combine``, so the same buckets serve both.
    kt_start() clears prior stats, so one collector can be reused per backend."""

    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self.lib.kt_start.restype = ctypes.c_int
        self.lib.kt_stop.restype = ctypes.c_int
        self.lib.kt_get_avg_us.restype = ctypes.c_double
        self.lib.kt_get_avg_us.argtypes = [ctypes.c_char_p]
        self.lib.kt_get_count.restype = ctypes.c_long
        self.lib.kt_get_count.argtypes = [ctypes.c_char_p]

    def start(self):
        return int(self.lib.kt_start())

    def stop(self):
        return int(self.lib.kt_stop())

    def avg_us(self, substr):
        return float(self.lib.kt_get_avg_us(substr.encode()))

    def count(self, substr):
        return int(self.lib.kt_get_count(substr.encode()))


_SETUP = {"mscclpp": setup_mscclpp, "nccl": setup_nccl, "deepep": setup_deepep, "flashinfer": setup_flashinfer}


def main() -> None:
    args = parse_args()
    comm, rank, num_ranks, local_rank = init_mpi()
    # Debug aid: EP_FAULTHANDLER_SECS>0 dumps every thread's Python traceback if the
    # process is still alive after N seconds (surfaces the exact hang location under
    # an mpirun timeout). Off unless the env var is set.
    _fh_secs = float(os.environ.get("EP_FAULTHANDLER_SECS", "0") or "0")
    if _fh_secs > 0:
        import faulthandler

        faulthandler.dump_traceback_later(_fh_secs, repeat=True)
    assert args.num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    inputs = make_inputs(args.num_tokens, args.hidden, args.num_topk, args.num_experts, rank, args.seed)

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

    cupti = None
    if args.kernel_timing:
        so_path = _ensure_cupti_lib(comm, local_rank)
        if so_path is not None:
            try:
                cupti = _InProcCupti(so_path)
            except OSError as exc:
                if rank == 0:
                    print(f"[warn] CUPTI collector unavailable ({exc}); host-observed only", flush=True)
        elif rank == 0:
            print("[warn] libcupti_kernel_timer.so missing/unbuilt; host-observed only", flush=True)

    # nccl is benchmarked before mscclpp (see module docstring: mscclpp LL init
    # perturbs CUDA state that breaks a later NCCL-EP cooperative launch).
    if args.backend == "all":
        backends = ["nccl", "mscclpp", "deepep", "flashinfer"]
    else:
        backends = [args.backend]

    for name in backends:
        # CUDA-graph replay needs the PAIRED kineto pass ONLY for the paired-collective
        # backends whose combine consumes fresh dispatch output (nccl, flashinfer):
        # the separate pass replays combine alone and dead-spins on the peer receive
        # under graph replay. DeepEP replays a FIXED primed handle, so its separate
        # pass is safe AND necessary -- separate-pass inserts a per-op GPU barrier that
        # collapses DeepEP's combine recv-spin skew (the paired loop only barriers
        # before dispatch, inflating the combine avg). So keep DeepEP (and mscclpp) on
        # the user's default. Restore the snapshot each iteration so a prior backend's
        # override does not leak in --backend all.
        if _user_kineto_separate is None:
            os.environ.pop("EP_KINETO_SEPARATE", None)
        else:
            os.environ["EP_KINETO_SEPARATE"] = _user_kineto_separate
        if args.cuda_graph and name in ("nccl", "flashinfer"):
            os.environ["EP_KINETO_SEPARATE"] = "0"

        try:
            _setup_ret = _SETUP[name](args, comm, rank, num_ranks, inputs)
            if len(_setup_ret) == 4:
                dispatch_fn, combine_fn, teardown, backend_barrier = _setup_ret
            else:
                dispatch_fn, combine_fn, teardown = _setup_ret
                backend_barrier = None
        except Exception as exc:
            if rank == 0:
                print(f"\n[skip] backend '{name}' setup failed: {type(exc).__name__}: {exc}", flush=True)
            comm.Barrier()
            continue
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
                cupti=cupti,
                nccl_barrier=nccl_barrier,
                bench_barrier=backend_barrier,
            )
        finally:
            torch.cuda.synchronize()
            teardown()
            comm.Barrier()


if __name__ == "__main__":
    main()
