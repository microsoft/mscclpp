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

* **Paired** ``dispatch -> sync -> combine -> sync -> barrier`` per iteration.
* **Per-iteration CUDA events** recorded on the stream around each launch, with the
  ``stream.synchronize()`` and the cross-rank barrier *outside* the timed region.
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

Backend ordering (``--backend both``)
-------------------------------------
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
        python run_ep_bench_python.py --backend both -e 128

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
            --backend both -e 128 -t 128 -d 7168 -k 8 -w 10 -i 50

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
import gc
import glob
import os
import subprocess
import time

import torch
from mpi4py import MPI


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
        choices=["mscclpp", "nccl", "both"],
        default="both",
        help="which backend(s) to benchmark in this run (both runs nccl then mscclpp)",
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
        help="mscclpp: capture dispatch and combine as CUDA graphs and replay them in the timed loop.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="mscclpp: run a one-time combine correctness check before timing.",
    )
    p.add_argument(
        "--kernel-timing",
        action="store_true",
        help="also measure pure device (kernel) time via the in-process CUPTI Activity "
        "collector (libcupti_kernel_timer.so), reported per backend as a separate "
        "'--- Kernel-only performance ---' block. Off by default.",
    )
    # NCCL-EP JIT knobs (defaults match the in-tree build; used by nccl/both).
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
    if args.dispatch_dtype == "fp8_e4m3" and args.backend in ("nccl", "both"):
        raise SystemExit("--dispatch-dtype fp8_e4m3 is only supported by the mscclpp backend; use --backend mscclpp")
    return args


# ----------------------------------------------------------------------------
# MPI bootstrap shared by both backends.
# ----------------------------------------------------------------------------
def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return comm, rank, size, local_rank


def _mpi_stats(comm, avg: float, mn: float, mx: float, num_ranks: int):
    """Cross-rank reduction mirroring printLowLatencyResults: avg=mean of per-rank
    avgs, min=global MIN, max=global MAX."""
    g_avg = comm.allreduce(avg, op=MPI.SUM) / num_ranks
    g_min = comm.allreduce(mn, op=MPI.MIN)
    g_max = comm.allreduce(mx, op=MPI.MAX)
    return g_avg, g_min, g_max


# ----------------------------------------------------------------------------
# Routing inputs — shared by both backends so the comparison is apples-to-apples.
# BF16 tokens + top-k routing setup.
# ----------------------------------------------------------------------------
def make_inputs(num_tokens, hidden, num_topk, num_experts, rank, seed):
    torch.manual_seed(seed + rank)
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    # ep_bench byte accounting: num_valid_selections = count(topk_idx >= 0); every
    # selection is valid here (a full LL load), so this equals num_tokens * top_k.
    num_valid_selections = int((topk_idx >= 0).sum().item())
    return x, topk_idx, topk_weights, num_valid_selections


# ----------------------------------------------------------------------------
# LL dtype / combine helpers (ported from test_low_latency_multirank.py).
# ----------------------------------------------------------------------------
def fp8_e4m3_block128_scales(x):
    blocks = x.float().reshape(*x.shape[:-1], x.size(-1) // 128, 128)
    max_abs = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    return max_abs / 448.0


def simulated_gemm_output(dispatch_out):
    """Simulate the downstream expert GEMM so combine consumes BF16 expert output:
    identity for BF16 dispatch; dequantize (tokens * block_scales) for FP8_E4M3."""
    if dispatch_out.quant is None:
        return dispatch_out.tokens
    tokens = dispatch_out.tokens
    token_blocks = tokens.float().reshape(*tokens.shape[:-1], tokens.size(-1) // 128, 128)
    return (token_blocks * dispatch_out.quant.block_scales.unsqueeze(-1)).reshape(tokens.shape).to(torch.bfloat16)


def validate_combine_output_mpi(actual, expected, comm, *, exact):
    """MPI analog of the test's validate_combine_output: global max abs diff plus a
    cross-rank finiteness (and, for direct_send, bit-exactness) assertion."""
    local_diff = float((actual.float() - expected.float()).abs().max().item())
    global_diff = comm.allreduce(local_diff, op=MPI.MAX)
    local_finite = int(torch.isfinite(actual).all().item())
    assert comm.allreduce(local_finite, op=MPI.MIN) == 1, "LL combine output contains NaN or Inf"
    if exact:
        local_equal = int(torch.equal(actual, expected))
        assert comm.allreduce(local_equal, op=MPI.MIN) == 1, f"LL direct-send combine not bit-exact; diff={global_diff}"
    else:
        assert global_diff <= 8.0, f"LL rank-local combine mismatch; max diff={global_diff}"
    return global_diff


# ============================================================================
# Backend: mscclpp EP (MoECommunicator).
# ============================================================================
def setup_mscclpp(args, comm, rank, num_ranks, inputs):
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks

    num_rdma_bytes = 0  # not exposed by current mscclpp API; 0 over the CUDA-IPC path
    if rank == 0:
        print(
            f"[cfg] backend=mscclpp algorithm=LOW_LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters} num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    ep_group = CommGroup(mpi_comm=comm)
    combine_mode = {
        "rank_local_reduce": ep.CombineMode.RANK_LOCAL_REDUCE,
        "direct_send": ep.CombineMode.DIRECT_SEND,
    }[args.combine_mode]
    dispatch_quant = ep.QuantConfig(format=ep.DispatchDataType.FP8_E4M3) if args.dispatch_dtype == "fp8_e4m3" else None
    dispatch_dtype = torch.float8_e4m3fn if dispatch_quant is not None else torch.bfloat16
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        low_latency_combine_mode=combine_mode,
        quant=dispatch_quant,
    )
    assert moe_comm.is_available()
    if rank == 0:
        print(
            f"[cfg] mscclpp MoECommunicator is_internode={moe_comm.is_internode()} "
            f"dispatch_dtype={args.dispatch_dtype} combine_mode={args.combine_mode} cuda_graph={args.cuda_graph}",
            flush=True,
        )

    # Hoist output tensors out of the timed loop (the communicator owns its
    # src_info/layout_range/count buffers internally).
    output_buffer = torch.empty(
        (num_local_experts, num_ranks * num_tokens, hidden), dtype=dispatch_dtype, device="cuda"
    )
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def _dispatch():
        # Full (send+recv) LL dispatch inline on the stream; returns (dispatch_out, handle).
        return moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)

    def _combine(dispatch_out, handle):
        # Feed BF16 expert output (identity for BF16, dequantized for FP8) into combine.
        moe_comm.combine(simulated_gemm_output(dispatch_out), handle, out=out)

    # Optional one-time correctness check (mirrors test_low_latency_multirank).
    if args.validate:
        v_dispatch_out, v_handle = _dispatch()
        v_out = torch.empty_like(out)
        moe_comm.combine(simulated_gemm_output(v_dispatch_out), v_handle, out=v_out)
        torch.cuda.synchronize()
        if dispatch_quant is None:
            expected_f = torch.zeros_like(x, dtype=torch.float32)
            x_f = x.float()
            for j in range(num_topk):
                weight_j = topk_weights[:, j].masked_fill(topk_idx[:, j] < 0, 0.0).view(-1, 1)
                expected_f = torch.addcmul(expected_f, x_f, weight_j)
            gdiff = validate_combine_output_mpi(
                v_out, expected_f.to(torch.bfloat16), comm, exact=args.combine_mode == "direct_send"
            )
            if rank == 0:
                print(f"[validate] mscclpp combine OK max|got-expected|={gdiff:.4e}", flush=True)
        else:
            assert torch.isfinite(v_out).all().item(), "FP8 LL combine produced NaN/Inf"
            if rank == 0:
                print("[validate] mscclpp FP8 combine finite OK", flush=True)

    state = {"moe": moe_comm, "obuf": output_buffer, "out": out, "grp": ep_group}

    if args.cuda_graph:
        # Prime once, then capture dispatch and combine as separate CUDA graphs so
        # the shared timed loop keeps its per-phase dispatch/combine events.
        prime_out, prime_handle = _dispatch()
        _combine(prime_out, prime_handle)
        torch.cuda.synchronize()

        g_dispatch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_dispatch):
            g_dispatch_out, g_handle = moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)
        g_combine = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_combine):
            moe_comm.combine(simulated_gemm_output(g_dispatch_out), g_handle, out=out)
        state["graphs"] = (g_dispatch, g_combine)

        def dispatch_fn():
            g_dispatch.replay()
            return (g_dispatch_out, g_handle)

        def combine_fn(dout):
            g_combine.replay()

    else:

        def dispatch_fn():
            return _dispatch()

        def combine_fn(dout):
            dispatch_out, handle = dout
            _combine(dispatch_out, handle)

    def teardown():
        state.clear()
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown


# ============================================================================
# Backend: NVIDIA NCCL-EP (nccl.ep Group/Handle).
# ============================================================================
def setup_nccl(args, comm, rank, num_ranks, inputs):
    os.environ.setdefault("NCCL_EP_JIT_SOURCE_DIR", args.nccl_jit_source_dir)
    os.environ.setdefault("NCCL_EP_JIT_BUILD_INCLUDE_DIR", args.nccl_jit_include_dir)

    import nccl.core as nccl_core
    import nccl.ep as nccl_ep

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks

    if rank == 0:
        print(
            f"[cfg] backend=nccl algorithm=LOW_LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )

    # NCCL communicator: rank 0 makes a unique id, broadcast over MPI.
    uid = nccl_core.get_unique_id() if rank == 0 else None
    uid = comm.bcast(uid, root=0)
    ncomm = nccl_core.Communicator.init(nranks=num_ranks, rank=rank, unique_id=uid)

    config = nccl_ep.GroupConfig(
        algorithm=nccl_ep.Algorithm.LOW_LATENCY,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=num_tokens,
        max_recv_tokens_per_rank=num_tokens * num_ranks,
        max_token_bytes=hidden * 2,  # BF16
        alloc=nccl_ep.AllocConfig(),  # default cudaMalloc/cudaFree
    )
    ep_group = nccl_ep.Group.create(ncomm, config)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Routing is encoded in the handle at create time (topk_idx is fixed for the run).
    topk_idx_t = nccl_ep.Tensor(topk_idx)
    ep_handle = ep_group.create_handle(
        nccl_ep.Layout.EXPERT_MAJOR,
        topk_idx_t,
        layout_info=None,
        config=nccl_ep.HandleConfig(),
        stream=stream_ptr,
    )

    # Pre-allocated EP tensors (hoisted out of the timed loop).
    recv = torch.empty((num_local_experts, num_ranks * num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    recv_count = torch.empty((num_local_experts,), dtype=torch.int32, device="cuda")
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    x_t = nccl_ep.Tensor(x)
    recv_t = nccl_ep.Tensor(recv)
    recv_count_t = nccl_ep.Tensor(recv_count)
    out_t = nccl_ep.Tensor(out)
    topk_weights_t = nccl_ep.Tensor(topk_weights)

    dispatch_inputs = nccl_ep.DispatchInputs(tokens=x_t)
    dispatch_outputs = nccl_ep.DispatchOutputs(tokens=recv_t)
    dispatch_layout = nccl_ep.LayoutInfo(expert_counters=recv_count_t)
    dispatch_config = nccl_ep.DispatchConfig()
    combine_inputs = nccl_ep.CombineInputs(tokens=recv_t)
    combine_outputs = nccl_ep.CombineOutputs(tokens=out_t, topk_weights=topk_weights_t)
    combine_config = nccl_ep.CombineConfig()

    def dispatch_fn():
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=dispatch_config,
            stream=stream_ptr,
        )
        ep_handle.complete(stream=stream_ptr)
        return None

    def combine_fn(_dout):
        ep_handle.combine(combine_inputs, combine_outputs, config=combine_config, stream=stream_ptr)
        ep_handle.complete(stream=stream_ptr)

    def teardown():
        ep_handle.destroy()
        ep_group.destroy()
        ncomm.destroy()
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown


# ============================================================================
# Shared paired benchmark + summary (mirrors mscclpp_ep_bench.cu / ep_bench).
# ============================================================================
def run_backend(name, args, comm, rank, num_ranks, inputs, dispatch_fn, combine_fn, cupti=None):
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

    # In-process CUPTI collector (ep_bench KernelTimer analog): start after warmup,
    # stop after the timed loop -- the same window as the host CUDA events.
    inproc_rc = -1
    if cupti is not None:
        torch.cuda.synchronize()
        comm.Barrier()
        inproc_rc = cupti.start()

    d_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # --- Timed loop (paired); sync + barrier are OUTSIDE the recorded region. ---
    for i in range(iters):
        d_start[i].record(stream)
        dout = dispatch_fn()
        d_end[i].record(stream)
        stream.synchronize()
        c_start[i].record(stream)
        combine_fn(dout)
        c_end[i].record(stream)
        stream.synchronize()
        comm.Barrier()

    torch.cuda.synchronize()

    ck_disp = ck_comb = 0.0
    inproc_ok = False
    if cupti is not None and inproc_rc == 0:
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
    if cupti is not None:
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

        if cupti is not None:
            print("\n--- Kernel-only performance (in-process CUPTI Activity API) ---")
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


_SETUP = {"mscclpp": setup_mscclpp, "nccl": setup_nccl}


def main() -> None:
    args = parse_args()
    comm, rank, num_ranks, local_rank = init_mpi()
    assert args.num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"

    inputs = make_inputs(args.num_tokens, args.hidden, args.num_topk, args.num_experts, rank, args.seed)

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
    if args.backend == "both":
        backends = ["nccl", "mscclpp"]
    else:
        backends = [args.backend]

    for name in backends:
        try:
            dispatch_fn, combine_fn, teardown = _SETUP[name](args, comm, rank, num_ranks, inputs)
        except Exception as exc:
            if rank == 0:
                print(f"\n[skip] backend '{name}' setup failed: {type(exc).__name__}: {exc}", flush=True)
            comm.Barrier()
            continue
        try:
            run_backend(name, args, comm, rank, num_ranks, inputs, dispatch_fn, combine_fn, cupti=cupti)
        finally:
            torch.cuda.synchronize()
            teardown()
            comm.Barrier()


if __name__ == "__main__":
    main()
