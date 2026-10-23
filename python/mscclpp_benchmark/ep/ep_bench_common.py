# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared helpers for the unified EP benchmark backends (bootstrap, input generation, quantization, and cross-rank validation)."""

from __future__ import annotations

import os
import torch

from mpi4py import MPI


# ----------------------------------------------------------------------------
# Bootstrap shared by the benchmark backends.
# ----------------------------------------------------------------------------
def init_mpi():
    if "TORCHELASTIC_RUN_ID" in os.environ:
        raise RuntimeError("Launch the EP benchmark with mpirun; torchrun is unsupported")
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local.rank
    local.Free()
    torch.cuda.set_device(local_rank)
    return comm, rank, size


def _ensure_torch_dist(comm, rank, num_ranks):
    """Lazily initialize the default torch.distributed NCCL group alongside MPI
    (idempotent). MPI supplies the rendezvous (rank-0 IP + port broadcast). Both
    the kineto GPU barrier and the DeepEP backend reuse this single group.
    Returns the world ProcessGroup."""
    import torch.distributed as dist

    if not dist.is_initialized():
        addr = comm.bcast(os.environ.get("MASTER_ADDR") if rank == 0 else None, root=0)
        if not addr:
            raise RuntimeError("Set MASTER_ADDR to rank 0's reachable address before launching the EP benchmark")
        port = int(comm.bcast(os.environ.get("MASTER_PORT", "29700") if rank == 0 else None, root=0))
        host = f"[{addr}]" if ":" in addr and not addr.startswith("[") else addr
        import datetime as _dt

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{host}:{port}",
            world_size=num_ranks,
            rank=rank,
            timeout=_dt.timedelta(seconds=120),
        )
    group = dist.group.WORLD
    if str(dist.get_backend(group)).lower() != "nccl":
        raise RuntimeError("EP GPU profiling and DeepEP require an NCCL process group")
    # DeepEP accesses the native NCCL communicator, which PyTorch initializes lazily.
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
    return group


def _init_torch_nccl(comm, rank, num_ranks):
    """Return a zero-arg GPU-side barrier (torch NCCL all_reduce) for the kineto
    timing loop -- aligns ranks on-device, much tighter than an MPI host barrier."""
    import torch.distributed as dist

    group = _ensure_torch_dist(comm, rank, num_ranks)
    _sync = torch.ones(1, dtype=torch.float, device="cuda")

    def _barrier():
        dist.all_reduce(_sync, group=group)

    _barrier()
    torch.cuda.synchronize()
    return _barrier


def _mpi_stats(comm, avg: float, mn: float, mx: float, num_ranks: int):
    """Cross-rank reduction mirroring printLowLatencyResults: avg=mean of per-rank
    avgs, min=global MIN, max=global MAX."""
    g_avg = comm.allreduce(avg, op=MPI.SUM) / num_ranks
    g_min = comm.allreduce(mn, op=MPI.MIN)
    g_max = comm.allreduce(mx, op=MPI.MAX)
    return g_avg, g_min, g_max


# ----------------------------------------------------------------------------
# Kineto kernel-name parsing. Each backend module owns a `parse_kineto_kernels`
# that maps a torch.profiler key_averages() table to (dispatch_us, combine_us)
# using that library's own kernel names; they delegate the summation here so the
# only per-library knowledge in each backend is the kernel-name substrings.
# ----------------------------------------------------------------------------
def sum_matching_kernel_us(key_averages, substrs):
    """Sum each DISTINCT matching kernel's average-per-launch, so single-kernel
    backends yield that kernel's avg and multi-kernel backends yield their
    per-iteration SUM (scope-matched). ``substrs`` is an iterable of lowercase-
    insensitive name substrings; matching strips C++ template arguments so a
    combine kernel templated on DispatchLayout (e.g. combineKernel<..,
    DispatchLayout::RANK_MAJOR>) is NOT mis-counted into the 'dispatch' bucket
    (and vice-versa) -- the dispatch/combine word always precedes the first '<'."""
    total_us = 0.0
    matched = False
    subs = tuple(s.lower() for s in substrs)
    for e in key_averages:
        name = e.key.split("<", 1)[0].lower()
        if any(s in name for s in subs) and int(e.count) > 0:
            total_us += float(e.self_device_time_total) / int(e.count)
            matched = True
    return total_us if matched else 0.0


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
# Latency dtype / combine helpers (ported from test_latency_multirank.py).
# ----------------------------------------------------------------------------
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
