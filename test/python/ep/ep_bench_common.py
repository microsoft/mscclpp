# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared helpers for the unified EP benchmark backends (bootstrap, input generation, quantization, and cross-rank validation)."""

from __future__ import annotations

import os
import torch
from mpi4py import MPI


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


def _ensure_torch_dist(comm, rank, num_ranks):
    """Lazily initialize the default torch.distributed NCCL group alongside MPI
    (idempotent). MPI supplies the rendezvous (rank-0 IP + port broadcast). Both
    the kineto GPU barrier and the DeepEP backend reuse this single group.
    Returns the world ProcessGroup."""
    import torch.distributed as dist

    if not dist.is_initialized():
        addr = None
        if rank == 0:
            addr = os.environ.get("MASTER_ADDR")
            if not addr:
                import socket

                _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    _s.connect(("10.255.255.255", 1))  # no packets sent; picks outbound iface IP
                    addr = _s.getsockname()[0]
                finally:
                    _s.close()
        addr = comm.bcast(addr, root=0)
        port = comm.bcast(int(os.environ.get("MASTER_PORT", "29700")), root=0)
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(num_ranks)
        os.environ["RANK"] = str(rank)
        import datetime as _dt

        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=num_ranks, rank=rank, timeout=_dt.timedelta(seconds=120)
        )
    return dist.distributed_c10d._get_default_group()


def _init_torch_nccl(comm, rank, num_ranks, local_rank):
    """Return a zero-arg GPU-side barrier (torch NCCL all_reduce) for the kineto
    timing loop -- aligns ranks on-device, much tighter than an MPI host barrier."""
    import torch.distributed as dist

    _ensure_torch_dist(comm, rank, num_ranks)
    _sync = torch.ones(1, dtype=torch.float, device="cuda")

    def _barrier():
        dist.all_reduce(_sync)

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
