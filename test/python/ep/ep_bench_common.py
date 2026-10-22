# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared helpers for the unified EP benchmark backends (bootstrap, input generation, quantization, and cross-rank validation)."""

from __future__ import annotations

import os
import torch

try:
    from mpi4py import MPI
except ImportError:

    class _FallbackMPI:
        SUM = object()
        MIN = object()
        MAX = object()

    MPI = _FallbackMPI()


class TorchComm:
    def __init__(self, group):
        self.torch_group = group

    def Get_rank(self):
        import torch.distributed as dist

        return dist.get_rank(self.torch_group)

    def Get_size(self):
        import torch.distributed as dist

        return dist.get_world_size(self.torch_group)

    def Barrier(self):
        import torch.distributed as dist

        dist.barrier(group=self.torch_group)

    def bcast(self, value, root=0):
        import torch.distributed as dist

        values = [value]
        dist.broadcast_object_list(values, src=root, group=self.torch_group)
        return values[0]

    def allreduce(self, value, op=MPI.SUM):
        import torch.distributed as dist

        if op is MPI.SUM:
            reduce_op = dist.ReduceOp.SUM
        elif op is MPI.MIN:
            reduce_op = dist.ReduceOp.MIN
        elif op is MPI.MAX:
            reduce_op = dist.ReduceOp.MAX
        else:
            raise ValueError(f"unsupported reduction op: {op}")
        tensor = torch.tensor(value, dtype=torch.float64)
        dist.all_reduce(tensor, op=reduce_op, group=self.torch_group)
        result = tensor.item()
        return type(value)(result)

    def gather(self, value, root=0):
        import torch.distributed as dist

        output = [None] * self.Get_size() if self.Get_rank() == root else None
        dist.gather_object(value, output, dst=root, group=self.torch_group)
        return output


# ----------------------------------------------------------------------------
# MPI bootstrap shared by both backends.
# ----------------------------------------------------------------------------
def init_mpi():
    if os.environ.get("EP_BOOTSTRAP") == "torch":
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo")
        group = dist.distributed_c10d._get_default_group()
        comm = TorchComm(group)
        return comm, comm.Get_rank(), comm.Get_size(), local_rank

    if not hasattr(MPI, "COMM_WORLD"):
        raise RuntimeError("mpi4py is required unless EP_BOOTSTRAP=torch")
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
    default_group = dist.distributed_c10d._get_default_group()
    group = (
        default_group
        if str(dist.get_backend(default_group)).lower() == "nccl"
        else dist.new_group(ranks=list(range(num_ranks)), backend="nccl")
    )
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
def _apply_mscclpp_cross_rack_topk_half(topk_idx, scores, num_experts, num_ranks, rank, local_world_size):
    if topk_idx.size(1) != 8:
        raise ValueError("cross-rack top-k half routing requires --num-topk 8")
    if num_ranks != 8 or local_world_size != 4:
        raise ValueError("cross-rack top-k half routing requires 8 ranks with 4 local ranks per node")
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    num_tokens, num_topk = topk_idx.shape
    half_topk = num_topk // 2
    num_local_experts = num_experts // num_ranks
    remote_rank_base = 0 if rank >= local_world_size else local_world_size
    remote_expert_begin = remote_rank_base * num_local_experts
    remote_expert_count = local_world_size * num_local_experts

    # First half: choose the top four experts from the remote node only. This keeps
    # four cross-rack top-k selections while preserving the natural random spread
    # across remote ranks instead of forcing every token to all four remote ranks.
    remote_scores = scores[:, remote_expert_begin : remote_expert_begin + remote_expert_count]
    remote_topk = torch.topk(remote_scores, half_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_idx[:, :half_topk] = remote_expert_begin + remote_topk
    topk_idx[:, half_topk:] = -1


def _apply_mscclpp_same_rack_topk(topk_idx, scores, num_experts, num_ranks, rank, local_world_size):
    if num_ranks != 8 or local_world_size != 4:
        raise ValueError("same-rack top-k routing requires 8 ranks with 4 local ranks per node")
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    num_topk = topk_idx.size(1)
    num_local_experts = num_experts // num_ranks
    local_rank_base = (rank // local_world_size) * local_world_size
    local_expert_begin = local_rank_base * num_local_experts
    local_expert_count = local_world_size * num_local_experts

    # Route every top-k selection to experts on the local node/rack only. This
    # removes cross-rack GPUNetIO payloads while preserving top-k=8 work per token.
    local_scores = scores[:, local_expert_begin : local_expert_begin + local_expert_count]
    local_topk = torch.topk(local_scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_idx[:, :] = local_expert_begin + local_topk


def _apply_mscclpp_cross_rack_token_half_all_remote(topk_idx, num_experts, num_ranks, rank, local_world_size):
    if topk_idx.size(0) != 128 or topk_idx.size(1) != 8:
        raise ValueError("cross-rack token-half all-remote routing requires -t 128 -k 8")
    if num_ranks != 8 or local_world_size != 4:
        raise ValueError("cross-rack token-half all-remote routing requires 8 ranks with 4 local ranks per node")
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    num_tokens, num_topk = topk_idx.shape
    half_tokens = num_tokens // 2
    half_topk = num_topk // 2
    num_local_experts = num_experts // num_ranks
    remote_rank_base = 0 if rank >= local_world_size else local_world_size
    token_offsets = torch.arange(half_tokens, device=topk_idx.device, dtype=torch.int64).view(-1, 1)
    remote_cols = torch.arange(half_topk, device=topk_idx.device, dtype=torch.int64).view(1, -1)
    remote_ranks = remote_rank_base + remote_cols

    # First 64 tokens: route one valid top-k entry to each of the four remote ranks,
    # forcing payload transfer to every GPU on the other rack. Remaining top-k lanes
    # and all later tokens are invalid, so no same-rack/local dispatch is launched.
    topk_idx[:, :] = -1
    topk_idx[:half_tokens, :half_topk] = (
        remote_ranks * num_local_experts + ((token_offsets + remote_cols) % num_local_experts)
    )


def _apply_mscclpp_rack0_gpu0_one_to_rack1(topk_idx, num_experts, num_ranks, rank, local_world_size):
    if num_ranks != 8 or local_world_size != 4:
        raise ValueError("rack0/GPU0 one-to-rack1 routing requires 8 ranks with 4 local ranks per node")
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    # Only rack0/GPU0 (global rank 0) sends. Rank 0 keeps the normal random
    # top-k routing over all experts/ranks. Other ranks publish invalid top-k
    # entries, so they do not send payload data in dispatch.
    if rank != 0:
        topk_idx[:, :] = -1


def _apply_mscclpp_rack0_gpu0_rack1_only(topk_idx, num_experts, num_ranks, rank, local_world_size):
    _apply_mscclpp_rack0_gpu0_one_to_rack1(topk_idx, num_experts, num_ranks, rank, local_world_size)
    # Rank 0 keeps only its random top-k entries whose expert lives on rack1
    # (ranks 4..7); entries targeting rack0 (ranks 0..3, incl. self) become
    # invalid so dispatch carries pure cross-rack traffic.
    # MSCCLPP_EP_ROUTE_RANK0_DST_RANKS (default "4,5,6,7") narrows the allowed
    # rack1 destinations further, e.g. "4,5".
    if rank == 0:
        num_local_experts = num_experts // num_ranks
        dst_rank = topk_idx // num_local_experts
        allowed = [
            int(r) for r in os.environ.get("MSCCLPP_EP_ROUTE_RANK0_DST_RANKS", "4,5,6,7").split(",") if r.strip()
        ]
        if any(r < local_world_size or r >= num_ranks for r in allowed):
            raise ValueError(f"rank0 dst ranks must be on rack1 [{local_world_size}, {num_ranks}): {allowed}")
        keep = torch.zeros_like(topk_idx, dtype=torch.bool)
        for r in allowed:
            keep |= dst_rank == r
        topk_idx.masked_fill_((topk_idx >= 0) & ~keep, -1)


def _apply_mscclpp_rank0_all_to_rack1(topk_idx, num_experts, num_ranks, rank, local_world_size):
    if num_ranks != 8 or local_world_size != 4:
        raise ValueError("rank0 all-to-rack1 routing requires 8 ranks with 4 local ranks per node")
    if topk_idx.size(1) < local_world_size:
        raise ValueError("rank0 all-to-rack1 routing requires --num-topk >= 4")
    # Only rank 0 sends: each of its first MSCCLPP_EP_ROUTE_RANK0_ACTIVE_TOKENS tokens (default all) goes to
    # every rank on rack1 (one top-k lane per rank 4..7), so each destination receives the same contiguous
    # rows. Everything else is invalid, so the only payload traffic is rank 0 -> ranks 4..7.
    num_tokens, num_topk = topk_idx.shape
    num_local_experts = num_experts // num_ranks
    active = int(os.environ.get("MSCCLPP_EP_ROUTE_RANK0_ACTIVE_TOKENS", str(num_tokens)))
    active = max(0, min(active, num_tokens))
    topk_idx[:, :] = -1
    if rank == 0 and active > 0:
        tokens = torch.arange(active, device=topk_idx.device, dtype=torch.int64).view(-1, 1)
        lanes = torch.arange(local_world_size, device=topk_idx.device, dtype=torch.int64).view(1, -1)
        topk_idx[:active, :local_world_size] = (local_world_size + lanes) * num_local_experts + (
            (tokens + lanes) % num_local_experts
        )


def make_inputs(
    num_tokens,
    hidden,
    num_topk,
    num_experts,
    rank,
    seed,
    *,
    route_mode=None,
    num_ranks=None,
    local_world_size=None,
):
    torch.manual_seed(seed + rank)
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    if route_mode == "mscclpp_cross_rack_topk_half":
        _apply_mscclpp_cross_rack_topk_half(
            topk_idx,
            scores,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode == "mscclpp_same_rack_topk":
        _apply_mscclpp_same_rack_topk(
            topk_idx,
            scores,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode == "mscclpp_cross_rack_token_half_all_remote":
        _apply_mscclpp_cross_rack_token_half_all_remote(
            topk_idx,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode == "mscclpp_rack0_gpu0_one_to_rack1":
        _apply_mscclpp_rack0_gpu0_one_to_rack1(
            topk_idx,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode == "mscclpp_rank0_all_to_rack1":
        _apply_mscclpp_rank0_all_to_rack1(
            topk_idx,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode == "mscclpp_rack0_gpu0_rack1_only":
        _apply_mscclpp_rack0_gpu0_rack1_only(
            topk_idx,
            num_experts,
            num_ranks if num_ranks is not None else 0,
            rank,
            local_world_size if local_world_size is not None else 0,
        )
    elif route_mode is not None:
        raise ValueError(f"unsupported route_mode: {route_mode}")
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    if route_mode in (
        "mscclpp_cross_rack_topk_half",
        "mscclpp_cross_rack_token_half_all_remote",
        "mscclpp_rack0_gpu0_one_to_rack1",
        "mscclpp_rack0_gpu0_rack1_only",
    ):
        topk_weights.masked_fill_(topk_idx < 0, 0.0)
    # ep_bench byte accounting: num_valid_selections = count(topk_idx >= 0); every
    # selection is valid in default mode (a full LL load), so this equals num_tokens * top_k.
    num_valid_selections = int((topk_idx >= 0).sum().item())
    return x, topk_idx, topk_weights, num_valid_selections


# ----------------------------------------------------------------------------
# Latency dtype / combine helpers (ported from test_latency_multirank.py).
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
