# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Multi-rank internode (HT) functional validation for mscclpp_ep.

Launch on each node with (example: 2 nodes x 8 GPUs = 16 ranks):

    # on master (NODE_RANK=0):
    MASTER_ADDR=<master_ip> MASTER_PORT=29600 NODE_RANK=0 \
        torchrun --nnodes=2 --nproc_per_node=8 \
            --rdzv-backend=c10d --rdzv-endpoint=<master_ip>:29600 \
            test/python/ext/ep/test_internode_multirank.py

    # on worker (NODE_RANK=1):
    MASTER_ADDR=<master_ip> MASTER_PORT=29600 NODE_RANK=1 \
        torchrun --nnodes=2 --nproc_per_node=8 \
            --rdzv-backend=c10d --rdzv-endpoint=<master_ip>:29600 \
            test/python/ext/ep/test_internode_multirank.py

Round-trip dispatch + combine using internode HT kernels across nodes.

Set ``MSCCLPP_EP_BENCH=1`` to also run a post-correctness benchmark pass
that times dispatch and combine **separately** with CUDA events. Reports
per-phase latency (max across ranks) plus aggregate effective bandwidth
(sum across ranks). Override iteration counts with
``MSCCLPP_EP_BENCH_WARMUP`` / ``MSCCLPP_EP_BENCH_ITERS`` and the bench
problem size with ``MSCCLPP_EP_BENCH_TOKENS`` / ``_HIDDEN``.
"""

from __future__ import annotations

import os
import sys

# Disable ProcessGroupNCCL's HeartbeatMonitor before importing torch.distributed.
# It runs in a background thread polling the TCPStore; under mpirun, rank 0
# (the store server) can exit before non-zero ranks finish teardown, producing
# noisy 'recvValue failed / Connection was likely closed' stack traces.
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, device_id=torch.device(f"cuda:{local_rank}")
    )
    return rank, world_size, local_rank, dist.new_group(list(range(world_size)))


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def main():
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp.ext import ep

    NUM_MAX_NVL_PEERS = 8
    assert (
        num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS
    ), f"expected >1 node with 8 GPUs each, got num_ranks={num_ranks}"
    num_nodes = num_ranks // NUM_MAX_NVL_PEERS
    num_local_ranks = NUM_MAX_NVL_PEERS

    # Small settings for functional check
    num_tokens = 128
    hidden = 1024
    num_topk = min(4, num_ranks)
    num_experts = num_ranks * 4  # multiple of num_ranks

    torch.manual_seed(0xA1B2 + rank)

    scores = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda")

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1).values
        cnt = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True).indices
        tokens[:cnt] = torch.sort(tokens[:cnt]).values
        token_idx_in_rank[i][tokens[:cnt]] = torch.arange(cnt, dtype=torch.long, device="cuda")
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * float(rank)

    # Buffer config for internode HT: needs num_rdma_bytes > 0. Size buffers
    # using max(hidden, bench_hidden) so the optional bench phase fits.
    cfg = ep.Config(20, 8, 256, 16, 128)
    _bench_on = os.environ.get("MSCCLPP_EP_BENCH", "0") == "1"
    _buf_hidden = max(hidden, int(os.environ.get("MSCCLPP_EP_BENCH_HIDDEN", "0"))) if _bench_on else hidden
    num_nvl_bytes = cfg.get_nvl_buffer_size_hint(_buf_hidden * x.element_size(), num_ranks)
    num_rdma_bytes = cfg.get_rdma_buffer_size_hint(_buf_hidden * x.element_size(), num_ranks)
    if rank == 0:
        print(
            f"[cfg] num_nodes={num_nodes} num_ranks={num_ranks} num_tokens={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} num_topk={num_topk} "
            f"num_nvl_bytes={num_nvl_bytes} num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    print(f"[rank {rank}] creating Buffer", flush=True)
    buf = ep.Buffer(group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes, low_latency_mode=False)
    print(
        f"[rank {rank}] Buffer created is_available={buf.is_available()} "
        f"is_internode={buf.is_internode_available()}",
        flush=True,
    )
    assert buf.is_available() and buf.is_internode_available()

    ref_rank, ref_rdma_rank, ref_exp, ref_in_rank, _ = buf.runtime.get_dispatch_layout(
        topk_idx, num_experts, None, False, False
    )
    assert torch.allclose(ref_rank, num_tokens_per_rank)
    assert torch.allclose(ref_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_exp, num_tokens_per_expert)
    assert torch.allclose(ref_in_rank, is_token_in_rank)
    if rank == 0:
        print("[layout] OK", flush=True)
    dist.barrier(group=group)

    # internode_dispatch signature (non-cached mode):
    # (x, x_scales, topk_idx, topk_weights,
    #  num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
    #  cached_num_recv_tokens=0, cached_num_rdma_recv_tokens=0,
    #  cached_rdma_channel_prefix_matrix=None, cached_recv_rdma_rank_prefix_sum=None,
    #  cached_gbl_channel_prefix_matrix=None, cached_recv_gbl_rank_prefix_sum=None,
    #  expert_alignment, config, previous_event, async, allocate_on_comm_stream)
    (
        recv_x,
        recv_x_scales,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        rdma_channel_prefix_matrix,
        gbl_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        recv_src_meta,
        send_rdma_head,
        send_nvl_head,
        _event,
    ) = buf.runtime.internode_dispatch(
        x,
        None,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        0,
        0,
        None,
        None,
        None,
        None,
        1,
        cfg,
        None,
        False,
        False,
    )
    dist.barrier(group=group)

    # Validate recv buffer: for each source rank i, the block carries value i.
    assert recv_x.dim() == 2 and recv_x.size(1) == hidden
    start = 0
    for src in range(num_ranks):
        end = recv_gbl_rank_prefix_sum[src].item()
        block = recv_x[start:end]
        if block.numel():
            lo = block.float().amin().item()
            hi = block.float().amax().item()
            assert (
                abs(lo - src) < 1e-3 and abs(hi - src) < 1e-3
            ), f"rank{rank}: block from src={src} has range=[{lo}, {hi}], expected {src}"
        start = end
    if rank == 0:
        print(f"[dispatch] OK (recv {recv_x.size(0)} tokens)", flush=True)

    # XXX: forcing a device+group sync here is currently required for combine
    # to see consistent dispatch outputs. Without this both send_nvl_head and
    # the various *_channel_prefix_matrix tensors can still be in flight on
    # the comm stream when combine launches, producing a deadlock inside the
    # combine forwarder (NVL check never advances). Investigate proper
    # stream-dependency hand-off in Buffer::internode_dispatch.
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # internode_combine signature:
    # (x, topk_weights,
    #  src_meta, is_combined_token_in_rank,
    #  rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
    #  combined_rdma_head, combined_nvl_head, config, previous_event, async, allocate_on_comm_stream)
    # NOTE: combine goes in the reverse direction of dispatch, so the prefix
    # matrices passed here must be the RECEIVER-side ones returned by dispatch
    # (`recv_rdma_channel_prefix_matrix`, `recv_rdma_rank_prefix_sum`,
    # `recv_gbl_channel_prefix_matrix`) — not the sender-side ones.
    combined_x, combined_topk_weights, _ = buf.runtime.internode_combine(
        recv_x,
        recv_topk_weights,
        recv_src_meta,
        is_token_in_rank,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        cfg,
        None,
        False,
        False,
    )

    num_dst = is_token_in_rank.sum(dim=1).to(torch.float32)
    expected = num_dst * float(rank)
    got = combined_x.float().mean(dim=1)
    diff = (got - expected).abs().max().item()
    max_exp = expected.abs().max().item()
    print(f"[combine r{rank}] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}", flush=True)
    assert diff < 1e-2, f"rank{rank}: combine mismatch max diff {diff}"

    dist.barrier(group=group)
    if rank == 0:
        print("PASS", flush=True)

    # ------------------------------------------------------------------
    # Optional benchmark (enable with MSCCLPP_EP_BENCH=1).
    # ------------------------------------------------------------------
    if os.environ.get("MSCCLPP_EP_BENCH", "0") != "1":
        return

    warmup = int(os.environ.get("MSCCLPP_EP_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("MSCCLPP_EP_BENCH_ITERS", "20"))
    bench_tokens = int(os.environ.get("MSCCLPP_EP_BENCH_TOKENS", "4096"))
    bench_hidden = int(os.environ.get("MSCCLPP_EP_BENCH_HIDDEN", "7168"))
    # Allow overriding num_experts / num_topk for the bench phase to match
    # NCCL-EP's `ep_bench -a ht` defaults (256 experts, top-8). The functional
    # check above still uses the smaller (num_experts=num_ranks*4, topk=4)
    # configuration.
    bench_num_experts = int(os.environ.get("MSCCLPP_EP_BENCH_EXPERTS", str(num_experts)))
    bench_num_topk = int(os.environ.get("MSCCLPP_EP_BENCH_TOPK", str(num_topk)))
    if bench_num_experts % num_ranks != 0:
        if rank == 0:
            print(
                f"[bench] skip: num_experts={bench_num_experts} not divisible " f"by num_ranks={num_ranks}", flush=True
            )
        return
    if bench_num_topk > bench_num_experts:
        if rank == 0:
            print(f"[bench] skip: topk={bench_num_topk} > experts={bench_num_experts}", flush=True)
        return

    # Respect the Buffer's pre-sized num_nvl_bytes / num_rdma_bytes budget.
    per_peer_nvl = num_nvl_bytes // max(1, num_ranks)
    per_peer_rdma = num_rdma_bytes // max(1, num_ranks)
    if bench_hidden * x.element_size() > min(per_peer_nvl, per_peer_rdma):
        if rank == 0:
            print(
                f"[bench] skip: hidden={bench_hidden} bytes/row={bench_hidden * x.element_size()} "
                f">= min(per-peer NVL {per_peer_nvl}, RDMA {per_peer_rdma}). "
                f"Rerun with a larger Buffer or smaller hidden.",
                flush=True,
            )
        return

    scores_b = torch.randn((bench_tokens, bench_num_experts), device="cuda", dtype=torch.float32).abs() + 1
    topk_idx_b = torch.topk(scores_b, bench_num_topk, dim=-1, sorted=False).indices
    topk_weights_b = torch.ones((bench_tokens, bench_num_topk), dtype=torch.float32, device="cuda")
    rank_idx_b = topk_idx_b // (bench_num_experts // num_ranks)
    rank_idx_b.masked_fill_(topk_idx_b == -1, -1)
    inplace_unique(rank_idx_b, num_ranks)
    rdma_rank_idx_b = rank_idx_b // num_local_ranks
    rdma_rank_idx_b.masked_fill_(rank_idx_b == -1, -1)
    inplace_unique(rdma_rank_idx_b, num_nodes)

    num_tokens_per_expert_b = torch.zeros((bench_num_experts,), dtype=torch.int, device="cuda")
    for i in range(bench_num_experts):
        num_tokens_per_expert_b[i] = (topk_idx_b == i).sum()
    num_tokens_per_rank_b = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank_b = torch.empty((num_nodes,), dtype=torch.int, device="cuda")
    token_idx_in_rank_b = torch.full((num_ranks, bench_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank_b[i] = (rank_idx_b == i).sum()
        token_sel = (rank_idx_b == i).max(dim=-1).values
        cnt = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True).indices
        tokens[:cnt] = torch.sort(tokens[:cnt]).values
        token_idx_in_rank_b[i][tokens[:cnt]] = torch.arange(cnt, dtype=torch.long, device="cuda")
    for i in range(num_nodes):
        num_tokens_per_rdma_rank_b[i] = (rdma_rank_idx_b == i).sum()
    token_idx_in_rank_b = token_idx_in_rank_b.T.contiguous().to(torch.int)
    is_token_in_rank_b = token_idx_in_rank_b >= 0
    x_b = torch.ones((bench_tokens, bench_hidden), dtype=torch.bfloat16, device="cuda") * float(rank)

    def _dispatch():
        return buf.runtime.internode_dispatch(
            x_b,
            None,
            topk_idx_b,
            topk_weights_b,
            num_tokens_per_rank_b,
            num_tokens_per_rdma_rank_b,
            is_token_in_rank_b,
            num_tokens_per_expert_b,
            0,
            0,
            None,
            None,
            None,
            None,
            1,
            cfg,
            None,
            False,
            False,
        )

    def _combine(dout):
        rx, _rxs, _rti, rtw, _lst, _rpm, _gpm, rrcpm, rrps, rgpm, _rgps, rsm, sh_rdma, sh_nvl, _ev = dout
        buf.runtime.internode_combine(
            rx,
            rtw,
            rsm,
            is_token_in_rank_b,
            rrcpm,
            rrps,
            rgpm,
            sh_rdma,
            sh_nvl,
            cfg,
            None,
            False,
            False,
        )

    # Warmup (full round-trip with the sync/barrier guard between phases,
    # matching the correctness-path invariant).
    for _ in range(warmup):
        dout = _dispatch()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        _combine(dout)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Time dispatch alone.
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    dout = None
    for _ in range(iters):
        dout = _dispatch()
    end_ev.record()
    torch.cuda.synchronize()
    disp_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Required guard before combine sees the dispatch outputs (see correctness
    # path's XXX note). Not included in either phase's timing.
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Time combine alone (reusing the same dispatch output each iter).
    start_ev.record()
    for _ in range(iters):
        _combine(dout)
    end_ev.record()
    torch.cuda.synchronize()
    comb_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Per-rank "send bytes" matches NCCL-EP's `ep_bench` accounting (`RDMA_send`):
    # bench_tokens * hidden * sizeof(bf16). Each rank ships its `bench_tokens`
    # input rows out (some replicated to multiple peers); NCCL-EP normalizes by
    # the input footprint, not by the recv-side fan-out. We use the same
    # convention here so `per_rank_bw` is directly comparable across stacks.
    bytes_one_way = bench_tokens * bench_hidden * x_b.element_size()

    # NCCL-EP `ep_bench` six-metric breakdown.
    # Send-side accounting follows NCCL-EP: count unique (token, dst_node) pairs.
    # `num_tokens_per_rdma_rank_b[n]` is exactly that count for node `n`.
    # Recv-side accounting: each rank reports `num_tokens_per_rank_b[r]`
    # (tokens it sends to dst rank `r`); an `all_to_all_single` lets every
    # rank read how many tokens each source rank sent to it.
    bytes_per_token = bench_hidden * x_b.element_size()
    local_node = rank // num_local_ranks
    nodes_unique = num_tokens_per_rdma_rank_b.to(torch.int64)
    total_send_tokens_local = int(nodes_unique.sum().item())
    nvl_send_tokens_local = int(nodes_unique[local_node].item())
    rdma_send_tokens_local = total_send_tokens_local - nvl_send_tokens_local
    recv_from_src = torch.empty(num_ranks, dtype=torch.int64, device="cuda")
    dist.all_to_all_single(
        recv_from_src,
        num_tokens_per_rank_b.to(torch.int64),
        group=group,
    )
    src_node = torch.arange(num_ranks, device="cuda") // num_local_ranks
    remote_mask = (src_node != local_node).to(torch.int64)
    total_recv_tokens_local = int(recv_from_src.sum().item())
    rdma_recv_tokens_local = int((recv_from_src * remote_mask).sum().item())

    # Average per-rank token counts across ranks (matches NCCL-EP `Byte counts (per rank avg)`).
    counts_t = torch.tensor(
        [total_send_tokens_local, rdma_send_tokens_local, total_recv_tokens_local, rdma_recv_tokens_local],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(counts_t, op=dist.ReduceOp.SUM, group=group)
    counts_avg = (counts_t / num_ranks).tolist()
    total_send_avg, rdma_send_avg, total_recv_avg, rdma_recv_avg = counts_avg
    total_send_bytes = total_send_avg * bytes_per_token
    rdma_send_bytes = rdma_send_avg * bytes_per_token
    total_recv_bytes = total_recv_avg * bytes_per_token
    rdma_recv_bytes = rdma_recv_avg * bytes_per_token
    nvl_send_bytes = total_send_bytes - rdma_send_bytes
    nvl_recv_bytes = total_recv_bytes - rdma_recv_bytes

    # Reduce timings: report min/avg/max and base BW on AVG to match NCCL-EP's
    # `ep_bench.cu` convention.
    disp_min_t = torch.tensor([disp_us], dtype=torch.float64, device="cuda")
    disp_avg_t = torch.tensor([disp_us], dtype=torch.float64, device="cuda")
    disp_max_t = torch.tensor([disp_us], dtype=torch.float64, device="cuda")
    comb_min_t = torch.tensor([comb_us], dtype=torch.float64, device="cuda")
    comb_avg_t = torch.tensor([comb_us], dtype=torch.float64, device="cuda")
    comb_max_t = torch.tensor([comb_us], dtype=torch.float64, device="cuda")
    dist.all_reduce(disp_min_t, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(disp_avg_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(disp_max_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(comb_min_t, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(comb_avg_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(comb_max_t, op=dist.ReduceOp.MAX, group=group)
    disp_avg_us = disp_avg_t.item() / num_ranks
    comb_avg_us = comb_avg_t.item() / num_ranks
    disp_bw_per_rank = bytes_one_way / (disp_avg_us * 1e-6) / 1e9
    comb_bw_per_rank = bytes_one_way / (comb_avg_us * 1e-6) / 1e9
    # Six-metric BW (NCCL-EP convention). Combine reverses send<->recv:
    # in combine, this rank pushes back what it received in dispatch.
    disp_t_s = disp_avg_us * 1e-6
    comb_t_s = comb_avg_us * 1e-6
    d_send_total_bw = total_send_bytes / disp_t_s / 1e9
    d_send_nvl_bw = nvl_send_bytes / disp_t_s / 1e9
    d_send_rdma_bw = rdma_send_bytes / disp_t_s / 1e9
    d_recv_total_bw = total_recv_bytes / disp_t_s / 1e9
    d_recv_nvl_bw = nvl_recv_bytes / disp_t_s / 1e9
    d_recv_rdma_bw = rdma_recv_bytes / disp_t_s / 1e9
    c_send_total_bw = total_recv_bytes / comb_t_s / 1e9
    c_send_nvl_bw = nvl_recv_bytes / comb_t_s / 1e9
    c_send_rdma_bw = rdma_recv_bytes / comb_t_s / 1e9
    c_recv_total_bw = total_send_bytes / comb_t_s / 1e9
    c_recv_nvl_bw = nvl_send_bytes / comb_t_s / 1e9
    c_recv_rdma_bw = rdma_send_bytes / comb_t_s / 1e9
    if rank == 0:
        print(
            f"[bench internode HT] nodes={num_nodes} num_ranks={num_ranks} "
            f"tokens={bench_tokens} hidden={bench_hidden} "
            f"experts={bench_num_experts} topk={bench_num_topk} "
            f"warmup={warmup} iters={iters}",
            flush=True,
        )
        print(
            f"  dispatch: avg={disp_avg_us:.1f}us min={disp_min_t.item():.1f}us max={disp_max_t.item():.1f}us  "
            f"per_rank_bw={disp_bw_per_rank:.2f} GB/s  "
            f"agg_bw={disp_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
            flush=True,
        )
        print(
            f"            send: total={d_send_total_bw:.2f}  nvl={d_send_nvl_bw:.2f}  rdma={d_send_rdma_bw:.2f} GB/s  "
            f"recv: total={d_recv_total_bw:.2f}  nvl={d_recv_nvl_bw:.2f}  rdma={d_recv_rdma_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"  combine : avg={comb_avg_us:.1f}us min={comb_min_t.item():.1f}us max={comb_max_t.item():.1f}us  "
            f"per_rank_bw={comb_bw_per_rank:.2f} GB/s  "
            f"agg_bw={comb_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
            flush=True,
        )
        print(
            f"            send: total={c_send_total_bw:.2f}  nvl={c_send_nvl_bw:.2f}  rdma={c_send_rdma_bw:.2f} GB/s  "
            f"recv: total={c_recv_total_bw:.2f}  nvl={c_recv_nvl_bw:.2f}  rdma={c_recv_rdma_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"  byte counts (per rank avg): "
            f"total_send={total_send_bytes/1e6:.2f} MB ({total_send_avg:.0f} tok)  "
            f"rdma_send={rdma_send_bytes/1e6:.2f} MB ({rdma_send_avg:.0f} tok)  "
            f"total_recv={total_recv_bytes/1e6:.2f} MB ({total_recv_avg:.0f} tok)  "
            f"rdma_recv={rdma_recv_bytes/1e6:.2f} MB ({rdma_recv_avg:.0f} tok)",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ordered shutdown: barrier so every rank reaches teardown before the
        # TCPStore server (rank 0) exits, then destroy the PG. Without this,
        # ProcessGroupNCCL's HeartbeatMonitor on non-zero ranks logs noisy
        # "recvValue failed / Connection was likely closed" stack traces.
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
