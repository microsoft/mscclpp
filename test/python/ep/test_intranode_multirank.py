# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Multi-rank direct-fabric HT functional validation for mscclpp_ep.

Launch with:
    torchrun --nproc_per_node=<N> test/python/ep/test_intranode_multirank.py

Tests that the high-level ``MoECommunicator`` succeeds across GPUs in one
detected GPU IPC/NVL fabric domain, including domains that span hosts, and that
a round-trip dispatch + combine preserves data.

Set ``MSCCLPP_EP_BENCH=1`` to also run a post-correctness benchmark pass
that times dispatch and combine **separately** with CUDA events and
reports per-phase latency (max across ranks) plus aggregate effective
NVLink bandwidth (sum across ranks). Override iteration counts with
``MSCCLPP_EP_BENCH_WARMUP`` / ``MSCCLPP_EP_BENCH_ITERS`` and the bench
problem size with ``MSCCLPP_EP_BENCH_TOKENS`` / ``_HIDDEN``.

This is a minimal adaptation of DeepEP's tests/test_intranode.py stripped
to exercise only the code paths we've ported.
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
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ.get('MASTER_ADDR','127.0.0.1')}:{os.environ.get('MASTER_PORT','29500')}",
        world_size=world_size,
        rank=rank,
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
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    ep_group = CommGroup(torch_group=group)

    # Small settings for functional check
    num_tokens = 128
    hidden = 1024
    num_topk = min(4, num_ranks)
    num_experts = num_ranks * 4

    torch.manual_seed(0xA1B2 + rank)

    # Build topk layout that maps each token to num_topk distinct ranks/experts
    scores = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda")

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert / rank meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1).values
        cnt = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True).indices
        tokens[:cnt] = torch.sort(tokens[:cnt]).values
        token_idx_in_rank[i][tokens[:cnt]] = torch.arange(cnt, dtype=torch.long, device="cuda")
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    # Token payload = rank id (cast to bf16) so we can check correctness
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * float(rank)

    moe = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.HIGH_THROUGHPUT,
        num_sms=int(os.environ.get("MSCCLPP_EP_NUM_SMS", "20")),
    )
    if rank == 0:
        print(
            f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk}",
            flush=True,
        )
    print(f"[rank {rank}] MoECommunicator created is_available={moe.is_available()}", flush=True)
    assert moe.is_available()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(num_ranks)))
    expected_internode = num_ranks > local_world_size
    assert moe.is_internode_available() == expected_internode
    assert moe.is_internode() == expected_internode

    dispatch_out, handle = moe.dispatch(
        x,
        topk_idx,
        topk_weights,
    )
    recv_x = dispatch_out.tokens
    dist.barrier(group=group)

    assert recv_x.dim() == 2 and recv_x.size(1) == hidden
    local_experts = num_experts // num_ranks
    assert dispatch_out.topk_ids is not None
    assert dispatch_out.topk_ids.shape == (recv_x.size(0), num_topk)
    assert ((dispatch_out.topk_ids >= -1) & (dispatch_out.topk_ids < local_experts)).all()
    assert dispatch_out.weights is not None
    assert dispatch_out.weights.shape == (recv_x.size(0), num_topk)
    assert torch.equal(
        dispatch_out.weights.masked_select(dispatch_out.topk_ids < 0),
        torch.zeros_like(dispatch_out.weights.masked_select(dispatch_out.topk_ids < 0)),
    )
    all_expert_counts = torch.empty((num_ranks, num_experts), dtype=num_tokens_per_expert.dtype, device="cuda")
    dist.all_gather_into_tensor(all_expert_counts, num_tokens_per_expert, group=group)
    expected_counts = all_expert_counts[:, rank * local_experts : (rank + 1) * local_experts].sum(dim=0).cpu().tolist()
    assert dispatch_out.layout.num_tokens_per_expert is not None
    actual_counts = [int(count) for count in dispatch_out.layout.num_tokens_per_expert]
    assert actual_counts == [int(count) for count in expected_counts]
    if rank == 0:
        print(f"[dispatch] OK (recv {recv_x.size(0)} tokens)", flush=True)

    # Use a distinct expert-output allocation so the direct TMA path cannot
    # accidentally read stale dispatch payloads from its receive pool.
    expert_out = recv_x + torch.ones_like(recv_x)
    context = handle.combine_context
    combined_x, combined_weights = moe._backend._runtime.combine(
        expert_out,
        context.recv_topk_weights,
        context.send_head,
    )

    # We dispatched rank-valued rows, then each destination added one.
    num_dst = is_token_in_rank.sum(dim=1).to(torch.float32)
    expected = num_dst * float(rank + 1)

    got = combined_x.float().mean(dim=1)
    diff = (got - expected).abs().max().item()
    max_exp = expected.abs().max().item()
    if rank == 0:
        print(f"[combine] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}", flush=True)
    assert diff < 1e-2, f"rank{rank}: combine mismatch max diff {diff}"
    assert combined_weights is not None
    assert torch.equal(combined_weights, topk_weights)

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

    # Rebuild inputs at bench size. The benchmark creates its own communicator
    # below so its internal buffers are sized for the benchmark shape.
    scores_b = torch.randn((bench_tokens, bench_num_experts), device="cuda", dtype=torch.float32).abs() + 1
    topk_idx_b = torch.topk(scores_b, bench_num_topk, dim=-1, sorted=False).indices
    topk_weights_b = torch.ones((bench_tokens, bench_num_topk), dtype=torch.float32, device="cuda")
    rank_idx_b = topk_idx_b // (bench_num_experts // num_ranks)
    rank_idx_b.masked_fill_(topk_idx_b == -1, -1)
    inplace_unique(rank_idx_b, num_ranks)
    num_tokens_per_expert_b = torch.zeros((bench_num_experts,), dtype=torch.int, device="cuda")
    for i in range(bench_num_experts):
        num_tokens_per_expert_b[i] = (topk_idx_b == i).sum()
    num_tokens_per_rank_b = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank_b = torch.full((num_ranks, bench_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank_b[i] = (rank_idx_b == i).sum()
        token_sel = (rank_idx_b == i).max(dim=-1).values
        cnt = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True).indices
        tokens[:cnt] = torch.sort(tokens[:cnt]).values
        token_idx_in_rank_b[i][tokens[:cnt]] = torch.arange(cnt, dtype=torch.long, device="cuda")
    token_idx_in_rank_b = token_idx_in_rank_b.T.contiguous().to(torch.int)
    is_token_in_rank_b = token_idx_in_rank_b >= 0
    x_b = torch.ones((bench_tokens, bench_hidden), dtype=torch.bfloat16, device="cuda") * float(rank)

    # Drive the benchmark through the public high-level API. The first
    # (uncached) dispatch records the routing layout on the returned handle;
    # subsequent dispatches reuse it via previous_handle, skipping notify's
    # host-side counter wait. This isolates the on-GPU dispatch-kernel cost
    # (NCCL-EP ep_bench convention).
    moe = ep.MoECommunicator(
        comm=ep_group,
        num_experts=bench_num_experts,
        hidden_size=bench_hidden,
        topk=bench_num_topk,
        max_tokens_per_rank=bench_tokens,
        mode=ep.MoEMode.HIGH_THROUGHPUT,
        num_sms=int(os.environ.get("MSCCLPP_EP_NUM_SMS", "20")),
    )
    assert moe.is_available()

    # One uncached dispatch to build the cached routing layout on the handle.
    _handle0 = moe.dispatch(x_b, topk_idx_b, topk_weights_b)[1]

    def _dispatch_cached():
        return moe.dispatch(x_b, topk_idx_b, topk_weights_b, previous_handle=_handle0)

    def _combine(dout):
        dispatch_out_, handle_ = dout
        moe.combine(dispatch_out_.tokens, handle_)

    # Warmup (full round-trip) using cached dispatch.
    for _ in range(warmup):
        _combine(_dispatch_cached())
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Time dispatch alone (cached mode -- skips notify_dispatch host wait).
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    dout = None
    for _ in range(iters):
        dout = _dispatch_cached()
    end_ev.record()
    torch.cuda.synchronize()
    disp_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Time combine alone (reusing the same dispatch output each iter).
    dist.barrier(group=group)
    start_ev.record()
    for _ in range(iters):
        _combine(dout)
    end_ev.record()
    torch.cuda.synchronize()
    comb_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Per-rank "send bytes" matches NCCL-EP's `ep_bench` accounting:
    # bench_tokens * hidden * sizeof(bf16). Each rank ships its `bench_tokens`
    # input rows out (some replicated to multiple peers); NCCL-EP normalizes by
    # the input footprint, not by the recv-side fan-out. We use the same
    # convention here so `per_rank_bw` is directly comparable across stacks.
    bytes_one_way = bench_tokens * bench_hidden * x_b.element_size()

    # Send side follows NCCL-EP: count unique (token, dst_node) pairs. With a
    # single node every selected destination collapses to that node, so a
    # token with at least one valid expert contributes exactly one to
    # `total_send_tokens`. Recv side counts unique (src_rank, token) pairs
    # landing on this rank.
    bytes_per_token = bench_hidden * x_b.element_size()
    total_send_tokens_local = int(is_token_in_rank_b.any(dim=1).sum().item())
    _send_row = num_tokens_per_rank_b.to(torch.int64).contiguous()
    _gathered = torch.empty(num_ranks * num_ranks, dtype=torch.int64, device="cuda")
    dist.all_gather_into_tensor(_gathered, _send_row, group=group)
    recv_from_src = _gathered.view(num_ranks, num_ranks)[:, rank].contiguous()
    total_recv_tokens_local = int(recv_from_src.sum().item())

    # Average per-rank token counts across ranks (matches NCCL-EP `Byte counts (per rank avg)`).
    counts_t = torch.tensor(
        [total_send_tokens_local, total_recv_tokens_local],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(counts_t, op=dist.ReduceOp.SUM, group=group)
    counts_avg = (counts_t / num_ranks).tolist()
    total_send_avg, total_recv_avg = counts_avg
    total_send_bytes = total_send_avg * bytes_per_token
    total_recv_bytes = total_recv_avg * bytes_per_token

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
    disp_t_s = disp_avg_us * 1e-6
    comb_t_s = comb_avg_us * 1e-6
    d_send_total_bw = total_send_bytes / disp_t_s / 1e9
    d_recv_total_bw = total_recv_bytes / disp_t_s / 1e9
    c_send_total_bw = total_recv_bytes / comb_t_s / 1e9  # combine sends back what dispatch received
    c_recv_total_bw = total_send_bytes / comb_t_s / 1e9  # combine receives back what dispatch sent
    if rank == 0:
        print(
            f"[bench intranode HT] tokens={bench_tokens} hidden={bench_hidden} "
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
            f"            send={d_send_total_bw:.2f} GB/s  recv={d_recv_total_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"  combine : avg={comb_avg_us:.1f}us min={comb_min_t.item():.1f}us max={comb_max_t.item():.1f}us  "
            f"per_rank_bw={comb_bw_per_rank:.2f} GB/s  "
            f"agg_bw={comb_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
            flush=True,
        )
        print(
            f"            send={c_send_total_bw:.2f} GB/s  recv={c_recv_total_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"  byte counts (per rank avg): "
            f"total_send={total_send_bytes/1e6:.2f} MB ({total_send_avg:.0f} tok)  "
            f"total_recv={total_recv_bytes/1e6:.2f} MB ({total_recv_avg:.0f} tok)",
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
        # TCPStore server (rank 0) exits, then destroy the PG. Avoids noisy
        # "recvValue failed / Connection was likely closed" stack traces from
        # ProcessGroupNCCL's HeartbeatMonitor.
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
