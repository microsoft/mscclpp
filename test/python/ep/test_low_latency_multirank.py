# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Multi-rank low-latency functional test for mscclpp_ep.

Launch with (intra-node, 8 GPUs):
    torchrun --nproc_per_node=8 test/python/ep/test_low_latency_multirank.py \
        --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256
    # Optional CUDA graph smoke/benchmark:
    torchrun --nproc_per_node=8 test/python/ep/test_low_latency_multirank.py \
        --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 \
        --cuda-graph --bench

Launch with (2 nodes, 1 GPU per node -- DeepEP's recommended LL topology):
    # node 0:
    MASTER_ADDR=<master> MASTER_PORT=29600 NODE_RANK=0 \
        torchrun --nnodes=2 --nproc_per_node=1 --rdzv-backend=c10d \
            --rdzv-endpoint=<master>:29600 test/python/ep/test_low_latency_multirank.py
    # node 1:
    MASTER_ADDR=<master> MASTER_PORT=29600 NODE_RANK=1 \
        torchrun --nnodes=2 --nproc_per_node=1 --rdzv-backend=c10d \
            --rdzv-endpoint=<master>:29600 test/python/ep/test_low_latency_multirank.py

Exercises the optimized BF16 LL dispatch plus the default combine path on a
single node. The experimental optimized combine performs rank-local partial
reduction, TMA send, and source-rank reduction. The minimal correctness check:
  - dispatch: per-expert received token counts agree with an all-gathered
    reference computed from topk_idx across all ranks;
  - combine: the reconstructed x matches the analytical sum
    ``x * sum(topk_weights, masked by topk_idx == -1)``.

Adapted from DeepEP/tests/test_low_latency.py stripped to the bare checks
we need for an LL port smoke test. BF16-only (no FP8 check).
"""

from __future__ import annotations

import argparse
import os
import random

# Disable ProcessGroupNCCL's HeartbeatMonitor before importing torch.distributed.
# It runs in a background thread polling the TCPStore; under mpirun, rank 0
# (the store server) can exit before non-zero ranks finish teardown, producing
# noisy 'recvValue failed / Connection was likely closed' stack traces.
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="MSCCL++ EP low-latency multi-rank correctness/benchmark test")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        choices=(4096, 7168, 8192, 9216),
        help="BF16 hidden size compiled into the optimized low-latency kernels",
    )
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-active-ranks", type=int, default=0, help="Limit routing to the first N ranks")
    parser.add_argument("--no-weights", action="store_true", help="Use implicit unit routing weights")
    parser.add_argument("--dispatch-num-sms", type=int, default=64)
    parser.add_argument("--combine-num-sms", type=int, default=64)
    parser.add_argument(
        "--optimized-combine-mode",
        choices=("disabled", "rank_local_reduce", "direct_send"),
        default="disabled",
    )
    parser.add_argument("--bench", action="store_true", help="Run dispatch/combine benchmark after correctness")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture dispatch/combine in CUDA graphs; correctness captures both in one graph",
    )
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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


def main():
    args = parse_args()
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    ep_group = CommGroup(torch_group=group)

    # Shrink the "bf16 precision" anchor to keep values small.
    rank_offset = 128
    assert num_ranks - rank_offset < 257, "too many ranks for bf16 precision anchor"

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    combine_mode = {
        "disabled": ep.OptimizedCombineMode.DISABLED,
        "rank_local_reduce": ep.OptimizedCombineMode.RANK_LOCAL_REDUCE,
        "direct_send": ep.OptimizedCombineMode.DIRECT_SEND,
    }[args.optimized_combine_mode]

    torch.manual_seed(0xB3C4 + rank)
    random.seed(0xB3C4 + rank)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    # Encode the per-token index into the last 128 elements so the receiver
    # can verify which source token it is looking at.
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    if args.num_active_ranks:
        assert num_topk <= args.num_active_ranks * num_local_experts
        scores[:, args.num_active_ranks * num_local_experts :] = float("-inf")
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = (
        None if args.no_weights else torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    )

    # Randomly mask some positions
    for _ in range(min(10, num_tokens)):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        num_rdma_qps_per_rank=max(1, num_experts // num_ranks),
        low_latency_dispatch_num_sms=args.dispatch_num_sms,
        low_latency_combine_num_sms=args.combine_num_sms,
        low_latency_combine_mode=combine_mode,
    )
    if rank == 0:
        print(
            f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk}",
            flush=True,
        )
    print(
        f"[rank {rank}] MoECommunicator created is_available={moe_comm.is_available()} "
        f"is_internode={moe_comm.is_internode_available()}",
        flush=True,
    )
    assert moe_comm.is_available()

    dist.barrier(group=group)
    torch.cuda.synchronize()
    print(f"[rank {rank}] pre-dispatch", flush=True)

    # --- Dispatch ---
    dispatch_output_buffer = torch.empty(
        (num_local_experts, num_ranks * num_tokens, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    dispatch_out, handle = moe_comm.dispatch(
        x,
        topk_idx,
        topk_weights,
        output_buffer=dispatch_output_buffer,
    )
    packed_recv_x = dispatch_out.tokens
    assert dispatch_out.layout.num_tokens_per_expert is not None
    packed_recv_count = dispatch_out.layout.num_tokens_per_expert
    packed_recv_layout_range = handle.combine_context.layout_range
    torch.cuda.synchronize()
    print(f"[rank {rank}] post-dispatch", flush=True)
    # packed_recv_x: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]
    # packed_recv_count: [num_local_experts] int32

    # Reference: gather all ranks' topk_idx and count expected tokens per expert.
    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device="cuda")
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    int_mask = (1 << 32) - 1
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        recv_count = int(packed_recv_count[i].item())
        expected_count = int((all_topk_idx == expert_id).sum().item())
        recv_layout_range = packed_recv_layout_range[i]
        layout_sum = int((recv_layout_range & int_mask).sum().item())
        assert (
            recv_count == expected_count
        ), f"rank{rank} expert{expert_id}: recv_count={recv_count} != expected={expected_count}"
        assert (
            layout_sum == recv_count
        ), f"rank{rank} expert{expert_id}: layout range sum {layout_sum} != recv_count {recv_count}"

        if recv_count:
            recv_x = packed_recv_x[i, :recv_count]
            # All columns except the last 128 should share the value (src_rank - rank_offset)
            recv_x_lo = recv_x[:, :-128]
            amin = recv_x_lo.amin(dim=-1)
            amax = recv_x_lo.amax(dim=-1)
            assert torch.equal(amin, amax), f"rank{rank} expert{expert_id}: non-uniform recv block"

    if rank == 0:
        print(f"[dispatch] OK (ranks={num_ranks})", flush=True)

    # --- Combine ---
    # Simulate the downstream GEMM output = identity (bf16 copy) so combine
    # returns sum(x * weight) across experts.
    simulated_gemm_x = packed_recv_x.clone()
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    combined_x = moe_comm.combine(simulated_gemm_x, handle, out=out)

    # Analytical expected: each token i, weighted sum over topk entries that
    # are not -1. Accumulate in the same top-k order as the kernel; multiplying
    # by the pre-summed weights can differ by one BF16 ULP for large token IDs.
    expected_f = torch.zeros_like(x, dtype=torch.float32)
    x_f = x.float()
    for j in range(num_topk):
        weight_j = (
            (topk_idx[:, j] != -1).float()
            if topk_weights is None
            else topk_weights[:, j].masked_fill(topk_idx[:, j] == -1, 0.0)
        ).view(-1, 1)
        expected_f += x_f * weight_j
    expected = expected_f.to(torch.bfloat16)
    diff = (combined_x.float() - expected.float()).abs().max().item()
    max_exp = expected.float().abs().max().item()
    print(
        f"[combine r{rank}] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}",
        flush=True,
    )
    assert torch.isnan(combined_x).any().item() is False
    combine_tolerance = 8.0 if combine_mode == ep.OptimizedCombineMode.RANK_LOCAL_REDUCE else 1e-2
    assert diff <= combine_tolerance, f"rank{rank}: LL combine mismatch diff={diff}"

    dist.barrier(group=group)
    if rank == 0:
        print("PASS", flush=True)

    def _graph_capture(dispatch_buffer, combine_out):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        with torch.cuda.graph(graph):
            graph_dout = moe_comm.dispatch(
                x,
                topk_idx,
                topk_weights,
                output_buffer=dispatch_buffer,
            )
            graph_combined_x = moe_comm.combine(graph_dout[0].tokens, graph_dout[1], out=combine_out)
        return graph, graph_dout, graph_combined_x

    def _run_cuda_graph_correctness():
        graph_dispatch_output_buffer = torch.empty_like(dispatch_output_buffer)
        graph_out = torch.empty_like(out)
        graph, _, graph_combined_x = _graph_capture(graph_dispatch_output_buffer, graph_out)
        graph.replay()
        torch.cuda.synchronize()

        graph_diff = (graph_combined_x.float() - expected.float()).abs().max().item()
        assert torch.isnan(graph_combined_x).any().item() is False
        assert graph_diff <= combine_tolerance, f"rank{rank}: LL CUDA graph combine mismatch diff={graph_diff}"
        dist.barrier(group=group)
        if rank == 0:
            print(f"[cuda graph dispatch+combine] OK max|got-expected|={graph_diff:.4e}", flush=True)

    if args.cuda_graph:
        _run_cuda_graph_correctness()

    # ------------------------------------------------------------------
    # Optional benchmark. In CUDA graph mode, captures dispatch+combine in one
    # graph; otherwise times dispatch and combine separately. Reports per-iter
    # latency (max across ranks) and aggregate effective bandwidth.
    # ------------------------------------------------------------------
    if not args.bench:
        return

    warmup = args.bench_warmup
    iters = args.bench_iters
    bench_dispatch_output_buffer = torch.empty_like(dispatch_output_buffer)

    def _dispatch():
        return moe_comm.dispatch(
            x,
            topk_idx,
            topk_weights,
            output_buffer=bench_dispatch_output_buffer,
        )

    # Hoist combine's output-tensor allocation out of the timed loop so the
    # measurement reflects the kernel cost. (The original test also cloned the
    # ~58 MB dispatch recv buffer on every iter, adding ~20 us of D2D memcpy
    # to each combine sample and masking kernel-level changes.)
    bench_out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def _combine(dout, out_):
        dispatch_out_, handle_ = dout
        moe_comm.combine(dispatch_out_.tokens, handle_, out=out_)

    if args.cuda_graph:
        e2e_graph, e2e_dout, _ = _graph_capture(bench_dispatch_output_buffer, bench_out)
        for _ in range(warmup):
            e2e_graph.replay()
        torch.cuda.synchronize()
        assert e2e_dout[0].layout.num_tokens_per_expert is not None
        recv_tokens = int(e2e_dout[0].layout.num_tokens_per_expert.sum().item())

        dist.barrier(group=group)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            e2e_graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        e2e_us = start_ev.elapsed_time(end_ev) * 1e3 / iters
    else:
        for _ in range(warmup):
            _combine(_dispatch(), bench_out)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        dout = None
        for _ in range(iters):
            dout = _dispatch()
        end_ev.record()
        torch.cuda.synchronize()
        disp_us = start_ev.elapsed_time(end_ev) * 1e3 / iters
        assert dout[0].layout.num_tokens_per_expert is not None
        recv_tokens = int(dout[0].layout.num_tokens_per_expert.sum().item())

        dist.barrier(group=group)
        start_ev.record()
        for _ in range(iters):
            _combine(dout, bench_out)
        end_ev.record()
        torch.cuda.synchronize()
        comb_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Dispatch payload: recv_tokens × hidden × bf16 (received on this rank).
    # Combine payload: recv_tokens × hidden × bf16 as well -- each local expert
    # sends one copy per dispatched token back to its owner, so the bytes on
    # the wire match dispatch. Using num_tokens × hidden here would under-count
    # the actual send payload by ~num_topk×.
    disp_bytes = recv_tokens * hidden * 2
    comb_bytes = recv_tokens * hidden * 2

    if args.cuda_graph:
        e2e_min_t = torch.tensor([e2e_us], dtype=torch.float64, device="cuda")
        e2e_avg_t = torch.tensor([e2e_us], dtype=torch.float64, device="cuda")
        e2e_max_t = torch.tensor([e2e_us], dtype=torch.float64, device="cuda")
        dist.all_reduce(e2e_min_t, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(e2e_avg_t, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(e2e_max_t, op=dist.ReduceOp.MAX, group=group)
        e2e_avg_us = e2e_avg_t.item() / num_ranks
        e2e_bw_per_rank = (disp_bytes + comb_bytes) / (e2e_avg_us * 1e-6) / 1e9
        if rank == 0:
            print(
                f"[bench LL cuda_graph] num_ranks={num_ranks} tokens={num_tokens} hidden={hidden} "
                f"num_experts={num_experts} num_topk={num_topk} warmup={warmup} iters={iters}",
                flush=True,
            )
            print(
                f"  dispatch+combine graph: avg={e2e_avg_us:.1f}us "
                f"min={e2e_min_t.item():.1f}us max={e2e_max_t.item():.1f}us  "
                f"per_rank_bw={e2e_bw_per_rank:.2f} GB/s  "
                f"agg_bw={e2e_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
                flush=True,
            )
        return

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
    disp_bw_per_rank = disp_bytes / (disp_avg_us * 1e-6) / 1e9
    comb_bw_per_rank = comb_bytes / (comb_avg_us * 1e-6) / 1e9
    if rank == 0:
        print(
            f"[bench LL] num_ranks={num_ranks} tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk} warmup={warmup} iters={iters}",
            flush=True,
        )
        print(
            f"  dispatch: avg={disp_avg_us:.1f}us min={disp_min_t.item():.1f}us max={disp_max_t.item():.1f}us  "
            f"per_rank_bw={disp_bw_per_rank:.2f} GB/s  "
            f"agg_bw={disp_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
            flush=True,
        )
        print(
            f"  combine : avg={comb_avg_us:.1f}us min={comb_min_t.item():.1f}us max={comb_max_t.item():.1f}us  "
            f"per_rank_bw={comb_bw_per_rank:.2f} GB/s  "
            f"agg_bw={comb_bw_per_rank * num_ranks:.2f} GB/s  (BW @ avg time)",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
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
            try:
                dist.destroy_process_group()
            except Exception:
                pass
