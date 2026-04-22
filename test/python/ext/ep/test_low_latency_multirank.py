"""Multi-rank low-latency functional test for mscclpp_ep.

Launch with (intra-node, 8 GPUs):
    torchrun --nproc_per_node=8 test/python/ext/ep/test_low_latency_multirank.py

Launch with (2 nodes, 1 GPU per node -- DeepEP's recommended LL topology):
    # node 0:
    MASTER_ADDR=<master> MASTER_PORT=29600 NODE_RANK=0 \
        torchrun --nnodes=2 --nproc_per_node=1 --rdzv-backend=c10d \
            --rdzv-endpoint=<master>:29600 test/python/ext/ep/test_low_latency_multirank.py
    # node 1:
    MASTER_ADDR=<master> MASTER_PORT=29600 NODE_RANK=1 \
        torchrun --nnodes=2 --nproc_per_node=1 --rdzv-backend=c10d \
            --rdzv-endpoint=<master>:29600 test/python/ext/ep/test_low_latency_multirank.py

Exercises the LL dispatch + combine round-trip on a single node. The
minimal correctness check:
  - dispatch: per-expert received token counts agree with an all-gathered
    reference computed from topk_idx across all ranks;
  - combine: the reconstructed x matches the analytical sum
    ``x * sum(topk_weights, masked by topk_idx == -1)``.

Known limitation (see src/ext/ep/README.md): the LL kernels drive every
peer via MSCCL++ PortChannel. Intra-node IB loopback between two HCAs on
the same host (what an 8-GPU single-node launch exercises) currently hangs
during dispatch; cross-node LL with one GPU per node works as designed.

Adapted from DeepEP/tests/test_low_latency.py stripped to the bare checks
we need for an LL port smoke test. BF16-only (no FP8 check).
"""

from __future__ import annotations

import os
import random
import sys

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


def main():
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp.ext import ep

    # Shrink the "bf16 precision" anchor to keep values small.
    rank_offset = 128
    assert num_ranks - rank_offset < 257, "too many ranks for bf16 precision anchor"

    num_tokens = 64
    hidden = 7168  # LL kernels are compiled for a fixed set; see SWITCH_HIDDEN
    num_topk = 4
    num_experts = num_ranks * 4
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    torch.manual_seed(0xB3C4 + rank)
    random.seed(0xB3C4 + rank)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    # Encode the per-token index into the last 128 elements so the receiver
    # can verify which source token it is looking at.
    x[:, -128:] = (
        torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    )
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()

    # Randomly mask some positions
    for _ in range(min(10, num_tokens)):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    num_rdma_bytes = ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    if rank == 0:
        print(
            f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk} "
            f"num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    buf = ep.Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=max(1, num_experts // num_ranks),
    )
    print(
        f"[rank {rank}] Buffer created is_available={buf.is_available()} "
        f"is_internode={buf.is_internode_available()}",
        flush=True,
    )
    assert buf.is_available()

    dist.barrier(group=group)
    torch.cuda.synchronize()
    print(f"[rank {rank}] pre-dispatch", flush=True)

    # --- Dispatch ---
    # Return tuple (7 items):
    #   packed_recv_x, packed_recv_x_scales (optional, FP8-only),
    #   packed_recv_count, packed_recv_src_info, packed_recv_layout_range,
    #   event, hook
    (
        packed_recv_x, _packed_recv_x_scales,
        packed_recv_count, packed_recv_src_info, packed_recv_layout_range,
        _event, recv_hook,
    ) = buf.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts,
        False, False, True,  # use_fp8, async, return_recv_hook
    )
    # Send phase launched on compute_stream; wait for local launch.
    torch.cuda.synchronize()
    dist.barrier(group=group)
    print(f"[rank {rank}] dispatch-send done, calling hook", flush=True)
    recv_hook()  # Recv phase.
    torch.cuda.synchronize()
    print(f"[rank {rank}] post-dispatch", flush=True)
    handle = (packed_recv_src_info, packed_recv_layout_range)
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
        recv_layout_range = handle[1][i]
        layout_sum = int((recv_layout_range & int_mask).sum().item())
        assert recv_count == expected_count, (
            f"rank{rank} expert{expert_id}: recv_count={recv_count} != expected={expected_count}"
        )
        assert layout_sum == recv_count, (
            f"rank{rank} expert{expert_id}: layout range sum {layout_sum} != recv_count {recv_count}"
        )

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
    # Signature: (x, topk_idx, topk_weights, src_info, layout_range,
    #             num_max_dispatch_tokens_per_rank, num_experts,
    #             zero_copy, async, return_recv_hook, out)
    src_info, layout_range = handle[0], handle[1]
    combined_x, _event, _hook = buf.low_latency_combine(
        simulated_gemm_x, topk_idx, topk_weights,
        src_info, layout_range,
        num_tokens, num_experts,
        False, False, False,  # zero_copy, async, return_recv_hook
        out,
    )

    # Analytical expected: each token i, weighted sum over topk entries that
    # are not -1. Every expert returns the original x[i] (since simulated
    # gemm is identity), so the combine output should be
    # x[i] * sum(topk_weights[i, j] for j where topk_idx[i,j] != -1).
    weight_sum = topk_weights.masked_fill(topk_idx == -1, 0.0).sum(dim=1).view(-1, 1)
    expected = (x.float() * weight_sum).to(torch.bfloat16)
    diff = (combined_x.float() - expected.float()).abs().max().item()
    max_exp = expected.float().abs().max().item()
    print(
        f"[combine r{rank}] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}",
        flush=True,
    )
    assert torch.isnan(combined_x).any().item() is False
    assert diff < 1e-2, f"rank{rank}: LL combine mismatch diff={diff}"

    dist.barrier(group=group)
    if rank == 0:
        print("PASS", flush=True)

    # ------------------------------------------------------------------
    # Optional benchmark (enable with MSCCLPP_EP_BENCH=1). Times dispatch
    # and combine separately, reporting per-iter latency (max across ranks)
    # and aggregate effective bandwidth (sum across ranks).
    # ------------------------------------------------------------------
    if os.environ.get("MSCCLPP_EP_BENCH", "0") != "1":
        return

    warmup = int(os.environ.get("MSCCLPP_EP_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("MSCCLPP_EP_BENCH_ITERS", "20"))

    def _dispatch():
        return buf.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            False, False, False,  # use_fp8, async, return_recv_hook
        )

    def _combine(dout):
        (recv_x, _scales, _cnt, src_info_, layout_range_, _ev, _hk) = dout
        out_ = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        buf.low_latency_combine(
            recv_x.clone(), topk_idx, topk_weights,
            src_info_, layout_range_,
            num_tokens, num_experts,
            False, False, False,
            out_,
        )

    for _ in range(warmup):
        _combine(_dispatch())
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
    recv_tokens = int(dout[2].sum().item())  # packed_recv_count summed over local experts

    dist.barrier(group=group)
    start_ev.record()
    for _ in range(iters):
        _combine(dout)
    end_ev.record()
    torch.cuda.synchronize()
    comb_us = start_ev.elapsed_time(end_ev) * 1e3 / iters

    # Dispatch payload: recv_tokens × hidden × bf16 (received on this rank).
    # Combine payload: num_tokens × hidden × bf16 (sent from each local expert
    # back to the owning rank; one token's worth of bytes per reduction).
    disp_bytes = recv_tokens * hidden * 2
    comb_bytes = num_tokens * hidden * 2
    disp_bw = disp_bytes / (disp_us * 1e-6) / 1e9
    comb_bw = comb_bytes / (comb_us * 1e-6) / 1e9

    disp_us_t = torch.tensor([disp_us], dtype=torch.float64, device="cuda")
    comb_us_t = torch.tensor([comb_us], dtype=torch.float64, device="cuda")
    disp_bw_t = torch.tensor([disp_bw], dtype=torch.float64, device="cuda")
    comb_bw_t = torch.tensor([comb_bw], dtype=torch.float64, device="cuda")
    dist.all_reduce(disp_us_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(comb_us_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(disp_bw_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(comb_bw_t, op=dist.ReduceOp.SUM, group=group)
    if rank == 0:
        print(
            f"[bench LL] num_ranks={num_ranks} tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} warmup={warmup} iters={iters}",
            flush=True,
        )
        print(
            f"  dispatch: {disp_us_t.item():.1f}us (max)  agg_bw={disp_bw_t.item():.2f} GB/s",
            flush=True,
        )
        print(
            f"  combine : {comb_us_t.item():.1f}us (max)  agg_bw={comb_bw_t.item():.2f} GB/s",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
