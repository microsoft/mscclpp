# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Multi-rank low-latency functional test for mscclpp_ep.

Launch with (intra-node, 8 GPUs):
    torchrun --nproc_per_node=8 test/python/ext/ep/test_low_latency_multirank.py \
        --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256

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
we need for an LL port smoke test. Covers BF16, FP8-E4M3 (block-128 FP32
scales), and MXFP8 (block-32 E8M0 micro-scales) dispatch via ``--quant``.
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
    parser.add_argument("--hidden", type=int, default=7168, help="LL kernels are compiled for a fixed hidden set")
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--quant",
        choices=("none", "fp8", "mxfp8"),
        default="none",
        help="dispatch activation quant: none (BF16), fp8 (E4M3 block-128), mxfp8 (E8M0 block-32)",
    )
    parser.add_argument(
        "--prequantized",
        action="store_true",
        help="pass pre-quantized FP8/MXFP8 input (in==out passthrough) instead of BF16 (kernel quantizes)",
    )
    parser.add_argument("--bench", action="store_true", help="Run dispatch/combine benchmark after correctness")
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


def _dequantize_recv(recv_x_fp8, scales_local_expert, block):
    """Dequantize [tokens, hidden] FP8 codes using [tokens, num_scales] block scales.

    Works for both FP8-E4M3 (FP32 reciprocal scales) and MXFP8 (E8M0 scales):
    ``scales.float()`` yields the multiply-back factor in either case (E8M0 -> the
    power-of-two 2^(e-127); FP32 reciprocal -> passthrough).
    """
    recv_f = recv_x_fp8.float()
    scale_mul = scales_local_expert.float()
    scale_full = scale_mul.repeat_interleave(block, dim=1)[:, : recv_f.size(1)]
    return recv_f * scale_full


def _dequantize_full(recv_x_fp8, scales_local, block):
    """Dequantize a full [E, S, H] FP8 recv buffer with [E, S, H//block] scales to BF16."""
    recv_f = recv_x_fp8.float()
    scale_full = scales_local.float().repeat_interleave(block, dim=2)[..., : recv_f.size(2)]
    return (recv_f * scale_full).nan_to_num(0.0).to(torch.bfloat16)


def _quantize_fp8(x, block=128):
    """Quantize [T, H] BF16 -> (FP8-E4M3 tokens [T, H], FP32 reciprocal scales [T, H//block])."""
    tokens, hidden = x.shape
    xb = x.float().reshape(tokens, hidden // block, block)
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    x_q = (xb * (448.0 / amax).unsqueeze(-1)).reshape(tokens, hidden).to(torch.float8_e4m3fn)
    scale_inv = (amax / 448.0).contiguous()
    return x_q.contiguous(), scale_inv


def _quantize_mxfp8(x, block=32):
    """Quantize [T, H] BF16 -> (FP8-E4M3 tokens [T, H], E8M0 scales [T, H//block])."""
    tokens, hidden = x.shape
    xb = x.float().reshape(tokens, hidden // block, block)
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127).to(torch.int32)
    x_q = (xb * torch.exp2(-exp.float()).unsqueeze(-1)).reshape(tokens, hidden).to(torch.float8_e4m3fn)
    e8m0 = (exp + 127).to(torch.uint8).contiguous().view(torch.float8_e8m0fnu)
    return x_q.contiguous(), e8m0


def _validate_dispatch_scales(
    dispatch_out,
    counts,
    packed_recv_x,
    num_local_experts,
    num_ranks,
    num_tokens,
    hidden,
    quant_format,
    block,
    scale_dtype,
    rank,
):
    """Check the quantized-dispatch scale metadata/shape/dtype and that the
    constant source region dequantizes back to a per-row constant."""
    scales = dispatch_out.scales
    assert scales is not None and scales.local is not None, "quantized dispatch must return scales"
    assert scales.format == quant_format, f"scales.format={scales.format} != {quant_format}"
    assert scales.block_size == block, f"scales.block_size={scales.block_size} != {block}"
    assert scales.local.dtype == scale_dtype, f"scales.local dtype {scales.local.dtype} != {scale_dtype}"
    slots = num_ranks * num_tokens
    num_scales = hidden // block
    assert tuple(scales.local.shape) == (
        num_local_experts,
        slots,
        num_scales,
    ), f"scales.local shape {tuple(scales.local.shape)} != {(num_local_experts, slots, num_scales)}"
    tol = 0.02 if quant_format == "fp8_e4m3" else 0.05
    for i in range(num_local_experts):
        recv_count = int(counts[i].item())
        if recv_count == 0:
            continue
        deq = _dequantize_recv(packed_recv_x[i, :recv_count], scales.local[i, :recv_count], block)
        assert torch.isfinite(deq).all(), f"rank{rank} expert{i}: non-finite dequant"
        deq_lo = deq[:, :-128]
        row_span = (deq_lo.amax(dim=1) - deq_lo.amin(dim=1)).abs()
        row_scale = deq_lo.abs().amax(dim=1).clamp_min(1e-3)
        rel = (row_span / row_scale).amax().item()
        assert rel < tol, f"rank{rank} expert{i}: dequant lo region not constant, rel={rel:.4e}"
        break  # one populated expert is a sufficient smoke check


def main():
    args = parse_args()
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp import CommGroup
    from mscclpp.ext import ep

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

    quant_format = {"none": None, "fp8": "fp8_e4m3", "mxfp8": "mxfp8"}[args.quant]
    is_quant = quant_format is not None
    recv_dtype = torch.float8_e4m3fn if is_quant else torch.bfloat16
    expected_scale_block = {"fp8": 128, "mxfp8": 32}.get(args.quant)
    expected_scale_dtype = {"fp8": torch.float32, "mxfp8": getattr(torch, "float8_e8m0fnu", None)}.get(args.quant)
    assert not (args.prequantized and not is_quant), "--prequantized requires --quant fp8 or mxfp8"

    torch.manual_seed(0xB3C4 + rank)
    random.seed(0xB3C4 + rank)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    # Encode the per-token index into the last 128 elements so the receiver
    # can verify which source token it is looking at.
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()

    # Randomly mask some positions
    for _ in range(min(10, num_tokens)):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # Pre-quantized passthrough (MXFP8/FP8 in -> MXFP8/FP8 out): the caller quantizes
    # x and supplies FP8 tokens + block scales, which dispatch transports verbatim.
    dispatch_scales = None
    input_dtype = None
    if args.prequantized:
        input_dtype = torch.float8_e4m3fn
        if quant_format == "mxfp8":
            disp_x, x_scales = _quantize_mxfp8(x, expected_scale_block)
        else:
            disp_x, x_scales = _quantize_fp8(x, expected_scale_block)
        dispatch_scales = ep.QuantScales(local=x_scales, format=quant_format, block_size=expected_scale_block)
    else:
        disp_x = x

    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        num_rdma_qps_per_rank=max(1, num_experts // num_ranks),
        quant_format=quant_format,
        input_dtype=input_dtype,
    )
    if rank == 0:
        print(
            f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk} quant={args.quant} prequantized={args.prequantized}",
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
        dtype=recv_dtype,
        device="cuda",
    )
    dispatch_out, handle = moe_comm.dispatch(
        disp_x,
        topk_idx,
        topk_weights,
        scales=dispatch_scales,
        output_buffer=dispatch_output_buffer,
    )
    packed_recv_x = dispatch_out.tokens
    packed_recv_count = dispatch_out.num_tokens_per_expert
    packed_recv_layout_range = handle.layout_range
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
            # All columns except the last 128 share the source value (src_rank -
            # rank_offset); for quantized dispatch the raw FP8 codes are uniform
            # too because each source block is constant.
            recv_x_lo = recv_x[:, :-128].float()
            amin = recv_x_lo.amin(dim=-1)
            amax = recv_x_lo.amax(dim=-1)
            assert torch.equal(amin, amax), f"rank{rank} expert{expert_id}: non-uniform recv block"

    if rank == 0:
        print(f"[dispatch] OK (ranks={num_ranks})", flush=True)

    if is_quant:
        _validate_dispatch_scales(
            dispatch_out,
            packed_recv_count,
            packed_recv_x,
            num_local_experts,
            num_ranks,
            num_tokens,
            hidden,
            quant_format,
            expected_scale_block,
            expected_scale_dtype,
            rank,
        )
        if rank == 0:
            print(f"[dispatch quant={args.quant}] scale + dequant check OK", flush=True)

    # --- Combine (BF16 in -> BF16 out) ---
    # Combine consumes BF16 expert outputs; for quantized dispatch, dequantize the
    # received tokens to BF16 first (a real expert GEMM would emit BF16). This makes
    # combine return sum(x * weight) across experts.
    if is_quant:
        simulated_gemm_x = _dequantize_full(packed_recv_x, dispatch_out.scales.local, expected_scale_block)
    else:
        simulated_gemm_x = packed_recv_x.clone()
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    combined_x = moe_comm.combine(simulated_gemm_x, handle, out=out)

    # Analytical expected: each token i, weighted sum over topk entries that
    # are not -1. Accumulate in the same top-k order as the kernel; multiplying
    # by the pre-summed weights can differ by one BF16 ULP for large token IDs.
    expected_f = torch.zeros_like(x, dtype=torch.float32)
    x_f = x.float()
    for j in range(num_topk):
        weight_j = topk_weights[:, j].masked_fill(topk_idx[:, j] == -1, 0.0).view(-1, 1)
        expected_f += x_f * weight_j
    expected = expected_f.to(torch.bfloat16)
    diff = (combined_x.float() - expected.float()).abs().max().item()
    max_exp = expected.float().abs().max().item()
    print(
        f"[combine r{rank}] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}",
        flush=True,
    )
    assert torch.isnan(combined_x).any().item() is False
    if is_quant:
        rel = diff / max(max_exp, 1e-3)
        tol = 0.05 if quant_format == "fp8_e4m3" else 0.15
        assert rel < tol, f"rank{rank}: LL {args.quant} combine mismatch rel={rel:.4e} diff={diff} max={max_exp}"
    else:
        assert diff < 1e-2, f"rank{rank}: LL combine mismatch diff={diff}"

    dist.barrier(group=group)
    if rank == 0:
        print("PASS", flush=True)

    # ------------------------------------------------------------------
    # Optional benchmark. Times dispatch and combine separately, reporting
    # per-iter latency (max across ranks) and aggregate effective bandwidth
    # (sum across ranks).
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
    recv_tokens = int(dout[0].num_tokens_per_expert.sum().item())

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
