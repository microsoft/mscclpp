# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Intranode MXFP8 dispatch correctness for the HT MoE backend.

Validates that per-token MXFP8 micro-scales (E8M0, 32-element blocks) are
dispatched **byte-exact** alongside the FP8-E4M3 tokens, and that the high-level
``MoECommunicator(quant_format="mxfp8")`` returns ``dispatch_out.scales`` with
the correct MXFP8 metadata and layout.

The HT dispatch kernel moves scales as FP32 words, so the 1-byte E8M0
micro-scale tensor ``[T, H/32]`` is transported by reinterpreting it as FP32
``[T, H/128]`` (byte-exact when ``H % 128 == 0``). This test checks both the
low-level ``ExpertParallelRuntime`` transport and the high-level API.

Launch with:
    torchrun --nproc_per_node=<N> test/python/ext/ep/test_intranode_mxfp8.py
"""

from __future__ import annotations

import os
import sys

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


def build_routing(num_tokens, num_experts, num_ranks, num_topk):
    """Build a topk layout mapping each token to num_topk distinct ranks/experts,
    plus the rank/expert meta the raw dispatch consumes."""
    scores = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda")

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

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
    return topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert


def main():
    rank, num_ranks, local_rank, group = init_dist()
    from mscclpp.ext import ep

    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"):
        if rank == 0:
            print("[skip] this torch build lacks float8_e4m3fn / float8_e8m0fnu", flush=True)
        return

    num_tokens = 128
    hidden = 512  # must be a multiple of 128 for the E8M0 -> FP32 scale reinterpret
    num_topk = min(4, num_ranks)
    num_experts = num_ranks * 4
    block_size = 32
    num_blocks = hidden // block_size  # 16 E8M0 micro-scale blocks per token

    torch.manual_seed(0xC3D4 + rank)
    topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert = build_routing(
        num_tokens, num_experts, num_ranks, num_topk
    )

    # Encode the SOURCE rank into both the FP8 tokens and the E8M0 micro-scales
    # (value == rank + 1, distinct per rank, exact in both encodings) so we can
    # verify the scales are routed identically to the tokens.
    src_val = rank + 1
    x_fp8 = (torch.ones((num_tokens, hidden), device="cuda") * float(src_val)).to(torch.float8_e4m3fn)
    scales_e8m0 = torch.full((num_tokens, num_blocks), src_val, dtype=torch.uint8, device="cuda").view(
        torch.float8_e8m0fnu
    )

    cfg = ep.Config(
        int(os.environ.get("MSCCLPP_EP_NUM_SMS", "20")),
        int(os.environ.get("MSCCLPP_EP_NVL_SEND", "8")),
        int(os.environ.get("MSCCLPP_EP_NVL_RECV", "256")),
    )
    # Size the NVL buffer with the BF16 upper bound (safe for FP8 tokens); the
    # buffer already reserves scale slots.
    num_nvl_bytes = cfg.get_nvl_buffer_size_hint(hidden * 2, num_ranks)
    if rank == 0:
        print(
            f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
            f"num_experts={num_experts} num_topk={num_topk} block_size={block_size} num_blocks={num_blocks}",
            flush=True,
        )

    buf = ep.ExpertParallelRuntime(group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0, low_latency_mode=False)
    assert buf.is_available()

    # ------------------------------------------------------------------
    # (1) Low-level transport: dispatch FP8 tokens + E8M0 scales, verify the
    #     scales are routed byte-exact to the same recv rows as the tokens.
    # ------------------------------------------------------------------
    x_scales_transport = scales_e8m0.view(torch.float32)  # [T, num_blocks/4] == [T, H/128]
    (
        recv_x,
        recv_x_scales,
        _recv_topk_idx,
        _recv_topk_weights,
        _num_recv_tokens_per_expert_list,
        rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        _recv_src_idx,
        _send_head,
    ) = buf.intranode_dispatch(
        x_fp8,
        x_scales_transport,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        0,
        None,
        None,
        1,
        cfg,
    )
    dist.barrier(group=group)

    assert recv_x.dtype == torch.float8_e4m3fn and recv_x.size(1) == hidden, "recv_x must stay FP8 [recv, H]"
    recv_scales_e8m0 = recv_x_scales.view(torch.float8_e8m0fnu)
    assert recv_scales_e8m0.shape == (recv_x.size(0), num_blocks), "recv scales must be [recv, H/32] E8M0"

    # recv_x is grouped by source rank; rank_prefix_matrix[src][rank] is the
    # cumulative recv count from sources <= src. Each block must decode to its
    # source (src + 1) in BOTH the tokens and the micro-scales.
    start = 0
    for src in range(num_ranks):
        end = rank_prefix_matrix[src][rank].item()
        if end > start:
            tok_block = recv_x[start:end].float()
            sc_block = recv_scales_e8m0[start:end].view(torch.uint8)
            assert (tok_block == float(src + 1)).all(), f"rank{rank}: token block from src={src} != {src + 1}"
            assert (sc_block == (src + 1)).all(), f"rank{rank}: E8M0 scale block from src={src} != {src + 1}"
        start = end
    if rank == 0:
        print(
            f"[raw mxfp8] FP8 tokens + E8M0 micro-scales routed byte-exact (recv {recv_x.size(0)} tokens)  OK",
            flush=True,
        )

    # ------------------------------------------------------------------
    # (2) High-level MoECommunicator(quant_format="mxfp8"): verify the returned
    #     dispatch_out.scales metadata/layout and token<->scale self-consistency.
    # ------------------------------------------------------------------
    moe = ep.MoECommunicator(
        group=group,
        num_experts=num_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.HIGH_THROUGHPUT,
        quant_format="mxfp8",
        num_sms=int(os.environ.get("MSCCLPP_EP_NUM_SMS", "20")),
    )
    assert moe.is_available()
    dout, _handle = moe.dispatch(
        x_fp8, topk_idx, topk_weights, scales=ep.QuantScales(local=scales_e8m0, format="mxfp8")
    )
    assert dout.tokens.dtype == torch.float8_e4m3fn, "dispatch tokens must stay FP8"
    assert dout.scales is not None, "MXFP8 dispatch must return scales"
    assert dout.scales.format == "mxfp8", f"format={dout.scales.format}"
    assert dout.scales.block_size == block_size, f"block_size={dout.scales.block_size}"
    assert dout.scales.local.dtype == torch.float8_e8m0fnu, f"scale dtype={dout.scales.local.dtype}"
    assert dout.scales.local.shape == (dout.tokens.size(0), num_blocks), f"scale shape={tuple(dout.scales.local.shape)}"
    # Self-consistency: each recv row's token and scale must encode the same source.
    tok_src = dout.tokens.float()[:, 0].round().to(torch.int64)
    sc_src = dout.scales.local.view(torch.uint8)[:, 0].to(torch.int64)
    assert torch.equal(tok_src, sc_src), f"rank{rank}: recv token/scale source mismatch"

    dist.barrier(group=group)
    if rank == 0:
        print("[moe mxfp8] dispatch_out.scales metadata + token/scale self-consistency  OK", flush=True)
        print("PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
