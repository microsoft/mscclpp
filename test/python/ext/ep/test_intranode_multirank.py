"""Multi-rank intranode functional validation for mscclpp_ep.

Launch with:
    torchrun --nproc_per_node=<N> test/python/ext/ep/test_intranode_multirank.py

Tests that Buffer::sync() succeeds across N GPUs on a single node and that
a round-trip dispatch + combine preserves data (sum of top-k weighted copies).

This is a minimal adaptation of DeepEP's tests/test_intranode.py stripped
to exercise only the code paths we've ported.
"""

from __future__ import annotations

import os
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

    # Allocate Buffer (intranode only: num_rdma_bytes=0)
    cfg = ep.Config(20, 8, 256)
    num_nvl_bytes = cfg.get_nvl_buffer_size_hint(hidden * x.element_size(), num_ranks)
    if rank == 0:
        print(f"[cfg] num_ranks={num_ranks} num_tokens={num_tokens} hidden={hidden} "
              f"num_experts={num_experts} num_topk={num_topk} num_nvl_bytes={num_nvl_bytes}",
              flush=True)

    print(f"[rank {rank}] creating Buffer", flush=True)
    buf = ep.Buffer(group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0, low_latency_mode=False)
    print(f"[rank {rank}] Buffer created is_available={buf.is_available()}", flush=True)
    assert buf.is_available()

    # get_dispatch_layout sanity
    ref_rank, _, ref_exp, ref_in_rank, _ = buf.runtime.get_dispatch_layout(topk_idx, num_experts, None, False, False)
    assert torch.allclose(ref_rank, num_tokens_per_rank)
    assert torch.allclose(ref_exp, num_tokens_per_expert)
    assert torch.allclose(ref_in_rank, is_token_in_rank)
    if rank == 0:
        print("[layout] OK", flush=True)

    # Dispatch
    (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
     num_recv_tokens_per_expert_list,
     rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
     send_head, _event) = buf.runtime.intranode_dispatch(
        x, None, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        0, None, None,
        1, cfg, None, False, False,
    )
    dist.barrier(group=group)

    # Validate received payloads: for each source rank i, the block of tokens
    # we received from it should be filled with `i`.
    assert recv_x.dim() == 2 and recv_x.size(1) == hidden
    start = 0
    for src in range(num_ranks):
        end = rank_prefix_matrix[src][rank].item()
        block = recv_x[start:end]
        if block.numel():
            actual = block.float().amin().item()
            assert abs(actual - src) < 1e-3, (
                f"rank{rank}: block from src={src} has min={actual}, expected {src}"
            )
            assert abs(block.float().amax().item() - src) < 1e-3
        start = end
    if rank == 0:
        print(f"[dispatch] OK (recv {recv_x.size(0)} tokens)", flush=True)

    # Combine (scatter-reduce back). Using recv_topk_weights=None path with
    # dispatched tokens unchanged => every source rank should receive its
    # contribution back, unweighted sum across topk copies.
    handle_recv_src_idx = recv_src_idx
    handle_rank_prefix_matrix = rank_prefix_matrix
    handle_channel_prefix_matrix = recv_channel_prefix_matrix

    combined_x, combined_topk_weights, _ = buf.runtime.intranode_combine(
        recv_x, recv_topk_weights,
        handle_recv_src_idx, handle_rank_prefix_matrix, handle_channel_prefix_matrix,
        send_head, cfg, None, False, False,
    )

    # Expected: we dispatched with x = rank * ones, so every destination r
    # received the value `rank` for our token. On combine the destinations
    # send that value back and we sum: combined[t] = rank * (#destinations).
    num_dst = is_token_in_rank.sum(dim=1).to(torch.float32)
    expected = num_dst * float(rank)

    got = combined_x.float().mean(dim=1)
    diff = (got - expected).abs().max().item()
    max_exp = expected.abs().max().item()
    if rank == 0:
        print(f"[combine] max|got-expected|={diff:.4e} max|expected|={max_exp:.4e}", flush=True)
    assert diff < 1e-2, f"rank{rank}: combine mismatch max diff {diff}"

    dist.barrier(group=group)
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
