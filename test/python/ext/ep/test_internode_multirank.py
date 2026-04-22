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
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
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
    assert num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS, \
        f"expected >1 node with 8 GPUs each, got num_ranks={num_ranks}"
    num_nodes = num_ranks // NUM_MAX_NVL_PEERS
    num_local_ranks = NUM_MAX_NVL_PEERS

    # Small settings for functional check
    num_tokens = 128
    hidden = 1024
    num_topk = min(4, num_ranks)
    num_experts = (num_ranks * 4)  # multiple of num_ranks

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

    # Buffer config for internode HT: needs num_rdma_bytes > 0.
    cfg = ep.Config(20, 8, 256, 16, 128)
    num_nvl_bytes = cfg.get_nvl_buffer_size_hint(hidden * x.element_size(), num_ranks)
    num_rdma_bytes = cfg.get_rdma_buffer_size_hint(hidden * x.element_size(), num_ranks)
    if rank == 0:
        print(f"[cfg] num_nodes={num_nodes} num_ranks={num_ranks} num_tokens={num_tokens} "
              f"hidden={hidden} num_experts={num_experts} num_topk={num_topk} "
              f"num_nvl_bytes={num_nvl_bytes} num_rdma_bytes={num_rdma_bytes}",
              flush=True)

    print(f"[rank {rank}] creating Buffer", flush=True)
    buf = ep.Buffer(group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes, low_latency_mode=False)
    print(f"[rank {rank}] Buffer created is_available={buf.is_available()} "
          f"is_internode={buf.is_internode_available()}", flush=True)
    assert buf.is_available() and buf.is_internode_available()

    ref_rank, ref_rdma_rank, ref_exp, ref_in_rank, _ = \
        buf.runtime.get_dispatch_layout(topk_idx, num_experts, None, False, False)
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
    (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
     num_recv_tokens_per_expert_list,
     rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
     recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
     recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
     recv_src_meta, send_rdma_head, send_nvl_head, _event) = buf.runtime.internode_dispatch(
        x, None, topk_idx, topk_weights,
        num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
        0, 0,
        None, None, None, None,
        1, cfg, None, False, False,
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
            assert abs(lo - src) < 1e-3 and abs(hi - src) < 1e-3, (
                f"rank{rank}: block from src={src} has range=[{lo}, {hi}], expected {src}"
            )
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
        recv_x, recv_topk_weights,
        recv_src_meta, is_token_in_rank,
        recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix,
        send_rdma_head, send_nvl_head,
        cfg, None, False, False,
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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
