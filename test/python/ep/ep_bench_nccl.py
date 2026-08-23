# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import torch

from ep_bench_common import sum_matching_kernel_us


def parse_kineto_kernels(key_averages):
    """Map a kineto key_averages() table to (dispatch_us, combine_us) for NCCL-EP,
    which runs a single dispatch kernel and a single combine kernel with the
    phase word in each function name."""
    return (
        sum_matching_kernel_us(key_averages, ("dispatch",)),
        sum_matching_kernel_us(key_averages, ("combine",)),
    )


# ============================================================================
# Backend: NVIDIA NCCL-EP (nccl.ep Group/Handle).
# ============================================================================
def setup_nccl(args, comm, rank, num_ranks, inputs):
    import nccl.core as nccl_core
    import nccl.ep as nccl_ep

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks
    if args.ep_layout == "token_major":
        raise ValueError("NCCL-EP supports expert_major or rank_major layout")
    if args.mode == "latency":
        algorithm = nccl_ep.Algorithm.LOW_LATENCY
    else:
        algorithm = getattr(
            nccl_ep.Algorithm,
            "THROUGHPUT",
            getattr(nccl_ep.Algorithm, "HIGH_THROUGHPUT", None),
        )
        if algorithm is None:
            raise RuntimeError("installed NCCL-EP does not expose a throughput algorithm")
    algorithm_name = getattr(algorithm, "name", str(algorithm))

    if rank == 0:
        print(
            f"[cfg] backend=nccl algorithm={algorithm_name} num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )

    # NCCL communicator: rank 0 makes a unique id, broadcast over MPI.
    uid = nccl_core.get_unique_id() if rank == 0 else None
    uid = comm.bcast(uid, root=0)
    ncomm = nccl_core.Communicator.init(nranks=num_ranks, rank=rank, unique_id=uid)

    config = nccl_ep.GroupConfig(
        algorithm=algorithm,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=num_tokens,
        max_recv_tokens_per_rank=num_tokens * num_ranks,
        max_token_bytes=hidden * 2,  # BF16
        alloc=nccl_ep.AllocConfig(),  # default cudaMalloc/cudaFree
    )
    ep_group = nccl_ep.Group.create(ncomm, config)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Received-token layout: default/expert_major -> EXPERT_MAJOR (the nccl default),
    # rank_major -> RANK_MAJOR. NCCL-EP supports both natively via the handle Layout.
    nccl_layout = nccl_ep.Layout.RANK_MAJOR if args.ep_layout == "rank_major" else nccl_ep.Layout.EXPERT_MAJOR
    if rank == 0:
        _lname = getattr(nccl_layout, "name", str(nccl_layout))
        print(f"[cfg] nccl ep_layout={args.ep_layout or 'default'} -> Layout.{_lname}", flush=True)

    # Routing is encoded in the handle at create time (topk_idx is fixed for the run).
    topk_idx_t = nccl_ep.Tensor(topk_idx)
    ep_handle = ep_group.create_handle(
        nccl_layout,
        topk_idx_t,
        layout_info=None,
        config=nccl_ep.HandleConfig(),
        stream=stream_ptr,
    )

    # Pre-allocated EP tensors (hoisted out of the timed loop). The recv-buffer shape
    # and layout counters differ by layout: EXPERT_MAJOR groups received tokens per
    # local expert ([num_local_experts, num_ranks*num_tokens, hidden] + per-expert
    # counters); RANK_MAJOR groups them per source rank ([num_ranks, num_tokens, hidden]
    # + per-source-rank counters, which the LL kernel asserts has leading dim == nRanks).
    if nccl_layout == nccl_ep.Layout.RANK_MAJOR:
        recv = torch.empty((num_ranks, num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        recv_count = torch.empty((num_ranks,), dtype=torch.int32, device="cuda")
    else:
        recv = torch.empty((num_local_experts, num_ranks * num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        recv_count = torch.empty((num_local_experts,), dtype=torch.int32, device="cuda")
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    x_t = nccl_ep.Tensor(x)
    recv_t = nccl_ep.Tensor(recv)
    recv_count_t = nccl_ep.Tensor(recv_count)
    out_t = nccl_ep.Tensor(out)
    topk_weights_t = nccl_ep.Tensor(topk_weights)

    if nccl_layout == nccl_ep.Layout.RANK_MAJOR:
        # RANK_MAJOR dispatch carries the per-token weights through as well (the LL
        # kernel asserts topk_weights_in != NULL), and writes them into a matching
        # per-source-rank recv buffer.
        recv_weights = torch.empty((num_ranks, num_tokens, num_topk), dtype=torch.float32, device="cuda")
        recv_weights_t = nccl_ep.Tensor(recv_weights)
        recv_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=torch.int32, device="cuda")
        recv_idx_t = nccl_ep.Tensor(recv_idx)
        dispatch_inputs = nccl_ep.DispatchInputs(tokens=x_t, topk_weights=topk_weights_t)
        dispatch_outputs = nccl_ep.DispatchOutputs(tokens=recv_t, topk_weights=recv_weights_t, topk_idx=recv_idx_t)
        dispatch_layout = nccl_ep.LayoutInfo(src_rank_counters=recv_count_t)
    else:
        dispatch_inputs = nccl_ep.DispatchInputs(tokens=x_t)
        dispatch_outputs = nccl_ep.DispatchOutputs(tokens=recv_t)
        dispatch_layout = nccl_ep.LayoutInfo(expert_counters=recv_count_t)
    dispatch_config = nccl_ep.DispatchConfig()
    combine_inputs = nccl_ep.CombineInputs(tokens=recv_t)
    combine_outputs = nccl_ep.CombineOutputs(tokens=out_t, topk_weights=topk_weights_t)
    combine_config = nccl_ep.CombineConfig()

    def _dispatch(stream):
        # Pure on-stream kernel launch; routing is baked into ep_handle, all
        # tensors are pre-allocated, so nothing runs host-side per call.
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=dispatch_config,
            stream=stream,
        )

    def _combine(stream):
        ep_handle.combine(combine_inputs, combine_outputs, config=combine_config, stream=stream)

    # NCCL-EP dispatch/combine are a PAIRED collective (combine consumes the peer
    # data produced by the matching dispatch). The op closures below re-fetch the
    # current stream so they are capture-safe: NCCL-EP takes an explicit stream
    # pointer, and under CUDA-graph capture the harness runs them on the graph's
    # capture stream (the cached stream_ptr would launch off-capture and break it).
    def dispatch_fn():
        _dispatch(torch.cuda.current_stream().cuda_stream)
        return None

    def combine_fn(_dout):
        _combine(torch.cuda.current_stream().cuda_stream)

    # Capturable at any scale; hand the same capture-safe ops to the harness, which
    # owns the single-graph capture. When captured, main() times the paired kineto
    # pass so per-phase kernel time is still attributed by kernel name.
    graph_spec = None
    if args.cuda_graph and args.mode == "latency":
        graph_spec = {
            "dispatch": lambda: _dispatch(torch.cuda.current_stream().cuda_stream),
            "combine": lambda: _combine(torch.cuda.current_stream().cuda_stream),
            "on_fail": None,
        }
    elif args.cuda_graph and rank == 0:
        print("[cfg] nccl throughput cuda_graph is not enabled in this benchmark; using eager mode", flush=True)

    def teardown():
        ep_handle.destroy()
        ep_group.destroy()
        ncomm.destroy()
        gc.collect()
        torch.cuda.synchronize()

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": None,
        "graph": graph_spec,
    }
