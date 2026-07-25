# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import os
import torch

from ep_bench_common import _ensure_torch_dist


# ============================================================================
# Backend: DeepEP V2 (deepseek-ai/DeepEP, ElasticBuffer low-latency).
# ============================================================================
def setup_deepep(args, comm, rank, num_ranks, inputs):
    """DeepEP V2 low-latency dispatch/combine via `deep_ep.ElasticBuffer`, wired
    the same way as the mscclpp / NCCL-EP backends: return (dispatch_fn,
    combine_fn, teardown). DeepEP needs a torch.distributed NCCL group (reused
    from `_ensure_torch_dist`) and, for a same-rack NVLink run, `EP_DISABLE_GIN=1`.
    The dispatch handle (routing) is fixed for the run, so we dispatch once to
    obtain the handle + combine input and then replay dispatch/combine in the
    timed loop -- dispatch_impl(+copy) and combine_impl(+reduce) are what the
    kineto timer buckets by the "dispatch"/"combine" substrings."""
    import deep_ep

    os.environ.setdefault("EP_DISABLE_GIN", "1")
    group = _ensure_torch_dist(comm, rank, num_ranks)

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk

    if rank == 0:
        print(
            f"[cfg] backend=deepep ElasticBuffer(LOW_LATENCY) num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )

    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        allow_hybrid_mode=1,
        allow_multiple_reduction=1,
        explicitly_destroy=True,
    )

    # Received-token layout: rank_major (plain, do_expand=False -- the deepep default)
    # or expert_major (do_expand=True -- the expanding layout, one slot per expert per
    # token). When --ep-layout is omitted, the plain rank-major layout is kept.
    do_expand = args.ep_layout == "expert_major"
    if rank == 0:
        _lay = "expert_major (do_expand)" if do_expand else "rank_major (plain)"
        print(f"[cfg] deepep ep_layout={args.ep_layout or 'default'} -> {_lay}", flush=True)

    # DeepEP dispatch args (BF16; non-cached so dispatch_impl + copy epilogue run).
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_tokens,
        do_cpu_sync=True,
        do_expand=do_expand,
        do_zero_padding=do_expand,
    )

    # Prime once to obtain the handle (routing) and the received-token count so we
    # can size the combine input. Build a realistic BF16 combine input in the
    # received layout (the role of `simulated_gemm_output`): random values only in
    # the valid received-token slots, matching test_ep.py's `input_for_combine`.
    #
    # NOTE: the PLAIN (rank-major) path uses the per-expert combine; the expanded
    # (expert-major) path runs DeepEP's reduce/expand combine, which processes more
    # rows in the reduce epilogue (measured more expensive on this uniform workload:
    # 1n 74 vs 46, 4n 188 vs 136, 8n 199 vs 120 us). Both are exercised via --ep-layout.
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
    recv_x_bf16 = recv_x[0] if isinstance(recv_x, (tuple, list)) else recv_x
    input_for_combine = torch.empty_like(recv_x_bf16, dtype=torch.bfloat16)
    input_for_combine.normal_(0.0, 0.1)
    if not do_expand:
        # Plain layout: zero the invalid received-token slots (rank-major padding).
        num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        if num_recv_tokens < input_for_combine.shape[0]:
            input_for_combine[num_recv_tokens:] = 0

    combine_args = dict(
        x=input_for_combine,
        topk_weights=recv_topk_weights,
        handle=handle,
    )

    # DeepEP native GPU barrier (comm-stream, sequential) -- aligns its dispatch/
    # combine recv-spin far more tightly than a generic all_reduce; this is what
    # DeepEP's own test_ep.py uses for kineto profiling.
    def deepep_barrier():
        buffer.barrier(use_comm_stream=True, with_cpu_sync=False, sequential=True)

    graphs = []

    # DeepEP CUDA-graph capture only works on the intranode NVLink path. On the
    # internode scale-out path DeepEP's symmetric-memory kernels crash under graph
    # capture (CUDA 719 in symmetric.hpp), so restrict capture to a single node.
    local_world = int(os.environ.get("MSCCLPP_EP_LOCAL_WORLD_SIZE", "0") or "0")
    deepep_can_graph = args.cuda_graph and (local_world <= 0 or num_ranks <= local_world)

    if args.cuda_graph and not deepep_can_graph and rank == 0:
        print(
            "[cfg] deepep cuda_graph requested but disabled: internode scale-out is "
            "not graph-capturable (symmetric-memory); using eager dispatch/combine",
            flush=True,
        )

    if deepep_can_graph:
        # Capturing dispatch+combine in a SINGLE graph replays both phases in one
        # shot, so the skew-free separate kineto pass (which times combine alone)
        # can no longer isolate combine. Force the PAIRED kineto pass so the
        # collector still attributes per-phase kernel time by kernel name. (Intranode
        # NVLink has negligible combine recv-spin skew, so the paired pass matches the
        # separate pass here.)
        os.environ["EP_KINETO_SEPARATE"] = "0"
        if rank == 0:
            print("[cfg] deepep cuda_graph=True (single graph, cached dispatch, do_cpu_sync=False)", flush=True)
        # The non-cached dispatch above did a CPU sync to size the layout; that is
        # illegal inside a CUDA graph. Replay the CACHED dispatch instead: pass the
        # primed handle (topk_idx reused from it), which forces do_cpu_sync=False and
        # skips the host-side count read, leaving a pure on-stream kernel launch.
        cached_dispatch_args = dict(
            x=x,
            handle=handle,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
        )
        # Prime the cached path once so any lazy alloc/autotune settles before capture.
        buffer.dispatch(**cached_dispatch_args)
        buffer.combine(**combine_args)
        torch.cuda.synchronize()

        g_all = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_all):
            buffer.dispatch(**cached_dispatch_args)
            buffer.combine(**combine_args)
        graphs = [g_all]

        def dispatch_fn():
            g_all.replay()
            return None

        def combine_fn(_dout):
            pass  # both phases already ran inside the combined graph replay

    else:

        def dispatch_fn():
            buffer.dispatch(**dispatch_args)
            return None

        def combine_fn(_dout):
            buffer.combine(**combine_args)

    def teardown():
        graphs.clear()  # drop graph refs before the buffer they captured is destroyed
        try:
            buffer.destroy()
        except Exception:
            pass
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown, deepep_barrier
