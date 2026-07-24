# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import os
import torch


# ============================================================================
# Backend: FlashInfer (flashinfer.comm.trtllm_moe_alltoall, MoeAlltoAll / MNNVL).
# ============================================================================
def setup_flashinfer(args, comm, rank, num_ranks, inputs):
    """FlashInfer throughput MoE all-to-all via `flashinfer.comm.trtllm_moe_alltoall.
    MoeAlltoAll` over the MNNVL fabric, wired like the other backends: return
    (dispatch_fn, combine_fn, teardown). No native barrier, so the harness aligns
    ranks with its GPU-side torch NCCL all_reduce (nccl_barrier). The kernels are
    named ``moeA2ADispatchKernel`` / ``moeA2ACombineKernel`` (+ prepare kernels),
    which the kineto timer buckets by the "dispatch"/"combine" substrings.

    Env EP_FLASHINFER_GPUS_PER_NODE (default 4) sets the MNNVL Mapping layout."""
    from flashinfer.comm.mapping import Mapping
    import flashinfer.comm.trtllm_moe_alltoall as a2a

    x, topk_idx, _topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    gpus_per_node = int(os.environ.get("EP_FLASHINFER_GPUS_PER_NODE", "4"))
    dev = x.device
    dtype = torch.bfloat16

    if rank == 0:
        print(
            f"[cfg] backend=flashinfer MoeAlltoAll(MNNVL) num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )
        if args.ep_layout == "expert_major":
            print(
                "[cfg] flashinfer ep_layout=expert_major requested but unsupported: "
                "MoeAlltoAll dispatch is fixed rank-major [ep_size, tokens, hidden]; "
                "keeping rank_major (expert grouping is a downstream model op, not part of the A2A)",
                flush=True,
            )

    # EP is expressed as tp_size = world with moe_ep_size = world, moe_tp_size = 1
    # (Mapping requires world_size == tp*pp*cp).
    mapping = Mapping(
        world_size=num_ranks,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=num_ranks,
        moe_tp_size=1,
        moe_ep_size=num_ranks,
    )
    moe = a2a.MoeAlltoAll(
        mapping,
        max_num_tokens=num_tokens,
        top_k=num_topk,
        num_experts=num_experts,
        hidden_size=hidden,
    )

    # Shared routing inputs: FlashInfer wants int32 expert ids [num_tokens, top_k]
    # and the hidden-state payload in BF16. Reuse the harness's topk_idx / x so the
    # workload matches the other backends.
    token_selected_experts = topk_idx[:, :num_topk].to(torch.int32).contiguous()
    hidden_payload = x.to(dtype).contiguous()
    # Combine payload lives in the received layout [ep_size, max_tokens, hidden].
    combine_payload = torch.randn(num_ranks, num_tokens, hidden, generator=None, device=dev, dtype=dtype)

    # FlashInfer's MoeAlltoAll is STATEFUL: dispatch() sets phase="dispatched" and
    # combine() requires it then resets to "idle". The harness's paired loop calls
    # dispatch_fn then combine_fn in order (phase satisfied). Each op begins with an
    # MPI barrier so all ranks enter the kernel roughly aligned -- FlashInfer's
    # in-kernel peer-readiness spin otherwise deadlocks multi-node when ranks drift
    # (this is exactly how the standalone FlashInfer bench aligns ranks). The
    # host barrier is safe here because FlashInfer dispatch/combine are independent
    # (unlike DeepEP, whose paired ops share symmetric-memory state).
    # EP_KINETO_SEPARATE is forced off for FlashInfer so the timer replays the
    # dispatch->combine pair in order (a lone combine would trip the phase assert).
    os.environ["EP_KINETO_SEPARATE"] = "0"

    graphs = []

    def _dispatch():
        moe.dispatch(token_selected_experts, [hidden_payload], num_tokens)

    def _combine():
        moe.combine(combine_payload, num_tokens)

    captured = False
    if args.cuda_graph:
        # FlashInfer's dispatch/combine each begin with an MPI host barrier to align
        # ranks -- but an MPI barrier cannot live inside a CUDA graph. So capture ONLY
        # the moe kernels (dispatch+combine in a SINGLE graph) and keep the barrier
        # OUTSIDE, replaying: barrier -> replay. The kineto collector attributes
        # per-phase kernel time by kernel name, so one replay per iteration keeps the
        # dispatch/combine breakdown. Capture is best-effort: FlashInfer's stateful
        # phase / in-kernel peer spin may not be graph-capturable on every build, so
        # on failure we rebuild the (now possibly mid-phase) communicator and fall
        # back to the direct barrier+launch.
        try:
            # Prime one full pair so phase returns to idle and lazy work settles.
            _dispatch()
            _combine()
            torch.cuda.synchronize()
            g_all = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_all):
                _dispatch()
                _combine()
            torch.cuda.synchronize()
            graphs = [g_all]
            captured = True
            if rank == 0:
                print("[cfg] flashinfer cuda_graph=True (single graph; MPI barrier kept outside)", flush=True)
        except Exception as e:  # noqa: BLE001 - capturability is an external-library boundary
            if rank == 0:
                print(f"[cfg] flashinfer cuda_graph capture failed ({e}); falling back to eager", flush=True)
            # Rebuild to clear any partially-advanced phase state.
            moe = a2a.MoeAlltoAll(
                mapping,
                max_num_tokens=num_tokens,
                top_k=num_topk,
                num_experts=num_experts,
                hidden_size=hidden,
            )
        comm.Barrier()

    if captured:
        g_all = graphs[0]

        def dispatch_fn():
            comm.Barrier()
            g_all.replay()
            return None

        def combine_fn(_dout):
            pass  # both phases already ran inside the combined graph replay

    else:

        def dispatch_fn():
            comm.Barrier()
            moe.dispatch(token_selected_experts, [hidden_payload], num_tokens)
            return None

        def combine_fn(_dout):
            comm.Barrier()
            moe.combine(combine_payload, num_tokens)

    def teardown():
        graphs.clear()
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown
