# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import torch

from ep_bench_common import simulated_gemm_output, validate_combine_output_mpi


# ============================================================================
# Backend: mscclpp EP (MoECommunicator).
# ============================================================================
def setup_mscclpp(args, comm, rank, num_ranks, inputs):
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks

    num_rdma_bytes = 0  # not exposed by current mscclpp API; 0 over the CUDA-IPC path
    if rank == 0:
        print(
            f"[cfg] backend=mscclpp algorithm=LOW_LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters} num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    ep_group = CommGroup(mpi_comm=comm)
    combine_mode = {
        "rank_local_reduce": ep.CombineMode.RANK_LOCAL_REDUCE,
        "direct_send": ep.CombineMode.DIRECT_SEND,
    }[args.combine_mode]
    # Received-token layout: expert-major (default) or rank-major. RANK_MAJOR LL
    # requires RANK_LOCAL_REDUCE combine (each rank's tokens grouped contiguously,
    # then reduced locally). Uses the merged binyli/ep rank-major LL kernels.
    use_rank_major = args.ep_layout == "rank_major"
    output_layout = ep.DispatchLayout.RANK_MAJOR if use_rank_major else ep.DispatchLayout.EXPERT_MAJOR
    if use_rank_major and combine_mode != ep.CombineMode.RANK_LOCAL_REDUCE:
        raise SystemExit("mscclpp --ep-layout rank_major requires --combine-mode rank_local_reduce")
    dispatch_quant = ep.QuantConfig(format=ep.DispatchDataType.FP8_E4M3) if args.dispatch_dtype == "fp8_e4m3" else None
    dispatch_dtype = torch.float8_e4m3fn if dispatch_quant is not None else torch.bfloat16
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        low_latency_combine_mode=combine_mode,
        output_layout=output_layout,
        quant=dispatch_quant,
    )
    assert moe_comm.is_available()
    if rank == 0:
        print(
            f"[cfg] mscclpp MoECommunicator is_internode={moe_comm.is_internode()} "
            f"dispatch_dtype={args.dispatch_dtype} combine_mode={args.combine_mode} "
            f"ep_layout={'rank_major' if use_rank_major else 'expert_major'} cuda_graph={args.cuda_graph}",
            flush=True,
        )

    # Hoist output tensors out of the timed loop (the communicator owns its
    # src_info/layout_range/count buffers internally). EXPERT_MAJOR supplies its
    # dispatch output buffer; RANK_MAJOR uses the runtime-owned expert-output buffer
    # (output_buffer=None) and feeds that buffer straight into combine.
    if use_rank_major:
        output_buffer = None
        expert_output = moe_comm.get_expert_output_buffer()
        expert_output.zero_()
    else:
        output_buffer = torch.empty(
            (num_local_experts, num_ranks * num_tokens, hidden), dtype=dispatch_dtype, device="cuda"
        )
        expert_output = None
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def _dispatch():
        # Full (send+recv) LL dispatch inline on the stream; returns (dispatch_out, handle).
        return moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)

    def _combine(dispatch_out, handle):
        # RANK_MAJOR: feed the runtime rank-major expert-output buffer directly.
        # EXPERT_MAJOR: feed BF16 expert output (identity for BF16, dequantized for FP8).
        combine_in = expert_output if use_rank_major else simulated_gemm_output(dispatch_out)
        moe_comm.combine(combine_in, handle, out=out)

    # Optional one-time correctness check (mirrors test_low_latency_multirank).
    if args.validate:
        v_dispatch_out, v_handle = _dispatch()
        v_out = torch.empty_like(out)
        v_combine_in = expert_output if use_rank_major else simulated_gemm_output(v_dispatch_out)
        moe_comm.combine(v_combine_in, v_handle, out=v_out)
        torch.cuda.synchronize()
        if dispatch_quant is None:
            expected_f = torch.zeros_like(x, dtype=torch.float32)
            x_f = x.float()
            for j in range(num_topk):
                weight_j = topk_weights[:, j].masked_fill(topk_idx[:, j] < 0, 0.0).view(-1, 1)
                expected_f = torch.addcmul(expected_f, x_f, weight_j)
            gdiff = validate_combine_output_mpi(
                v_out, expected_f.to(torch.bfloat16), comm, exact=args.combine_mode == "direct_send"
            )
            if rank == 0:
                print(f"[validate] mscclpp combine OK max|got-expected|={gdiff:.4e}", flush=True)
        else:
            assert torch.isfinite(v_out).all().item(), "FP8 LL combine produced NaN/Inf"
            if rank == 0:
                print("[validate] mscclpp FP8 combine finite OK", flush=True)

    state = {"moe": moe_comm, "obuf": output_buffer, "out": out, "grp": ep_group}

    if args.cuda_graph:
        # Prime once, then capture dispatch and combine as two separate CUDA graphs (rather than a
        # single graph wrapping a dispatch()+combine() loop). The shared timed loop replays and
        # times each phase independently so the kineto collector can attribute per-phase
        # dispatch-vs-combine kernel time; a single combined graph would fuse both phases into one
        # replay and lose that per-phase breakdown, which is the primary output of this benchmark.
        prime_out, prime_handle = _dispatch()
        _combine(prime_out, prime_handle)
        torch.cuda.synchronize()

        g_dispatch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_dispatch):
            g_dispatch_out, g_handle = moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)
        g_combine = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_combine):
            g_combine_in = expert_output if use_rank_major else simulated_gemm_output(g_dispatch_out)
            moe_comm.combine(g_combine_in, g_handle, out=out)
        state["graphs"] = (g_dispatch, g_combine)

        def dispatch_fn():
            g_dispatch.replay()
            return (g_dispatch_out, g_handle)

        def combine_fn(dout):
            g_combine.replay()

    else:

        def dispatch_fn():
            return _dispatch()

        def combine_fn(dout):
            dispatch_out, handle = dout
            _combine(dispatch_out, handle)

    def teardown():
        state.clear()
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown
