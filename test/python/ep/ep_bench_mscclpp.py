# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import os
import torch

from ep_bench_common import (
    simulated_gemm_output,
    validate_combine_output_mpi,
    sum_matching_kernel_us,
)


def parse_kineto_kernels(key_averages):
    """Map a kineto key_averages() table to (dispatch_us, combine_us) for mscclpp
    LL, which runs a single dispatch kernel and a single combine kernel with the
    phase word in each function name. Template-arg stripping in the shared helper
    matters here because the combine kernel is templated on DispatchLayout
    (combineKernel<.., RANK_MAJOR>)."""
    return (
        sum_matching_kernel_us(key_averages, ("dispatch",)),
        sum_matching_kernel_us(key_averages, ("combine",)),
    )


# ============================================================================
# Backend: mscclpp EP (MoECommunicator).
# ============================================================================
def setup_mscclpp(args, comm, rank, num_ranks, inputs):
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    input_samples = inputs if isinstance(inputs, list) else [inputs]
    x, topk_idx, topk_weights, _ = input_samples[0]
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks

    num_rdma_bytes = 0  # not exposed by current mscclpp API; 0 over the CUDA-IPC path
    if rank == 0:
        print(
            f"[cfg] backend=mscclpp algorithm=LATENCY num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters} num_rdma_bytes={num_rdma_bytes}",
            flush=True,
        )

    bootstrap = os.environ.get("EP_MSCCLPP_BOOTSTRAP")
    ep_group = (
        CommGroup(interfaceIpPortTrio=bootstrap, rank=rank, size=num_ranks)
        if bootstrap
        else (CommGroup(torch_group=comm.torch_group) if hasattr(comm, "torch_group") else CommGroup(mpi_comm=comm))
    )
    combine_mode = {
        "rank_local_reduce": ep.CombineMode.RANK_LOCAL_REDUCE,
        "direct_send": ep.CombineMode.DIRECT_SEND,
    }[args.combine_mode]
    rank_major = args.ep_layout == "rank_major"
    if rank_major and combine_mode != ep.CombineMode.RANK_LOCAL_REDUCE:
        raise ValueError("rank-major output requires rank_local_reduce combine")
    output_layout = ep.DispatchLayout.RANK_MAJOR if rank_major else ep.DispatchLayout.EXPERT_MAJOR
    dispatch_quant = ep.QuantConfig(format=ep.DispatchDataType.FP8_E4M3) if args.dispatch_dtype == "fp8_e4m3" else None
    if rank_major and dispatch_quant is not None:
        raise ValueError("rank-major output supports BF16 dispatch only")
    dispatch_dtype = torch.float8_e4m3fn if dispatch_quant is not None else torch.bfloat16
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LATENCY,
        combine_mode=combine_mode,
        output_layout=output_layout,
        invalid_token_expert_id=num_experts,
        quant=dispatch_quant,
    )
    assert moe_comm.is_available()
    if rank == 0:
        print(
            f"[cfg] mscclpp MoECommunicator is_internode={moe_comm.is_internode()} "
            f"dispatch_dtype={args.dispatch_dtype} combine_mode={args.combine_mode} cuda_graph={args.cuda_graph}",
            flush=True,
        )
        print(f"[cfg] mscclpp output_layout={args.ep_layout or 'expert_major'}", flush=True)

    # Hoist output tensors out of the timed loop (the communicator owns its
    # src_info/layout_range/count buffers internally).
    output_buffer = (
        None
        if rank_major
        else torch.empty((num_local_experts, num_ranks * num_tokens, hidden), dtype=dispatch_dtype, device="cuda")
    )
    expert_output_initialized = False
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def _dispatch():
        # Full (send+recv) LL dispatch inline on the stream; returns (dispatch_out, handle).
        return moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)

    def _combine(dispatch_out, handle):
        nonlocal expert_output_initialized
        # Rank-major MoE writes directly into the runtime-owned registered output
        # buffer. Pre-fill it once to benchmark communication without timing a copy.
        combine_input = dispatch_out.combine_input_buffer
        if combine_input is None:
            combine_input = simulated_gemm_output(dispatch_out)
        elif not expert_output_initialized:
            combine_input.normal_()
            expert_output_initialized = True
        moe_comm.combine(combine_input, handle, out=out)

    # Optional one-time correctness check (mirrors test_latency_multirank).
    if args.validate:
        v_dispatch_out, v_handle = _dispatch()
        v_out = torch.empty_like(out)
        validation_input = simulated_gemm_output(v_dispatch_out)
        if v_dispatch_out.combine_input_buffer is not None:
            # Rank-major combine reads the runtime-owned registered buffer, so the
            # simulated expert output has to be staged into it first.
            v_dispatch_out.combine_input_buffer.copy_(validation_input)
            validation_input = v_dispatch_out.combine_input_buffer
        moe_comm.combine(validation_input, v_handle, out=v_out)
        torch.cuda.synchronize()
        if dispatch_quant is None:
            expected_f = torch.zeros_like(x, dtype=torch.float32)
            x_f = x.float()
            if combine_mode == ep.CombineMode.RANK_LOCAL_REDUCE:
                # Rank-local reduce rounds each destination rank's partial sum to
                # BF16 before the cross-rank accumulation.
                for destination_rank in range(num_ranks):
                    rank_partial = torch.zeros_like(x, dtype=torch.float32)
                    for j in range(num_topk):
                        selected = (topk_idx[:, j] >= 0) & (topk_idx[:, j] // num_local_experts == destination_rank)
                        weight_j = topk_weights[:, j].masked_fill(~selected, 0.0).view(-1, 1)
                        rank_partial = torch.addcmul(rank_partial, x_f, weight_j)
                    expected_f += rank_partial.to(torch.bfloat16).float()
            else:
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
                print("[validate] mscclpp combine finite OK", flush=True)

    state = {
        "moe": moe_comm,
        "inputs": input_samples,
        "obuf": output_buffer,
        "out": out,
        "grp": ep_group,
    }

    # Eager ops (also the fallback if the harness graph capture fails). dispatch
    # returns (dispatch_out, handle); combine consumes them.
    def dispatch_fn():
        return _dispatch()

    def combine_fn(dout):
        dispatch_out, handle = dout
        _combine(dispatch_out, handle)

    # Capture-safe ops for the harness's single-graph capture: dispatch+combine run
    # as ONE graph (one replay does both, so combine_fn becomes a no-op), matching
    # the NCCL-EP / DeepEP path and how a real serving stack replays a fused MoE
    # step. Per-kernel kineto times are unaffected by the single vs two-graph
    # choice; only the host-level per-phase split changes (combine host timer folds
    # into dispatch). combine consumes the dispatch output produced in the same
    # capture, shared via the _cap holder.
    graph_spec = None
    if args.cuda_graph:
        _cap = {}

        def _graph_dispatch():
            _cap["out"], _cap["handle"] = moe_comm.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer)

        def _graph_combine():
            moe_comm.combine(simulated_gemm_output(_cap["out"]), _cap["handle"], out=out)

        graph_spec = {
            "dispatch": _graph_dispatch,
            "combine": _graph_combine,
            "pre_replay": None,
            "on_fail": None,
        }

    def teardown():
        state.clear()
        gc.collect()
        torch.cuda.synchronize()

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": None,
        "graph": graph_spec,
    }
