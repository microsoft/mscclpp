# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import os

import torch

from ep_bench_common import simulated_gemm_output, validate_combine_output_mpi


# ============================================================================
# Backend: mscclpp EP high-throughput (MoECommunicator, HIGH_THROUGHPUT mode).
# ============================================================================
def setup_mscclpp_ht(args, comm, rank, num_ranks, inputs):
    """mscclpp EP high-throughput dispatch/combine via `MoECommunicator` with
    `mode=MoEMode.HIGH_THROUGHPUT` (GB200 TMA, TOKEN_MAJOR), wired like the other
    backends: return (dispatch_fn, combine_fn, teardown). Follows the HT flow in
    test_intranode_multirank.py: an initial uncached dispatch records the routing
    layout on the handle, then the timed loop replays a cached dispatch
    (previous_handle=) + combine to isolate the on-GPU kernel cost."""
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    x, topk_idx, topk_weights, _ = inputs
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_sms = int(os.environ.get("MSCCLPP_EP_NUM_SMS", "20"))
    ep_layout = getattr(args, "ep_layout", None)
    output_layout = ep.DispatchLayout.RANK_MAJOR if ep_layout == "rank_major" else ep.DispatchLayout.TOKEN_MAJOR

    if rank == 0:
        print(
            f"[cfg] backend=mscclpp-ht algorithm=HIGH_THROUGHPUT layout={output_layout} num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} num_sms={num_sms} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )

    ep_group = CommGroup(mpi_comm=comm)
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.HIGH_THROUGHPUT,
        num_sms=num_sms,
        output_layout=output_layout,
    )
    assert moe_comm.is_available()
    if rank == 0:
        print(
            f"[cfg] mscclpp-ht MoECommunicator is_internode={moe_comm.is_internode()}",
            flush=True,
        )

    # Optional round-trip correctness check (mirrors the mscclpp LL --validate path). The
    # rank-major layout must reduce to the SAME combined output as the reference token-major
    # layout built from identical routing/input; validated bit-exact on GB200 (1/2/4 nodes).
    if getattr(args, "validate", False):
        v_dispatch_out, v_handle = moe_comm.dispatch(x, topk_idx, topk_weights)
        got = moe_comm.combine(simulated_gemm_output(v_dispatch_out), v_handle)
        torch.cuda.synchronize()
        assert torch.isfinite(got).all().item(), "mscclpp-ht combine produced NaN/Inf"
        if output_layout == ep.DispatchLayout.RANK_MAJOR:
            ref_comm = ep.MoECommunicator(
                comm=ep_group,
                num_experts=num_experts,
                hidden_size=hidden,
                topk=num_topk,
                max_tokens_per_rank=num_tokens,
                mode=ep.MoEMode.HIGH_THROUGHPUT,
                num_sms=num_sms,
                output_layout=ep.DispatchLayout.TOKEN_MAJOR,
            )
            r_dispatch_out, r_handle = ref_comm.dispatch(x, topk_idx, topk_weights)
            ref = ref_comm.combine(simulated_gemm_output(r_dispatch_out), r_handle)
            torch.cuda.synchronize()
            gdiff = validate_combine_output_mpi(got, ref, comm, exact=True)
            if rank == 0:
                print(
                    f"[validate] mscclpp-ht rank-major vs token-major ref bit-exact max|diff|={gdiff:.4e}",
                    flush=True,
                )
            del ref_comm
        elif rank == 0:
            print("[validate] mscclpp-ht token-major combine finite OK", flush=True)

    # One uncached dispatch to build the cached routing layout on the handle; the
    # timed loop reuses it via previous_handle to skip notify_dispatch's host wait
    # (isolates the on-GPU dispatch-kernel cost, NCCL-EP ep_bench convention).
    handle0 = moe_comm.dispatch(x, topk_idx, topk_weights)[1]

    def dispatch_fn():
        return moe_comm.dispatch(x, topk_idx, topk_weights, previous_handle=handle0)

    def combine_fn(dout):
        dispatch_out, handle = dout
        moe_comm.combine(dispatch_out.tokens, handle)

    _state = {"moe": moe_comm, "grp": ep_group}

    def teardown():
        _state.clear()
        gc.collect()
        torch.cuda.synchronize()

    return dispatch_fn, combine_fn, teardown
