# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import os
import torch

from .bench_common import _ensure_torch_dist, sum_matching_kernel_us


def parse_kineto_kernels(key_averages):
    """Map a kineto key_averages() table to (dispatch_us, combine_us) for DeepEP
    V2.5. Its path runs a dispatch_impl kernel + a copy epilogue and a
    combine_impl kernel + a reduce epilogue; every one carries the phase word in
    its function name, so a case-insensitive substring match sums each phase."""
    return (
        sum_matching_kernel_us(key_averages, ("dispatch",)),
        sum_matching_kernel_us(key_averages, ("combine",)),
    )


# ============================================================================
# Backend: DeepEP V2.5 (deepseek-ai/DeepEP, EPBuffer).
# ============================================================================
def setup_deepep(args, comm, rank, num_ranks, inputs):
    """DeepEP V2.5 dispatch/combine via `deep_ep.EPBuffer`, wired
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
    if args.ep_layout == "token_major":
        raise ValueError("DeepEP EPBuffer supports rank_major or expert_major layout")

    # DeepEP EPBuffer supports local pre-reduction (allow_multiple_reduction=True,
    # corresponding to rank_local_reduce) or single combine-epilogue reduction
    # (allow_multiple_reduction=False, corresponding to direct_send).
    allow_multiple_reduction = int(args.combine_mode != "direct_send")
    raw_disp_dtype = args.dispatch_dtype
    use_fp8_dispatch = raw_disp_dtype != "bf16"
    if use_fp8_dispatch and args.fp8_input_mode == "quantize" and rank == 0:
        print(
            "[cfg] deepep EPBuffer does not fuse dispatch quantization; pre-quantizing input outside the timed loop",
            flush=True,
        )

    if rank == 0:
        print(
            f"[cfg] backend=deepep EPBuffer num_sms={args.num_sms} "
            f"combine_mode={args.combine_mode} dispatch_dtype={raw_disp_dtype} "
            f"num_ranks={num_ranks} tokens/rank={num_tokens} "
            f"hidden={hidden} num_experts={num_experts} top_k={num_topk} "
            f"warmup={args.num_warmup} iters={args.num_iters}",
            flush=True,
        )

    buffer = deep_ep.EPBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
        allow_hybrid_mode=0,  # NVLink-only bench (same-rack MNNVL, EP_DISABLE_GIN=1); hybrid RDMA+NVLink tier unused
        allow_multiple_reduction=allow_multiple_reduction,
        explicitly_destroy=True,
    )

    # Received-token layout: rank_major (plain, do_expand=False -- the deepep default)
    # or expert_major (do_expand=True -- the expanding layout, one slot per expert per
    # token). When --ep-layout is omitted, the plain rank-major layout is kept.
    do_expand = args.ep_layout == "expert_major"
    if rank == 0:
        _lay = "expert_major (do_expand)" if do_expand else "rank_major (plain)"
        print(f"[cfg] deepep ep_layout={args.ep_layout or 'default'} -> {_lay}", flush=True)

    dispatch_x = x
    if raw_disp_dtype in ("fp8_e4m3_static", "fp8_e4m3") and not do_expand:
        # On rank_major, --dispatch-dtype fp8_e4m3 maps to static FP8 (no scale sideband).
        dispatch_x = (x.float() * float(args.fp8_scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    elif raw_disp_dtype == "fp8_e4m3":
        from deep_ep.utils.math import per_token_cast_to_fp8

        dispatch_x = per_token_cast_to_fp8(x)
    elif raw_disp_dtype == "mxfp8_e4m3":
        if hidden % 128 != 0:
            raise ValueError("DeepEP packed MXFP8 scale factors require hidden to be a multiple of 128")
        x_blocks = x.float().view(num_tokens, hidden // 32, 32)
        amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-4)
        exp = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32)
        e8m0 = (exp + 127).clamp(1, 254).to(torch.uint8)
        inv_scale = torch.ldexp(torch.ones_like(amax), -exp)
        x_fp8 = (x_blocks * inv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(num_tokens, hidden).contiguous()
        sf_packed = e8m0.contiguous().view(torch.int32)
        dispatch_x = (x_fp8, sf_packed)

    # DeepEP dispatch args (non-cached so dispatch_impl + copy epilogue run).
    dispatch_args = dict(
        x=dispatch_x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_tokens,
        do_cpu_sync=True,
        do_expand=do_expand,
        do_zero_padding=do_expand,
        num_sms=args.num_sms,
    )

    # Prime once to obtain the handle (routing) and the received-token count so we
    # can size the combine input. Build a realistic BF16 combine input in the
    # received layout (the role of `simulated_gemm_output`): random values only in
    # the valid received-token slots, matching test_ep.py's `input_for_combine`.
    #
    # NOTE: the PLAIN (rank-major) path uses the per-expert combine; the expanded
    # (expert-major) path runs DeepEP's reduce/expand combine, which processes more
    # rows in the reduce epilogue. Both are exercised via --ep-layout.
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
        topk_weights=None if (do_expand and not allow_multiple_reduction) else recv_topk_weights,
        handle=handle,
        num_sms=args.num_sms,
    )

    # DeepEP native GPU barrier (comm-stream, sequential) -- aligns its dispatch/
    # combine recv-spin far more tightly than a generic all_reduce; this is what
    # DeepEP's own test_ep.py uses for kineto profiling.
    def deepep_barrier():
        buffer.barrier(wait_comm_stream=True, with_cpu_sync=False, sequential=True)

    # DeepEP CUDA-graph capture requires the NVLink/MNNVL path (EP_DISABLE_GIN=1)
    # and cached dispatch (do_cpu_sync=False); the RDMA/IB GIN path runs eagerly.
    gin_disabled = os.environ.get("EP_DISABLE_GIN", "0") == "1"
    deepep_can_graph = args.cuda_graph and gin_disabled

    if args.cuda_graph and not deepep_can_graph and rank == 0:
        print(
            "[cfg] deepep cuda_graph requested but disabled: RDMA/IB scale-out (GIN) is "
            "not graph-capturable (symmetric-memory); set EP_DISABLE_GIN=1 for the "
            "NVLink/MNNVL path, or run eager dispatch/combine",
            flush=True,
        )

    def combine_fn(_dout):
        buffer.combine(**combine_args)

    graph_spec = None
    if deepep_can_graph:
        if rank == 0:
            print("[cfg] deepep cuda_graph=True (single graph, cached dispatch, do_cpu_sync=False)", flush=True)
        # The non-cached dispatch above did a CPU sync to size the layout; that is
        # illegal inside a CUDA graph. Use the CACHED dispatch instead: pass the
        # primed handle (topk_idx reused from it), which forces do_cpu_sync=False
        # and skips the host-side count read, leaving a pure on-stream kernel launch
        # -- capture-safe, and also runnable eagerly if the harness capture fails.
        cached_dispatch_args = dict(
            x=dispatch_x,
            topk_weights=topk_weights,
            handle=handle,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
            do_expand=do_expand,
            do_zero_padding=do_expand,
            num_sms=args.num_sms,
        )

        def dispatch_fn():
            buffer.dispatch(**cached_dispatch_args)
            return None

        # Capture-safe ops handed to the harness (it owns the graph capture).
        graph_spec = {
            "dispatch": lambda: buffer.dispatch(**cached_dispatch_args),
            "combine": lambda: buffer.combine(**combine_args),
        }
    else:

        def dispatch_fn():
            buffer.dispatch(**dispatch_args)
            return None

    def teardown():
        nonlocal buffer
        buffer.destroy()
        buffer = None
        deep_ep.destroy_all_managed_nccl_comm()

    effective_layout = "expert_major" if do_expand else "rank_major"
    if not use_fp8_dispatch:
        effective_disp = "BF16"
    elif raw_disp_dtype == "mxfp8_e4m3":
        effective_disp = "MXFP8 E4M3 (prequantized)"
    elif not do_expand:
        effective_disp = "FP8 E4M3 Static (prequantized)"
    else:
        effective_disp = "FP8 E4M3 (prequantized)"

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": deepep_barrier,
        "graph": graph_spec,
        "summary_meta": {
            "algorithm": "EPBuffer",
            "layout": effective_layout,
            "dispatch_format": effective_disp,
            "combine_format": "BF16",
            "combine_mode": "rank_local_reduce" if allow_multiple_reduction else "direct_send",
            "execution": "cuda_graph" if graph_spec is not None else "eager",
        },
    }
