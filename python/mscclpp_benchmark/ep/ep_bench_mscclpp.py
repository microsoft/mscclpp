# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from .ep_bench_common import (
    simulated_gemm_output,
    sum_matching_kernel_us,
    validate_combine_output_mpi,
)


def parse_kineto_kernels(key_averages):
    return (
        sum_matching_kernel_us(key_averages, ("dispatch", "synchronizethroughputpeers")),
        sum_matching_kernel_us(key_averages, ("combine",)),
    )


def _stage_expert_output(dispatch_out, config):
    import mscclpp.ep as ep

    tokens = simulated_gemm_output(dispatch_out).float()
    counts = dispatch_out.layout.num_tokens_per_rank
    if counts is None:
        counts = dispatch_out.layout.num_tokens_per_expert
    if config.output_layout != ep.DispatchLayout.TOKEN_MAJOR:
        rows = torch.arange(tokens.shape[-2], device=tokens.device)
        tokens = torch.where((rows[None, :] < counts[:, None])[..., None], tokens, 0)
    if config.output_layout == ep.DispatchLayout.EXPERT_MAJOR:
        return tokens.to(torch.bfloat16).contiguous()

    ids = dispatch_out.topk_ids
    valid = (ids >= 0) & (ids < (config.num_experts if config.mode == ep.MoEMode.LATENCY else config.num_local_experts))
    if config.combine_mode == ep.CombineMode.DIRECT_SEND:
        expert_output = torch.where(valid[..., None], tokens[..., None, :], 0)
    else:
        expert_output = torch.zeros_like(tokens)
        for lane in range(config.topk):
            weight = dispatch_out.weights[..., lane].masked_fill(~valid[..., lane], 0)
            expert_output = torch.addcmul(expert_output, tokens, weight[..., None])
    quant = dispatch_out.quant
    if quant is not None and quant.combine_format == ep.DispatchDataType.FP8_E4M3_STATIC:
        dispatch_out.combine_input_buffer.copy_((expert_output * quant.scale).to(torch.float8_e4m3fn))
    else:
        dispatch_out.combine_input_buffer.copy_(expert_output.to(torch.bfloat16))
    return dispatch_out.combine_input_buffer


def _expected_output(x, topk_idx, topk_weights, config, num_ranks):
    import mscclpp.ep as ep

    expected = torch.zeros_like(x, dtype=torch.float32)
    x_float = x.float()
    quant = config.quant
    static_scale = quant.scale if quant is not None and quant.format == ep.DispatchDataType.FP8_E4M3_STATIC else None
    combine_fp8 = quant is not None and quant.combine_format == ep.DispatchDataType.FP8_E4M3_STATIC
    if static_scale is not None:
        x_float = (x_float * static_scale).to(torch.float8_e4m3fn).float() / static_scale

    def stage(values):
        if combine_fp8:
            return (values * static_scale).to(torch.float8_e4m3fn).float() / static_scale
        return values.to(torch.bfloat16).float()

    if config.combine_mode == ep.CombineMode.DIRECT_SEND:
        staged = stage(x_float) if static_scale is not None else x_float
        for lane in range(config.topk):
            weight = topk_weights[:, lane].masked_fill(topk_idx[:, lane] < 0, 0)
            expected = torch.addcmul(expected, staged, weight[:, None])
    else:
        for destination in range(num_ranks):
            partial = torch.zeros_like(expected)
            for lane in range(config.topk):
                selected = (topk_idx[:, lane] >= 0) & (topk_idx[:, lane] // config.num_local_experts == destination)
                weight = topk_weights[:, lane].masked_fill(~selected, 0)
                partial = torch.addcmul(partial, x_float, weight[:, None])
            expected += stage(partial)
    return expected.to(torch.bfloat16)


def _build_quant_config(args, mode, layout):
    import mscclpp.ep as ep

    combine_dtype = getattr(args, "combine_dtype", "bf16")
    combine_format = (
        ep.DispatchDataType.FP8_E4M3_STATIC if combine_dtype == "fp8_e4m3_static" else ep.DispatchDataType.BF16
    )
    dtype = getattr(args, "dispatch_dtype", "bf16")
    if dtype == "bf16":
        if combine_format != ep.DispatchDataType.BF16:
            raise ValueError("--combine-dtype fp8_e4m3_static requires static FP8 dispatch")
        return None
    if dtype == "mxfp8_e4m3":
        if combine_format != ep.DispatchDataType.BF16:
            raise ValueError("--combine-dtype fp8_e4m3_static requires static FP8 dispatch")
        return ep.QuantConfig(format=ep.DispatchDataType.MXFP8_E4M3)
    if dtype == "fp8_e4m3_static" or (
        dtype == "fp8_e4m3" and mode == ep.MoEMode.LATENCY and layout == ep.DispatchLayout.RANK_MAJOR
    ):
        input_mode = (
            ep.QuantConfig.InputMode.PREQUANTIZED
            if getattr(args, "fp8_input_mode", "quantize") == "prequantized"
            else ep.QuantConfig.InputMode.QUANTIZE
        )
        return ep.QuantConfig(
            format=ep.DispatchDataType.FP8_E4M3_STATIC,
            scale=float(getattr(args, "fp8_scale", 1.0)),
            input_mode=input_mode,
            combine_format=combine_format,
        )
    if combine_format != ep.DispatchDataType.BF16:
        raise ValueError("--combine-dtype fp8_e4m3_static requires static FP8 dispatch")
    return ep.QuantConfig(format=ep.DispatchDataType.FP8_E4M3)


def setup_mscclpp(args, comm, rank, num_ranks, inputs):
    from mscclpp import CommGroup
    import mscclpp.ep as ep

    x, topk_idx, topk_weights, _ = inputs
    latency = args.mode == "latency"
    if args.cuda_graph and latency and args.iters_per_graph < 2:
        raise ValueError("MSCCL++ latency graph replay requires --iters-per-graph >= 2 to avoid reusing one epoch")
    mode = ep.MoEMode.LATENCY if latency else ep.MoEMode.THROUGHPUT
    layout = {
        "expert_major": ep.DispatchLayout.EXPERT_MAJOR,
        "rank_major": ep.DispatchLayout.RANK_MAJOR,
        "token_major": ep.DispatchLayout.TOKEN_MAJOR,
    }.get(args.ep_layout, ep.DispatchLayout.EXPERT_MAJOR if latency else ep.DispatchLayout.TOKEN_MAJOR)
    combine_mode = (
        ep.CombineMode.DIRECT_SEND if args.combine_mode == "direct_send" else ep.CombineMode.RANK_LOCAL_REDUCE
    )
    quant = _build_quant_config(args, mode, layout)
    group = CommGroup(mpi_comm=comm)
    config = ep.MoECommunicatorConfig(
        comm=group,
        num_experts=args.num_experts,
        hidden_size=args.hidden,
        topk=args.num_topk,
        max_tokens_per_rank=args.num_tokens,
        mode=mode,
        num_blocks=args.num_sms or None,
        combine_mode=combine_mode,
        output_layout=layout,
        quant=quant,
    )
    moe = ep.MoECommunicator(config)
    if not moe.is_available():
        raise RuntimeError("MSCCL++ EP is unavailable for this topology or capacity")
    moe.initialize()
    preparation = None
    if not latency:
        output_count = torch.empty(
            num_ranks if layout == ep.DispatchLayout.RANK_MAJOR else config.num_local_experts,
            dtype=torch.int32,
            device=x.device,
        )
        preparation = moe.prepare(topk_idx, output_count=output_count)
    dispatch_input = x
    dispatch_quant = None
    if not latency and quant is not None and quant.format == ep.DispatchDataType.FP8_E4M3:
        if args.hidden % 128 != 0:
            raise ValueError("MSCCL++ throughput FP8 dispatch requires hidden to be a multiple of 128")
        x_blocks = x.float().view(args.num_tokens, args.hidden // 128, 128)
        amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-4)
        dispatch_input = (
            (x_blocks * (448.0 / amax.unsqueeze(-1)))
            .to(torch.float8_e4m3fn)
            .view(args.num_tokens, args.hidden)
            .contiguous()
        )
        dispatch_quant = ep.QuantConfig(
            format=ep.DispatchDataType.FP8_E4M3,
            block_scales=(amax / 448.0).contiguous(),
        )
    elif (
        quant is not None
        and quant.format == ep.DispatchDataType.FP8_E4M3_STATIC
        and quant.input_mode == ep.QuantConfig.InputMode.PREQUANTIZED
    ):
        dispatch_input = (x.float() * quant.scale).to(torch.float8_e4m3fn)
    output_buffer = moe.get_dispatch_output_buffer()
    out = torch.empty_like(x)

    def dispatch_fn():
        return moe.dispatch(
            dispatch_input,
            topk_idx,
            topk_weights,
            quant=dispatch_quant,
            output_buffer=output_buffer,
            prepare_handle=preparation,
        )

    dispatch_out, handle = dispatch_fn()
    expert_output = _stage_expert_output(dispatch_out, config)
    moe.combine(expert_output, handle, out=out)
    if args.validate:
        torch.cuda.synchronize()
        expected = _expected_output(x, topk_idx, topk_weights, config, num_ranks)
        diff = validate_combine_output_mpi(
            out,
            expected,
            comm,
            exact=combine_mode == ep.CombineMode.DIRECT_SEND and quant is None,
        )
        if rank == 0:
            print(f"[validate] mscclpp combine OK max|got-expected|={diff:.4e}", flush=True)
    # Fixed routing lets communication-only timing reuse synthetic expert output.
    if expert_output.dtype == torch.float8_e4m3fn:
        expert_output.copy_(
            torch.empty(expert_output.shape, dtype=torch.bfloat16, device=expert_output.device)
            .normal_(0.0, 0.1)
            .to(torch.float8_e4m3fn)
        )
    else:
        expert_output.normal_(0.0, 0.1)
    del dispatch_out, handle

    def combine_fn(dout):
        moe.combine(expert_output, dout[1], out=out)

    graph_spec = None
    captured = {}
    if args.cuda_graph:

        def graph_dispatch():
            captured["dout"] = dispatch_fn()

        def graph_combine():
            combine_fn(captured["dout"])

        graph_spec = {"dispatch": graph_dispatch, "combine": graph_combine}

    if rank == 0:
        combine_dtype = getattr(args, "combine_dtype", "bf16")
        print(
            f"[cfg] backend=mscclpp algorithm={args.mode.upper()} layout={layout} "
            f"combine_mode={combine_mode} num_ranks={num_ranks} tokens/rank={args.num_tokens} "
            f"hidden={args.hidden} num_experts={args.num_experts} top_k={args.num_topk} "
            f"num_blocks={moe.num_blocks} dispatch_dtype={args.dispatch_dtype} combine_dtype={combine_dtype}",
            flush=True,
        )

    def teardown():
        nonlocal moe, preparation, dispatch_input, dispatch_quant, output_buffer, expert_output, out, group
        captured.clear()
        preparation = dispatch_input = dispatch_quant = output_buffer = expert_output = out = moe = group = None

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": None,
        "graph": graph_spec,
        "paired_profile": True,
    }
