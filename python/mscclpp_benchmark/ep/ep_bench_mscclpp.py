# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from .ep_bench_common import (
    simulated_gemm_output,
    sum_matching_kernel_us,
    validate_combine_output_mpi,
)


def _sum_separate_quantization_us(key_averages):
    operator_names = {"aten::mul", "aten::copy_"}
    operator_us = sum(
        float(event.self_device_time_total) / int(event.count)
        for event in key_averages
        if event.key in operator_names and int(event.count) > 0
    )
    if operator_us:
        return operator_us

    # CUDA graph replay omits the eager ATen scopes but retains the child kernel names.
    kernel_markers = ("MulFunctor<float>", "ConvertToFloat8E4M3fnOp<float>")
    return sum(
        float(event.self_device_time_total) / int(event.count)
        for event in key_averages
        if any(marker in event.key for marker in kernel_markers) and int(event.count) > 0
    )


def parse_kineto_kernels(key_averages, include_separate_quantization=False):
    dispatch_us = sum_matching_kernel_us(key_averages, ("dispatch", "synchronizepeers"))
    if include_separate_quantization:
        dispatch_us += _sum_separate_quantization_us(key_averages)
    return dispatch_us, sum_matching_kernel_us(key_averages, ("combine",))


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
    combine_input = dispatch_out.combine_input_buffer
    if combine_input.dtype == torch.float8_e4m3fn:
        scale = dispatch_out.quant.scale
        if scale is None:
            raise RuntimeError("rank-major FP8 expert output requires the static-FP8 dispatch scale")
        expert_output = (expert_output * scale).to(combine_input.dtype)
    else:
        expert_output = expert_output.to(torch.bfloat16)
    combine_input.copy_(expert_output)
    return dispatch_out.combine_input_buffer


def _expected_output(input_real, topk_idx, topk_weights, config, num_ranks):
    import mscclpp.ep as ep

    expected = torch.zeros_like(input_real, dtype=torch.float32)
    x_float = input_real.float()
    fp8_scale = (
        config.quant.scale
        if config.quant is not None and config.quant.combine_format == ep.DispatchDataType.FP8_E4M3
        else None
    )

    def stage(values):
        if fp8_scale is not None:
            return (values * fp8_scale).to(torch.float8_e4m3fn).float() * (1.0 / fp8_scale)
        return values.to(torch.bfloat16).float()

    if config.combine_mode == ep.CombineMode.DIRECT_SEND:
        for lane in range(config.topk):
            weight = topk_weights[:, lane].masked_fill(topk_idx[:, lane] < 0, 0)
            expected = torch.addcmul(expected, stage(x_float), weight[:, None])
    else:
        for destination in range(num_ranks):
            partial = torch.zeros_like(expected)
            for lane in range(config.topk):
                selected = (topk_idx[:, lane] >= 0) & (topk_idx[:, lane] // config.num_local_experts == destination)
                weight = topk_weights[:, lane].masked_fill(~selected, 0)
                partial = torch.addcmul(partial, x_float, weight[:, None])
            expected += stage(partial)
    return expected.to(torch.bfloat16)


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
    fp8 = args.dispatch_dtype == "fp8_e4m3"
    static_fp8 = fp8 and latency and layout == ep.DispatchLayout.RANK_MAJOR
    fused_static_fp8 = static_fp8 and args.static_fp8_input == "fused"
    separate_static_fp8 = static_fp8 and args.static_fp8_input == "separate"
    combine_fp8 = args.combine_dtype == "fp8_e4m3"
    quant = None
    if fp8 or combine_fp8:
        quant = ep.QuantConfig(
            format=ep.DispatchDataType.FP8_E4M3 if fp8 else ep.DispatchDataType.BF16,
            scale=args.static_fp8_scale if static_fp8 else None,
            input_mode=(
                ep.QuantConfig.InputMode.QUANTIZE
                if fused_static_fp8
                else ep.QuantConfig.InputMode.PREQUANTIZED
            ),
            combine_format=ep.DispatchDataType.FP8_E4M3 if combine_fp8 else ep.DispatchDataType.BF16,
        )
    dispatch_input = x
    input_real = x
    separate_quant_workspace = None
    separate_quant_scale = None
    if static_fp8:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        if x.float().abs().max().item() * args.static_fp8_scale > fp8_max:
            raise ValueError("--static-fp8-scale overflows the generated activations")
        quantized_reference = (x.float() * args.static_fp8_scale).to(torch.float8_e4m3fn)
        input_real = quantized_reference.float() * (1.0 / args.static_fp8_scale)
        if not fused_static_fp8:
            dispatch_input = quantized_reference.clone()
        if separate_static_fp8:
            separate_quant_workspace = torch.empty_like(x, dtype=torch.float32)
            separate_quant_scale = torch.tensor([args.static_fp8_scale], dtype=torch.float32, device=x.device)
            torch.mul(x, separate_quant_scale, out=separate_quant_workspace)
            dispatch_input.copy_(separate_quant_workspace)
            if not torch.equal(dispatch_input.view(torch.uint8), quantized_reference.view(torch.uint8)):
                raise AssertionError("separate static-FP8 quantization does not match the FP32 reference")
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
    output_buffer = moe.get_dispatch_output_buffer()
    out = torch.empty_like(x)

    def dispatch_fn():
        if separate_static_fp8:
            torch.mul(x, separate_quant_scale, out=separate_quant_workspace)
            dispatch_input.copy_(separate_quant_workspace)
        return moe.dispatch(
            dispatch_input,
            topk_idx,
            topk_weights,
            output_buffer=output_buffer,
            prepare_handle=preparation,
        )

    dispatch_out, handle = dispatch_fn()
    expert_output = _stage_expert_output(dispatch_out, config)
    moe.combine(expert_output, handle, out=out)
    if args.validate:
        torch.cuda.synchronize()
        if out.dtype != torch.bfloat16:
            raise AssertionError(f"MSCCL++ combine returned {out.dtype}, expected torch.bfloat16")
        expected = _expected_output(input_real, topk_idx, topk_weights, config, num_ranks)
        diff = validate_combine_output_mpi(
            out,
            expected,
            comm,
            exact=combine_mode == ep.CombineMode.DIRECT_SEND and quant is None,
        )
        if rank == 0:
            print(f"[validate] mscclpp combine OK max|got-expected|={diff:.4e}", flush=True)
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
        print(
            f"[cfg] backend=mscclpp algorithm={args.mode.upper()} layout={layout} "
            f"num_ranks={num_ranks} tokens/rank={args.num_tokens} hidden={args.hidden} "
            f"num_experts={args.num_experts} experts/rank={config.num_local_experts} top_k={args.num_topk} "
            f"num_blocks={moe.num_blocks} dispatch_dtype={args.dispatch_dtype} "
            f"static_fp8_scale={args.static_fp8_scale if static_fp8 else 'n/a'} "
            f"static_fp8_input={args.static_fp8_input if static_fp8 else 'n/a'} "
            f"combine_dtype={args.combine_dtype} combine_output=bf16",
            flush=True,
        )

    def teardown():
        nonlocal moe, preparation, output_buffer, expert_output, out, group, dispatch_input, input_real
        nonlocal separate_quant_workspace, separate_quant_scale
        captured.clear()
        preparation = output_buffer = expert_output = out = dispatch_input = input_real = None
        separate_quant_workspace = separate_quant_scale = None
        moe = group = None

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": None,
        "graph": graph_spec,
        "paired_profile": True,
        "parse_kernels": (
            lambda key_averages: parse_kineto_kernels(key_averages, include_separate_quantization=separate_static_fp8)
        ),
    }
