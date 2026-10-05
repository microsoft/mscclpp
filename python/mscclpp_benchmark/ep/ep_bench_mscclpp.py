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
        sum_matching_kernel_us(key_averages, ("dispatch", "synchronizepeers")),
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
    dispatch_out.combine_input_buffer.copy_(expert_output.to(torch.bfloat16))
    return dispatch_out.combine_input_buffer


def _expected_output(x, topk_idx, topk_weights, config, num_ranks):
    import mscclpp.ep as ep

    expected = torch.zeros_like(x, dtype=torch.float32)
    x_float = x.float()
    if config.combine_mode == ep.CombineMode.DIRECT_SEND:
        for lane in range(config.topk):
            weight = topk_weights[:, lane].masked_fill(topk_idx[:, lane] < 0, 0)
            expected = torch.addcmul(expected, x_float, weight[:, None])
    else:
        for destination in range(num_ranks):
            partial = torch.zeros_like(expected)
            for lane in range(config.topk):
                selected = (topk_idx[:, lane] >= 0) & (topk_idx[:, lane] // config.num_local_experts == destination)
                weight = topk_weights[:, lane].masked_fill(~selected, 0)
                partial = torch.addcmul(partial, x_float, weight[:, None])
            expected += partial.to(torch.bfloat16).float()
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
    quant = ep.QuantConfig(format=ep.DispatchDataType.FP8_E4M3) if args.dispatch_dtype == "fp8_e4m3" else None
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
        return moe.dispatch(x, topk_idx, topk_weights, output_buffer=output_buffer, prepare_handle=preparation)

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
        print(
            f"[cfg] backend=mscclpp algorithm={args.mode.upper()} layout={layout} "
            f"num_ranks={num_ranks} tokens/rank={args.num_tokens} hidden={args.hidden} "
            f"num_experts={args.num_experts} top_k={args.num_topk} "
            f"num_blocks={moe.num_blocks} dispatch_dtype={args.dispatch_dtype}",
            flush=True,
        )

    def teardown():
        nonlocal moe, preparation, output_buffer, expert_output, out, group
        captured.clear()
        preparation = output_buffer = expert_output = out = moe = group = None

    return {
        "dispatch": dispatch_fn,
        "combine": combine_fn,
        "teardown": teardown,
        "barrier": None,
        "graph": graph_spec,
        "paired_profile": True,
    }
