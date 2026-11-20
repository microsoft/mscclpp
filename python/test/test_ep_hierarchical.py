# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Correctness coverage for NVLink/IPC + GPUNetIO latency RANK_MAJOR.

Covers RANK_LOCAL_REDUCE within an IPC domain and across domains,
including chunked dispatch, gateway distribution, and FP32 aggregate combine.
Example (two nodes, four GPUs each)::

    mpirun -np 8 --map-by ppr:4:node \\
        python3 -m pytest python/test/test_ep_hierarchical.py -vx
"""

from contextlib import contextmanager
import importlib.util

from mpi4py import MPI
import pytest

torch = pytest.importorskip("torch")
if importlib.util.find_spec("mscclpp.mscclpp_ep_cpp") is None:
    pytest.skip("The CUDA EP extension was not built", allow_module_level=True)

from mscclpp import CommGroup
from mscclpp.ep import (
    CombineMode,
    DispatchDataType,
    DispatchLayout,
    MoECommunicator,
    MoECommunicatorConfig,
    MoEMode,
    QuantConfig,
)
from mscclpp.gpu import current_device

NUM_TOPK = 8
NUM_LOCAL_EXPERTS = 16


@pytest.fixture(scope="module")
def ep_group():
    if not torch.cuda.is_available():
        pytest.skip("EP requires CUDA")
    torch.cuda.set_device(current_device())
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("EP requires SM90 or newer")
    group = CommGroup(mpi_comm=MPI.COMM_WORLD)
    if not 1 <= group.nranks <= 128:
        pytest.skip("EP requires 1-128 ranks")
    yield group
    torch.cuda.synchronize()
    group.barrier()


@contextmanager
def initialized_runtime(group, **kwargs):
    device = torch.device("cuda", torch.cuda.current_device())
    runtime = MoECommunicator(MoECommunicatorConfig(comm=group, device=device, **kwargs))
    assert runtime.is_available()
    runtime.initialize()
    assert runtime.is_initialized()
    try:
        yield runtime
    finally:
        torch.cuda.synchronize()
        group.barrier()


def token_routes(rank, world_size, token, num_tokens, topk=NUM_TOPK):
    if token == num_tokens - 1:
        return [-1] * topk
    num_experts = NUM_LOCAL_EXPERTS * world_size
    base = (rank * NUM_LOCAL_EXPERTS + token * 7) % num_experts
    routes = [(base + slot * 5) % num_experts for slot in range(topk)]
    if token == 1:
        routes[1] = -1
        if topk > 8:
            routes[-2] = -1
    return routes


def collective_check(check):
    """Run ``check`` on every rank and fail on all ranks if any rank failed."""
    error = None
    try:
        check()
    except (Exception, pytest.fail.Exception) as exc:
        error = f"{type(exc).__name__}: {exc}"
    failed = {rank: e for rank, e in enumerate(MPI.COMM_WORLD.allgather(error)) if e is not None}
    if failed:
        pytest.fail(f"Validation failed on ranks {sorted(failed)}: {failed}")


def token_values(rank, num_tokens, scale, topk=NUM_TOPK):
    return (((torch.arange(num_tokens, device="cuda") + rank * num_tokens) % 15 + 1).float() * topk) * scale


def _dequantize_dispatch(result):
    quant = result.quant
    if quant is None:
        return result.tokens.float()
    if quant.format == DispatchDataType.MXFP8_E4M3:
        scales = torch.ldexp(
            torch.ones_like(quant.block_scales, dtype=torch.float32),
            quant.block_scales.to(torch.int32) - 127,
        )
        return result.tokens.float() * scales.repeat_interleave(32, dim=-1)
    dispatch_scale = quant.scale[0] if isinstance(quant.scale, tuple) else quant.scale
    return result.tokens.float() / dispatch_scale


def _stage_identity_expert(result, active_capacity, combine_mode=CombineMode.RANK_LOCAL_REDUCE):
    valid_rows = torch.arange(active_capacity, device="cuda")[None, :] < result.layout.num_tokens_per_rank[:, None]
    tokens = _dequantize_dispatch(result)
    if combine_mode == CombineMode.DIRECT_SEND:
        valid_slots = valid_rows[..., None] & (result.weights != 0)
        expert_out = torch.where(valid_slots[..., None], tokens[:, :, None, :], 0.0)
    else:
        local_weights = torch.where(valid_rows[..., None], result.weights, 0.0).sum(dim=-1, keepdim=True)
        expert_out = torch.where(valid_rows[..., None], tokens * local_weights, 0.0)
    quant = result.quant
    if quant is not None and quant.combine_format == DispatchDataType.FP8_E4M3_STATIC:
        combine_scale = quant.scale[1] if isinstance(quant.scale, tuple) else quant.scale
        result.combine_input_buffer.copy_((expert_out * combine_scale).to(torch.float8_e4m3fn))
    else:
        result.combine_input_buffer.copy_(expert_out.to(torch.bfloat16))
    return result.combine_input_buffer


def _check_dispatch_metadata(ep_group, result, num_tokens, topk=NUM_TOPK):
    expected_counts = []
    for source in range(ep_group.nranks):
        routes = [token_routes(source, ep_group.nranks, token, num_tokens, topk) for token in range(num_tokens)]
        local = [[e >= 0 and e // NUM_LOCAL_EXPERTS == ep_group.my_rank for e in r] for r in routes]
        expected_counts.append(sum(any(row) for row in local))
    assert result.layout.num_tokens_per_rank.cpu().tolist() == expected_counts


def _reference_mxfp8_quantize(bf16_input):
    blocks = bf16_input.float().reshape(bf16_input.shape[0], bf16_input.shape[1] // 32, 32)
    maximum = blocks.abs().amax(dim=-1)
    bits = maximum.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    scale = torch.clamp(exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32), 1, 254)
    scale = torch.where(maximum == 0, 127, torch.where(exponent == 0xFF, 254, scale)).to(torch.uint8)
    inv_bits = torch.where(scale == 254, 0x00400000, (254 - scale.to(torch.int32)) << 23).to(torch.int32)
    quantized = (blocks * inv_bits.view(torch.float32)[..., None]).to(torch.float8_e4m3fn).reshape_as(bf16_input)
    dequantized = quantized.float() * torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127
    ).repeat_interleave(32, dim=-1)
    return quantized, scale, dequantized


def _expected_hierarchical_combine(
    real_input,
    routes,
    weights,
    ep_group,
    combine_mode,
    *,
    combine_scale=None,
    inter_domain_dtype=torch.float32,
):
    valid_routes = routes >= 0
    peers = torch.div(routes, NUM_LOCAL_EXPERTS, rounding_mode="floor")
    domain = ep_group.nranks_per_ipc_domain
    my_domain = ep_group.my_rank // domain
    local_sum = torch.zeros_like(real_input, dtype=torch.float32)
    remote_sum = torch.zeros_like(real_input, dtype=torch.float32)

    def stage(values):
        if combine_scale is not None:
            return (values * combine_scale).to(torch.float8_e4m3fn).float() / combine_scale
        return values.to(torch.bfloat16).float()

    for lane in range(routes.shape[1]):
        active = valid_routes[:, lane]
        peer = peers[:, lane]
        if combine_mode == CombineMode.DIRECT_SEND:
            contrib = torch.where(active[:, None], stage(real_input), 0.0) * weights[:, lane : lane + 1]
        else:
            for prev in range(lane):
                active = active & ~(valid_routes[:, prev] & (peers[:, prev] == peer))
            same_peer = valid_routes & (peers == peer[:, None])
            rank_weight = torch.where(same_peer, weights, 0.0).sum(dim=-1, keepdim=True)
            contrib = torch.where(active[:, None], stage(real_input * rank_weight), 0.0)
        is_local = (peer // domain) == my_domain
        local_sum = local_sum + torch.where(is_local[:, None], contrib, 0.0)
        remote_sum = remote_sum + torch.where(~is_local[:, None], contrib, 0.0)
    if inter_domain_dtype == torch.bfloat16 and domain < ep_group.nranks:
        remote_sum = remote_sum.to(torch.bfloat16).float()
    return (local_sum + remote_sum).to(torch.bfloat16).float()


def _capture_graph(ep_group, stream, op):
    ep_group.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = op()
    stream.synchronize()
    ep_group.barrier()
    return graph, out


@pytest.mark.parametrize("qps_per_peer,chunk_rows,hidden", [(1, 7, 2048), (2, 8, 7168), (4, 8, 7168), (4, 24, 7168)])
@pytest.mark.parametrize("emit_local_expert_ids", [False, True])
@pytest.mark.parametrize("combine_mode", [CombineMode.RANK_LOCAL_REDUCE, CombineMode.DIRECT_SEND])
def test_latency_hierarchical_send_distribute(
    ep_group, qps_per_peer, chunk_rows, hidden, combine_mode, emit_local_expert_ids
):
    """Single-copy RDMA chunks, gateway-excluding routes, and credit reuse across changing graph rounds."""
    domain = ep_group.nranks_per_ipc_domain
    if not (ep_group.nranks == 2 * domain and domain >= 2):
        pytest.skip("Requires two equal NVLink domains with at least two ranks each")
    capacity, invalid_expert = 272, NUM_LOCAL_EXPERTS * ep_group.nranks

    def routes_for(source, tokens, active, rotation, routing="mixed"):
        local_base = source // domain * domain
        remote_base = domain - local_base
        gateway = remote_base + source % domain
        peers = [p for p in range(remote_base, remote_base + domain) if p != gateway]
        if routing == "uneven":
            active = (0, active // 3, active)[source % 3]
        rows = []
        for t in range(tokens):
            if t >= active:
                rows.append([source * NUM_LOCAL_EXPERTS] + [-1] * 7 if t == active and routing == "mixed" else [-1] * 8)
                continue
            a, b, c = [peers[(source + t + rotation + s) % len(peers)] for s in range(3)]
            ranks = (
                [a, a, b, c, source, local_base + (source + 1) % domain, a]
                if routing == "mixed"
                else [a, a, b, c, a, b, a]
            )
            experts = [p * NUM_LOCAL_EXPERTS + (t + slot) % NUM_LOCAL_EXPERTS for slot, p in enumerate(ranks)]
            experts[1] = experts[0]
            rows.append(experts + [-1])
        return rows

    def inputs_for(source, tokens, scale):
        t = torch.arange(tokens, device="cuda")
        vals = (((t + source) % 7 + 1)[:, None] * 8 * (torch.arange(hidden, device="cuda") % 3 + 1)[None, :]).float()
        vals[:, 0], vals[:, 1] = t % 33 + 1, t // 33 + 1
        return (vals * scale).to(torch.bfloat16)

    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR,
        combine_mode=combine_mode,
        num_experts=invalid_expert,
        hidden_size=hidden,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
        invalid_token_expert_id=invalid_expert,
        emit_local_expert_ids=emit_local_expert_ids,
        qps_per_peer=qps_per_peer,
        n_tokens_per_chunk=chunk_rows,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for step, tokens in enumerate(sorted({0, 1, chunk_rows - 1, chunk_rows, chunk_rows + 1, 33, 257})):
                active_cap = max(tokens, 1) if step % 2 else capacity
                routes = torch.tensor(
                    routes_for(ep_group.my_rank, tokens, tokens, step), dtype=torch.int64, device="cuda"
                ).reshape(tokens, NUM_TOPK)
                weights = torch.full((tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
                inp = inputs_for(ep_group.my_rank, tokens, 1.0)
                result, handle = runtime.dispatch(inp, routes, weights, runtime_max_tokens_per_rank=active_cap)

                def check_dispatch():
                    stream.synchronize()
                    counts = result.layout.num_tokens_per_rank.cpu().tolist()
                    for src, count in enumerate(counts):
                        exp_routes = routes_for(src, tokens, tokens, step)
                        routed = [
                            t
                            for t, row in enumerate(exp_routes)
                            if any(e >= 0 and e // NUM_LOCAL_EXPERTS == ep_group.my_rank for e in row)
                        ]
                        assert count == len(routed)
                        ids, w = result.topk_ids[src].cpu(), result.weights[src].cpu()
                        row_ids = result.tokens[src, :count, :2].cpu().int() - 1
                        rows = (row_ids[:, 0] + 33 * row_ids[:, 1]).tolist()
                        assert sorted(rows) == routed
                        if rows:
                            torch.testing.assert_close(
                                result.tokens[src, :count], inputs_for(src, tokens, 1.0)[rows], rtol=0, atol=0
                            )
                        for idx, t in enumerate(rows):
                            for slot, expert in enumerate(exp_routes[t]):
                                local = expert >= 0 and expert // NUM_LOCAL_EXPERTS == ep_group.my_rank
                                exp_id = expert % NUM_LOCAL_EXPERTS if emit_local_expert_ids else expert
                                assert ids[idx, slot].item() == (exp_id if local else invalid_expert)
                                assert w[idx, slot].item() == (1.0 / NUM_TOPK if local else 0)
                        assert (ids[count:] == invalid_expert).all() and (w[count:] == 0).all()

                collective_check(check_dispatch)
                out = runtime.combine(_stage_identity_expert(result, active_cap, combine_mode), handle)
                expected = inp.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
                stream.synchronize()
                collective_check(lambda: torch.testing.assert_close(out.float(), expected, rtol=0, atol=0))

            tokens = 257
            routes = torch.tensor(routes_for(ep_group.my_rank, tokens, tokens, 0), dtype=torch.int64, device="cuda")
            weights = torch.full((tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            inp = inputs_for(ep_group.my_rank, tokens, 1.0)
            out = torch.empty_like(inp)

            def op():
                res, h = runtime.dispatch(inp, routes, weights)
                return runtime.combine(_stage_identity_expert(res, capacity, combine_mode), h, out=out)

            op()
            stream.synchronize()
            graph, _ = _capture_graph(ep_group, stream, op)
            rounds = [
                (0, "mixed"),
                (1, "mixed"),
                (chunk_rows - 1, "mixed"),
                (chunk_rows + 1, "mixed"),
                (257, "remote"),
                (257, "uneven"),
                (0, "mixed"),
                (257, "mixed"),
            ]
            observed, expected = [], []
            for step, (active, routing) in enumerate(rounds):
                nxt = (
                    [[-1] * NUM_TOPK] * tokens
                    if active == 0 and step % 2 == 0
                    else routes_for(ep_group.my_rank, tokens, active, step, routing)
                )
                routes.copy_(torch.tensor(nxt, dtype=torch.int64, device="cuda"))
                inp.copy_(inputs_for(ep_group.my_rank, tokens, (1.0, 0.5, 0.25, 2.0)[step % 4]))
                weights[:, 6] = 0 if step % 2 else 1.0 / NUM_TOPK
                expected.append(inp.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True))
                graph.replay()
                observed.append(out.clone())
            stream.synchronize()
            collective_check(
                lambda: [torch.testing.assert_close(a.float(), r, rtol=0, atol=0) for a, r in zip(observed, expected)]
            )


@pytest.mark.parametrize("qps_per_peer", [1, 4])
@pytest.mark.parametrize("combine_inter_domain_dtype", [torch.float32, torch.bfloat16])
def test_latency_hierarchical_aggregate_combine_partials(ep_group, qps_per_peer, combine_inter_domain_dtype):
    """FP32 group sums, random BF16 partials, asymmetric tokens, and optional BF16 inter-domain rounding."""
    domain = ep_group.nranks_per_ipc_domain
    if not (ep_group.nranks == 2 * domain and domain >= 3):
        pytest.skip("Requires two NVLink domains with at least three ranks each")
    hidden, capacity, tokens = 2048, 16, 9
    own_tokens = 0 if ep_group.my_rank == 0 else tokens
    invalid_expert = NUM_LOCAL_EXPERTS * ep_group.nranks

    def destinations(owner, t):
        local_base = owner // domain * domain
        remote_base = domain - local_base
        peers = [p for p in range(remote_base, remote_base + domain) if p != remote_base + owner % domain]
        return peers[(owner + t) % len(peers)], peers[(owner + t + 1) % len(peers)], local_base + (owner + 1) % domain

    def routes_for(mode):
        rows = []
        for t in range(own_tokens):
            a, b, nb = destinations(ep_group.my_rank, t)
            row = [
                (
                    p * NUM_LOCAL_EXPERTS + (t + s) % NUM_LOCAL_EXPERTS
                    if p >= 0
                    and mode != "empty"
                    and (mode == "full" or (mode == "local") == (p // domain == ep_group.my_rank // domain))
                    else -1
                )
                for s, p in enumerate([a, a, b, ep_group.my_rank, nb, a, -1, -1])
            ]
            row[1] = row[0]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.int64, device="cuda").reshape(own_tokens, NUM_TOPK)

    def partials(owner, peer):
        gen = torch.Generator(device="cuda").manual_seed(8191 + owner * ep_group.nranks + peer)
        vals = torch.randn((tokens, hidden), generator=gen, device="cuda").to(torch.bfloat16)
        for t in range(tokens):
            a, b, _ = destinations(owner, t)
            vals[t, 0] = 256 if peer == a else 1 if peer == b else -256 if peer == owner else 0
        return vals

    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR,
        combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        num_experts=invalid_expert,
        hidden_size=hidden,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
        invalid_token_expert_id=invalid_expert,
        qps_per_peer=qps_per_peer,
        n_tokens_per_chunk=8,
        combine_inter_domain_dtype=combine_inter_domain_dtype,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            table = torch.stack([partials(o, ep_group.my_rank) for o in range(ep_group.nranks)])
            exp_partials = torch.stack(
                [partials(ep_group.my_rank, p)[:own_tokens].float() for p in range(ep_group.nranks)]
            )
            owners, peers = torch.arange(ep_group.nranks, device="cuda")[:, None], torch.arange(
                ep_group.nranks, device="cuda"
            )
            row_idx = torch.arange(capacity, device="cuda")[None, :]
            inp = (
                torch.arange(own_tokens, device="cuda")[:, None]
                + 1
                + (torch.arange(hidden, device="cuda") % 7)[None, :]
            ).to(torch.bfloat16)
            weights, routes = torch.ones((own_tokens, NUM_TOPK), device="cuda"), routes_for("full")

            def oracle():
                routed = (
                    (routes[:, :, None] >= 0) & (routes[:, :, None] // NUM_LOCAL_EXPERTS == peers[None, None, :])
                ).any(dim=1)
                masked = exp_partials * routed.T[:, :, None]
                same_dom = (peers // domain == ep_group.my_rank // domain)[:, None, None]
                local_sum, remote_sum = torch.where(same_dom, masked, 0.0).sum(dim=0), torch.where(
                    ~same_dom, masked, 0.0
                ).sum(dim=0)
                if combine_inter_domain_dtype == torch.bfloat16:
                    remote_sum = remote_sum.to(torch.bfloat16).float()
                return (local_sum + remote_sum).to(torch.bfloat16).float()

            def op():
                res, h = runtime.dispatch(inp, routes, weights)
                t_ids = (res.tokens[:, :, 0].float().long() - 1).clamp(0, tokens - 1)
                valid = row_idx < res.layout.num_tokens_per_rank[:, None]
                res.combine_input_buffer.copy_(torch.where(valid[:, :, None], table[owners, t_ids], 0))
                return runtime.combine(res.combine_input_buffer, h)

            def check(actual, expected, mode):
                torch.testing.assert_close(actual.float(), expected, rtol=2**-7, atol=2**-10)
                if mode == "full":
                    target = (
                        torch.zeros(own_tokens, device="cuda")
                        if combine_inter_domain_dtype == torch.bfloat16
                        else torch.ones(own_tokens, device="cuda")
                    )
                    torch.testing.assert_close(actual[:, 0].float(), target, rtol=0, atol=0)

            for mode in ("full", "local", "empty", "remote"):
                routes.copy_(routes_for(mode))
                out, expected = op(), oracle()
                stream.synchronize()
                collective_check(lambda: check(out, expected, mode))
            graph, out = _capture_graph(ep_group, stream, op)
            modes = ("empty", "full", "remote", "local", "full")
            observed, refs = [], []
            for mode in modes:
                routes.copy_(routes_for(mode))
                refs.append(oracle())
                graph.replay()
                observed.append(out.clone())
            stream.synchronize()
            collective_check(lambda: [check(a, e, m) for a, e, m in zip(observed, refs, modes)])


def _quantized_bf16_inputs(source, num_tokens, hidden, scale_factor):
    t_idx = torch.arange(num_tokens, device="cuda", dtype=torch.float32)[:, None]
    b_idx = torch.arange(hidden // 32, device="cuda", dtype=torch.float32)[None, :]
    b_scale = torch.where(b_idx == 1, 0.0, torch.ldexp(torch.ones_like(b_idx), (b_idx % 5).int() - 2))
    pat = ((torch.arange(32, device="cuda", dtype=torch.float32) % 7) - 3.0) * 1.5
    vals = ((t_idx + source + 1.0)[..., None] * b_scale[..., None] * pat).reshape(num_tokens, hidden) * scale_factor
    vals[:, 0], vals[:, 1] = t_idx[:, 0] + 1.0, float(source + 1)
    return vals.to(torch.bfloat16).contiguous()


@pytest.mark.parametrize("combine_mode", [CombineMode.RANK_LOCAL_REDUCE, CombineMode.DIRECT_SEND])
@pytest.mark.parametrize("emit_local_expert_ids", [False, True])
@pytest.mark.parametrize(
    "hidden,quant,inter_domain_dtype",
    [
        (2048, QuantConfig(format=DispatchDataType.FP8_E4M3_STATIC, scale=0.5), torch.float32),
        (
            2048,
            QuantConfig(
                format=DispatchDataType.FP8_E4M3_STATIC,
                scale=0.5,
                input_mode=QuantConfig.InputMode.QUANTIZE,
                combine_format=DispatchDataType.FP8_E4M3_STATIC,
            ),
            torch.float32,
        ),
        (
            2048,
            QuantConfig(
                format=DispatchDataType.FP8_E4M3_STATIC,
                scale=(0.5, 0.25),
                input_mode=QuantConfig.InputMode.QUANTIZE,
                combine_format=DispatchDataType.FP8_E4M3_STATIC,
            ),
            torch.float32,
        ),
        (
            2048,
            QuantConfig(
                format=DispatchDataType.FP8_E4M3_STATIC,
                scale=(0.5, 448 / 79488.0),
                combine_format=DispatchDataType.FP8_E4M3_STATIC,
            ),
            torch.bfloat16,
        ),
        (2048, QuantConfig(format=DispatchDataType.MXFP8_E4M3), torch.float32),
        (4352, QuantConfig(format=DispatchDataType.MXFP8_E4M3), torch.bfloat16),
    ],
)
def test_latency_hierarchical_quantized(
    ep_group, combine_mode, emit_local_expert_ids, hidden, quant, inter_domain_dtype
):
    """Static FP8 (prequantized/fused, dual scale) and MXFP8 (including non-16B-aligned H=4352) with graph replay."""
    num_tokens, capacity = 9, 16
    is_mxfp8 = quant.format == DispatchDataType.MXFP8_E4M3
    prequantized = not is_mxfp8 and quant.input_mode == QuantConfig.InputMode.PREQUANTIZED
    combine_fp8 = quant.combine_format == DispatchDataType.FP8_E4M3_STATIC
    dispatch_scale = None if is_mxfp8 else (quant.scale[0] if isinstance(quant.scale, tuple) else quant.scale)
    combine_scale = (quant.scale[1] if isinstance(quant.scale, tuple) else quant.scale) if combine_fp8 else None

    def make_input(source, sf):
        bf16 = _quantized_bf16_inputs(source, num_tokens, hidden, sf)
        return (bf16.float() * dispatch_scale).to(torch.float8_e4m3fn) if prequantized else bf16

    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR,
        combine_mode=combine_mode,
        emit_local_expert_ids=emit_local_expert_ids,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=hidden,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
        quant=quant,
        combine_inter_domain_dtype=inter_domain_dtype,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, t, num_tokens) for t in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            dispatch_in = make_input(ep_group.my_rank, 1.0)
            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

            def run_round():
                res, h = runtime.dispatch(dispatch_in, routes, weights, stream=stream)
                return res, runtime.combine(
                    _stage_identity_expert(res, capacity, combine_mode), h, out=out, stream=stream
                )

            res, _ = run_round()

            def check_dispatch():
                assert res.tokens.dtype == torch.float8_e4m3fn
                assert res.combine_input_buffer.dtype == (torch.float8_e4m3fn if combine_fp8 else torch.bfloat16)
                stream.synchronize()
                _check_dispatch_metadata(ep_group, res, num_tokens)
                if is_mxfp8:
                    assert res.quant.block_scales.shape == (ep_group.nranks, capacity, hidden // 32)
                    for src, count in enumerate(res.layout.num_tokens_per_rank.cpu().tolist()):
                        if count == 0:
                            continue
                        ref_q, ref_s, _ = _reference_mxfp8_quantize(
                            _quantized_bf16_inputs(src, num_tokens, hidden, 1.0)
                        )
                        row_ids = _dequantize_dispatch(res)[src, :count, 0].round().long() - 1
                        torch.testing.assert_close(
                            res.tokens[src, :count].view(torch.uint8), ref_q[row_ids].view(torch.uint8), rtol=0, atol=0
                        )
                        torch.testing.assert_close(res.quant.block_scales[src, :count], ref_s[row_ids], rtol=0, atol=0)

            def check_combine(sf):
                bf16 = _quantized_bf16_inputs(ep_group.my_rank, num_tokens, hidden, sf)
                real_in = (
                    _reference_mxfp8_quantize(bf16)[2]
                    if is_mxfp8
                    else (bf16.float() * dispatch_scale).to(torch.float8_e4m3fn).float() / dispatch_scale
                )
                expected = _expected_hierarchical_combine(
                    real_in,
                    routes,
                    weights,
                    ep_group,
                    combine_mode,
                    combine_scale=combine_scale,
                    inter_domain_dtype=inter_domain_dtype,
                )
                torch.testing.assert_close(out.float(), expected, rtol=0, atol=0)

            collective_check(check_dispatch)
            collective_check(lambda: check_combine(1.0))
            graph, _ = _capture_graph(ep_group, stream, run_round)
            for sf in (0.5, 2.0):
                dispatch_in.copy_(make_input(ep_group.my_rank, sf))
                graph.replay()
                stream.synchronize()
                ep_group.barrier()
                collective_check(lambda s=sf: check_combine(s))


@pytest.mark.parametrize("combine_mode", [CombineMode.RANK_LOCAL_REDUCE, CombineMode.DIRECT_SEND])
@pytest.mark.parametrize("hidden", [5376, 9216])
@pytest.mark.parametrize("fmt", ["bf16", "mxfp8", "fp8_static"])
def test_latency_hierarchical_topk_16(ep_group, combine_mode, hidden, fmt):
    """Hierarchical top-k=16 across single-round (H=5376) and two-round (H=9216) combine staging and all formats."""
    topk, num_tokens, capacity = 16, 9, 16
    quant = {
        "bf16": None,
        "mxfp8": QuantConfig(format=DispatchDataType.MXFP8_E4M3),
        "fp8_static": QuantConfig(
            format=DispatchDataType.FP8_E4M3_STATIC,
            scale=0.5,
            input_mode=QuantConfig.InputMode.QUANTIZE,
            combine_format=DispatchDataType.FP8_E4M3_STATIC,
        ),
    }[fmt]
    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR,
        combine_mode=combine_mode,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=hidden,
        topk=topk,
        max_tokens_per_rank=capacity,
        quant=quant,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, t, num_tokens, topk) for t in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.full((num_tokens, topk), 1.0 / topk, device="cuda")
            vals = token_values(ep_group.my_rank, num_tokens, 1.0, topk)
            bf16_in = vals[:, None].expand(-1, hidden).to(torch.bfloat16).contiguous()
            out = torch.empty_like(bf16_in)

            def run_round():
                res, h = runtime.dispatch(bf16_in, routes, weights, stream=stream)
                return res, runtime.combine(
                    _stage_identity_expert(res, capacity, combine_mode), h, out=out, stream=stream
                )

            res, _ = run_round()

            def check_round():
                stream.synchronize()
                assert tuple(res.topk_ids.shape) == (ep_group.nranks, capacity, topk)
                real_in = (
                    _reference_mxfp8_quantize(bf16_in)[2]
                    if fmt == "mxfp8"
                    else (
                        (bf16_in.float() * 0.5).to(torch.float8_e4m3fn).float() / 0.5
                        if fmt == "fp8_static"
                        else bf16_in.float()
                    )
                )
                expected = _expected_hierarchical_combine(
                    real_in, routes, weights, ep_group, combine_mode, combine_scale=0.5 if fmt == "fp8_static" else None
                )
                torch.testing.assert_close(out.float(), expected, rtol=0, atol=0)

            collective_check(check_round)
            graph, _ = _capture_graph(ep_group, stream, run_round)
            bf16_in.copy_((vals * 0.5)[:, None].expand(-1, hidden).to(torch.bfloat16))
            graph.replay()
            stream.synchronize()
            ep_group.barrier()
            collective_check(check_round)
