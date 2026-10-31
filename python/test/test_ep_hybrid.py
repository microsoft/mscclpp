# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Correctness coverage for NVLink/IPC + GPUNetIO latency RANK_MAJOR.

Covers RANK_LOCAL_REDUCE within an IPC domain and across domains,
including chunked dispatch, gateway distribution, and FP32 aggregate combine.
Example (two nodes, four GPUs each)::

    mpirun -np 8 --map-by ppr:4:node \\
        python3 -m pytest python/test/test_ep_hybrid.py -vx
"""

from contextlib import contextmanager
import importlib.util

import cupy as cp
from mpi4py import MPI
import pytest

torch = pytest.importorskip("torch")
if importlib.util.find_spec("mscclpp.mscclpp_ep_cpp") is None:
    pytest.skip("The CUDA EP extension was not built", allow_module_level=True)

from mscclpp import CommGroup
from mscclpp.ep import (
    CombineMode,
    DispatchLayout,
    MoECommunicator,
    MoECommunicatorConfig,
    MoEMode,
)

NUM_TOPK = 8
HIDDEN = 4096
NUM_LOCAL_EXPERTS = 16


@pytest.fixture(scope="module")
def ep_group():
    if not torch.cuda.is_available():
        pytest.skip("EP requires CUDA")
    torch.cuda.set_device(cp.cuda.runtime.getDevice())
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("EP requires SM90 or newer")
    group = CommGroup(mpi_comm=MPI.COMM_WORLD)
    if not 1 <= group.nranks <= 64:
        pytest.skip("EP requires 1-64 ranks")
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


def token_routes(rank, world_size, token, num_tokens):
    if token == num_tokens - 1:
        return [-1] * NUM_TOPK
    num_experts = NUM_LOCAL_EXPERTS * world_size
    base = (rank * NUM_LOCAL_EXPERTS + token * 7) % num_experts
    routes = [(base + topk * 5) % num_experts for topk in range(NUM_TOPK)]
    if token == 1:
        routes[1] = -1
    return routes


def collective_check(check):
    """Run ``check`` on every rank and fail on all ranks if any rank failed.

    Use before launching combine: a rank that fails alone would otherwise skip combine while
    its peers wait for its GPU participation, hanging the job instead of reporting the error.
    """
    error = None
    try:
        check()
    except (Exception, pytest.fail.Exception) as exc:
        error = f"{type(exc).__name__}: {exc}"
    failed = {rank: e for rank, e in enumerate(MPI.COMM_WORLD.allgather(error)) if e is not None}
    if failed:
        pytest.fail(f"Validation failed on ranks {sorted(failed)}: {failed}")


def token_values(rank, num_tokens, scale):
    values = ((torch.arange(num_tokens, device="cuda") + rank * num_tokens) % 15 + 1).float() * NUM_TOPK
    return values * scale


def _expert_output(result, active_capacity):
    """Identity experts return weighted local sums for RANK_MAJOR."""
    counts = result.layout.num_tokens_per_rank
    valid_rows = torch.arange(active_capacity, device="cuda")[None, :] < counts[:, None]
    local_weights = torch.where(valid_rows[..., None], result.weights, 0).sum(dim=-1, keepdim=True)
    return torch.where(valid_rows[..., None], result.tokens.float() * local_weights, 0).to(torch.bfloat16)


def _check_dispatch_metadata(ep_group, result, num_tokens):
    expected_counts = []
    for source in range(ep_group.nranks):
        routes = [token_routes(source, ep_group.nranks, token, num_tokens) for token in range(num_tokens)]
        local = [[e >= 0 and e // NUM_LOCAL_EXPERTS == ep_group.my_rank for e in r] for r in routes]
        expected_counts.append(sum(any(row) for row in local))
    assert result.layout.num_tokens_per_rank.cpu().tolist() == expected_counts


@pytest.mark.parametrize("hidden", [HIDDEN, 2048])
def test_latency_hybrid_dispatch_combine(ep_group, hidden):
    num_tokens, capacity = 5, 8
    layout = DispatchLayout.RANK_MAJOR
    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=layout,
        combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=hidden,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, t, num_tokens) for t in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            input = token_values(ep_group.my_rank, num_tokens, 1.0)[:, None].expand(-1, hidden)
            input = input.to(torch.bfloat16).contiguous()
            result, handle = runtime.dispatch(input, routes, weights, stream=stream)

            def check_dispatch():
                shape = (ep_group.nranks, capacity, hidden)
                assert tuple(result.tokens.shape) == shape
                assert tuple(result.combine_input_buffer.shape) == shape
                assert tuple(result.topk_ids.shape) == shape[:-1] + (NUM_TOPK,)
                assert result.layout.kind == layout
                assert result.quant is None
                stream.synchronize()
                _check_dispatch_metadata(ep_group, result, num_tokens)

            collective_check(check_dispatch)

            result.combine_input_buffer.copy_(_expert_output(result, capacity))
            combined = runtime.combine(result.combine_input_buffer, handle, stream=stream)
            expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
            stream.synchronize()
            torch.testing.assert_close(combined.float(), expected, rtol=0, atol=0)


def test_latency_hybrid_multi_iteration_graph_replay(ep_group):
    layout = DispatchLayout.RANK_MAJOR
    num_tokens, capacity = 5, 8
    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=layout,
        combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, t, num_tokens) for t in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            input = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(input)

            def operation():
                result, handle = runtime.dispatch(input, routes, weights)
                result.combine_input_buffer.copy_(_expert_output(result, capacity))
                return runtime.combine(result.combine_input_buffer, handle, out=out)

            input.copy_(token_values(ep_group.my_rank, num_tokens, 1.0)[:, None].expand(-1, HIDDEN))
            operation()
            stream.synchronize()
            ep_group.barrier()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(2):
                    output = operation()
            stream.synchronize()
            ep_group.barrier()

            for scale in (2.0, 0.5, 4.0):
                input.copy_(token_values(ep_group.my_rank, num_tokens, scale)[:, None].expand(-1, HIDDEN))
                graph.replay()
                stream.synchronize()
                ep_group.barrier()
                expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
                torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)


def _alternate_routes(rank, num_tokens):
    """Only token 0 is routed, to this rank's first expert: every peer receives zero rows."""
    routes = [[-1] * NUM_TOPK for _ in range(num_tokens)]
    routes[0][0] = rank * NUM_LOCAL_EXPERTS
    return routes


def test_latency_hybrid_single_round_graph_changing_routes(ep_group):
    """A graph holding one dispatch/combine round must not reuse the previous replay's metadata."""
    hybrid_rank_major = ep_group.nranks_per_ipc_domain < ep_group.nranks
    if not hybrid_rank_major:
        pytest.skip("Single-domain RANK_MAJOR uses the IPC path, whose graphs need two captured rounds")
    layout = DispatchLayout.RANK_MAJOR
    num_tokens, capacity = 5, 8
    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=layout,
        combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            full = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, t, num_tokens) for t in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            sparse = torch.tensor(_alternate_routes(ep_group.my_rank, num_tokens), dtype=torch.int64, device="cuda")
            routes = full.clone()
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            input = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(input)

            def operation():
                result, handle = runtime.dispatch(input, routes, weights)
                result.combine_input_buffer.copy_(_expert_output(result, capacity))
                return runtime.combine(result.combine_input_buffer, handle, out=out)

            input.copy_(token_values(ep_group.my_rank, num_tokens, 1.0)[:, None].expand(-1, HIDDEN))
            operation()
            stream.synchronize()
            ep_group.barrier()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation()
            stream.synchronize()
            ep_group.barrier()

            for step, scale in enumerate((2.0, 0.5, 4.0, 1.0, 3.0, 0.25)):
                routes.copy_(sparse if step % 2 == 0 else full)
                input.copy_(token_values(ep_group.my_rank, num_tokens, scale)[:, None].expand(-1, HIDDEN))
                graph.replay()
                stream.synchronize()
                ep_group.barrier()
                expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
                torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "qps_per_peer,chunk_rows,hidden",
    [("1", 8, 7168), ("4", 8, 7168), ("4", 7, 7168), ("1", 7, 2048), ("4", 16, 4096)],
)
@pytest.mark.parametrize("aggregate_combine", [False, True])
def test_latency_hybrid_send_distribute(ep_group, qps_per_peer, chunk_rows, hidden, aggregate_combine, monkeypatch):
    """Single-copy IB chunks, gateway-excluding routes, and credit reuse across changing graph rounds."""
    domain = ep_group.nranks_per_ipc_domain
    if not (ep_group.nranks == 2 * domain and domain >= 2):
        pytest.skip("Send-and-distribute coverage requires two equal NVLink domains with at least two ranks each")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE", "1" if aggregate_combine else "0")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER", qps_per_peer)
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", str(chunk_rows))
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS", str(chunk_rows) if aggregate_combine else "0")
    layout = DispatchLayout.RANK_MAJOR
    capacity = 272
    invalid_expert = NUM_LOCAL_EXPERTS * ep_group.nranks

    def routes_for(source, tokens, active, rotation, routing="mixed"):
        local_base = source // domain * domain
        remote_base = domain - local_base
        gateway = remote_base + source % domain
        peers = [peer for peer in range(remote_base, remote_base + domain) if peer != gateway]
        if routing == "uneven":
            active = (0, active // 3, active)[source % 3]
        result = []
        for token in range(tokens):
            if token >= active:
                result.append(
                    [source * NUM_LOCAL_EXPERTS] + [-1] * (NUM_TOPK - 1)
                    if token == active and routing == "mixed"
                    else [-1] * NUM_TOPK
                )
                continue
            a, b, c = [peers[(source + token + rotation + shift) % len(peers)] for shift in range(3)]
            ranks = (
                [a, a, b, c, source, local_base + (source + 1) % domain, a]
                if routing == "mixed"
                else [a, a, b, c, a, b, a]
            )
            experts = [peer * NUM_LOCAL_EXPERTS + (token + slot) % NUM_LOCAL_EXPERTS for slot, peer in enumerate(ranks)]
            experts[1] = experts[0]  # Duplicate expert as well as duplicate destination rank.
            result.append(experts + [-1])
        return result

    def inputs_for(source, tokens, scale):
        token = torch.arange(tokens, device="cuda")
        features = torch.arange(hidden, device="cuda") % 3 + 1
        values = (((token + source) % 7 + 1)[:, None] * 8 * features[None, :]).float()
        # Two small exact BF16 components identify rows without enlarging weighted
        # identity results past the exact range used by the combine assertions.
        values[:, 0] = token % 33 + 1
        values[:, 1] = token // 33 + 1
        return (values * scale).to(torch.bfloat16)

    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=layout,
        combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        num_experts=invalid_expert,
        hidden_size=hidden,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
        invalid_token_expert_id=invalid_expert,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Change the active row stride within one runtime-owned registered allocation.
            # Larger rounds reuse forwarding TMA slots and assign multiple
            # gateway rows to each Combine worker before it may wait remotely.
            token_counts = sorted(
                {0, 1, 7, 8, 9, 31, 32, 33, 65, 73, 129, 257, chunk_rows - 1, chunk_rows, chunk_rows + 1}
            )
            for step, tokens in enumerate(token_counts):
                active_capacity = max(tokens, 1) if step % 2 else capacity
                routes = torch.tensor(
                    routes_for(ep_group.my_rank, tokens, tokens, step), dtype=torch.int64, device="cuda"
                ).reshape(tokens, NUM_TOPK)
                weights = torch.full((tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
                input = inputs_for(ep_group.my_rank, tokens, 1.0)
                result, handle = runtime.dispatch(input, routes, weights, runtime_max_tokens_per_rank=active_capacity)

                def check_dispatch():
                    stream.synchronize()
                    counts = result.layout.num_tokens_per_rank.cpu().tolist()
                    for source in range(ep_group.nranks):
                        expected_routes = routes_for(source, tokens, tokens, step)
                        routed = [
                            token
                            for token, row in enumerate(expected_routes)
                            if any(expert >= 0 and expert // NUM_LOCAL_EXPERTS == ep_group.my_rank for expert in row)
                        ]
                        count = counts[source]
                        assert count == len(routed)
                        ids = result.topk_ids[source].cpu()
                        actual_weights = result.weights[source].cpu()
                        row_ids = result.tokens[source, :count, :2].cpu().int() - 1
                        rows = (row_ids[:, 0] + 33 * row_ids[:, 1]).tolist()
                        assert sorted(rows) == routed
                        expected_input = inputs_for(source, tokens, 1.0)
                        if rows:
                            torch.testing.assert_close(
                                result.tokens[source, :count], expected_input[rows], rtol=0, atol=0
                            )
                        for index, token in enumerate(rows):
                            for slot, expert in enumerate(expected_routes[token]):
                                local = expert >= 0 and expert // NUM_LOCAL_EXPERTS == ep_group.my_rank
                                assert ids[index, slot].item() == (expert if local else invalid_expert)
                                assert actual_weights[index, slot].item() == (1.0 / NUM_TOPK if local else 0)
                        assert (ids[count:] == invalid_expert).all()
                        assert (actual_weights[count:] == 0).all()

                collective_check(check_dispatch)
                result.combine_input_buffer.copy_(_expert_output(result, active_capacity))
                output = runtime.combine(result.combine_input_buffer, handle)
                expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
                stream.synchronize()
                collective_check(lambda: torch.testing.assert_close(output.float(), expected, rtol=0, atol=0))

            # Capture one fixed-shape round, then vary routing, weights, and contents.
            # No inter-rank barriers between replays: Combine does not return data from
            # the gateway itself, so only the new protocol's ACK protects reuse.
            tokens = 257
            routes = torch.empty((tokens, NUM_TOPK), dtype=torch.int64, device="cuda")
            weights = torch.full((tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            input = inputs_for(ep_group.my_rank, tokens, 1.0)
            out = torch.empty_like(input)
            routes.copy_(torch.tensor(routes_for(ep_group.my_rank, tokens, tokens, 0), device="cuda"))

            def operation():
                result, handle = runtime.dispatch(input, routes, weights)
                result.combine_input_buffer.copy_(_expert_output(result, capacity))
                return runtime.combine(result.combine_input_buffer, handle, out=out)

            operation()
            stream.synchronize()
            ep_group.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation()
            stream.synchronize()
            ep_group.barrier()
            observed, expected = [], []
            rounds = [
                (active, "mixed")
                for active in (
                    0,
                    1,
                    chunk_rows - 1,
                    chunk_rows,
                    chunk_rows + 1,
                    31,
                    32,
                    33,
                    0,
                    129,
                    1,
                    0,
                    9,
                    194,
                    7,
                    257,
                )
            ] + [(257, "remote"), (257, "uneven"), (1, "uneven"), (257, "remote")]
            for step, (active, routing) in enumerate(rounds):
                next_routes = routes_for(ep_group.my_rank, tokens, active, step, routing)
                if active == 0 and step % 2 == 0:
                    next_routes = [[-1] * NUM_TOPK for _ in range(tokens)]
                routes.copy_(torch.tensor(next_routes, dtype=torch.int64, device="cuda"))
                scale = (1.0, 0.5, 0.25, 2.0)[step % 4]
                input.copy_(inputs_for(ep_group.my_rank, tokens, scale))
                weights[:, 6] = 0 if step % 2 else 1.0 / NUM_TOPK
                expected.append(input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True))
                graph.replay()
                observed.append(output.clone())
            stream.synchronize()

            def check_replays():
                for actual, reference in zip(observed, expected):
                    torch.testing.assert_close(actual.float(), reference, rtol=0, atol=0)

            collective_check(check_replays)


@pytest.mark.parametrize("qps_per_peer", ["1", "4"])
def test_latency_hybrid_aggregate_combine_partials(ep_group, qps_per_peer, monkeypatch):
    """FP32 group sums, random BF16 partials, asymmetric tokens, and no intermediate BF16 rounding."""
    domain = ep_group.nranks_per_ipc_domain
    if not (ep_group.nranks == 2 * domain and domain >= 3):
        pytest.skip("Aggregate partial coverage requires two NVLink domains with at least three ranks each")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE", "1")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER", qps_per_peer)
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", "8")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS", "8")
    hidden, capacity, tokens = 2048, 16, 9
    # Rank zero still has gateway rows to produce for its paired rank.
    own_tokens = 0 if ep_group.my_rank == 0 else tokens
    invalid_expert = NUM_LOCAL_EXPERTS * ep_group.nranks

    def destinations(owner, token):
        local_base = owner // domain * domain
        remote_base = domain - local_base
        gateway = remote_base + owner % domain
        peers = [peer for peer in range(remote_base, remote_base + domain) if peer != gateway]
        a, b = peers[(owner + token) % len(peers)], peers[(owner + token + 1) % len(peers)]
        return a, b, local_base + (owner + 1) % domain

    def routes_for(mode):
        rows = []
        for token in range(own_tokens):
            a, b, neighbor = destinations(ep_group.my_rank, token)
            peers = [a, a, b, ep_group.my_rank, neighbor, a, -1, -1]
            row = []
            for slot, peer in enumerate(peers):
                local = peer >= 0 and peer // domain == ep_group.my_rank // domain
                valid = peer >= 0 and mode != "empty" and (mode == "full" or (mode == "local") == local)
                row.append(peer * NUM_LOCAL_EXPERTS + (token + slot) % NUM_LOCAL_EXPERTS if valid else -1)
            row[1] = row[0]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.int64, device="cuda").reshape(own_tokens, NUM_TOPK)

    def partials(owner, peer):
        generator = torch.Generator(device="cuda").manual_seed(8191 + owner * ep_group.nranks + peer)
        values = torch.randn((tokens, hidden), generator=generator, device="cuda").to(torch.bfloat16)
        for token in range(tokens):
            a, b, _ = destinations(owner, token)
            # Remote sum 256+1 must remain FP32 until it is added to local -256.
            values[token, 0] = 256 if peer == a else 1 if peer == b else -256 if peer == owner else 0
        return values

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
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            table = torch.stack([partials(owner, ep_group.my_rank) for owner in range(ep_group.nranks)])
            expected_partials = torch.stack(
                [partials(ep_group.my_rank, peer)[:own_tokens].float() for peer in range(ep_group.nranks)]
            )
            owners = torch.arange(ep_group.nranks, device="cuda")[:, None]
            peers = torch.arange(ep_group.nranks, device="cuda")
            row_indices = torch.arange(capacity, device="cuda")[None, :]
            features = torch.arange(hidden, device="cuda") % 7
            input = (torch.arange(own_tokens, device="cuda")[:, None] + 1 + features[None, :]).to(torch.bfloat16)
            weights = torch.ones((own_tokens, NUM_TOPK), device="cuda")
            routes = routes_for("full")

            def oracle():
                routed = (
                    (routes[:, :, None] >= 0) & (routes[:, :, None] // NUM_LOCAL_EXPERTS == peers[None, None, :])
                ).any(dim=1)
                return (expected_partials * routed.T[:, :, None]).sum(dim=0).to(torch.bfloat16).float()

            def operation():
                result, handle = runtime.dispatch(input, routes, weights)
                token_ids = (result.tokens[:, :, 0].float().long() - 1).clamp(0, tokens - 1)
                valid = row_indices < result.layout.num_tokens_per_rank[:, None]
                result.combine_input_buffer.copy_(torch.where(valid[:, :, None], table[owners, token_ids], 0))
                return runtime.combine(result.combine_input_buffer, handle)

            def check(actual, expected, mode):
                torch.testing.assert_close(actual.float(), expected, rtol=2**-7, atol=2**-10)
                if mode == "full":
                    torch.testing.assert_close(
                        actual[:, 0].float(), torch.ones(own_tokens, device="cuda"), rtol=0, atol=0
                    )

            for mode in ("full", "local", "empty", "remote", "full"):
                routes.copy_(routes_for(mode))
                output = operation()
                expected = oracle()
                stream.synchronize()
                collective_check(lambda: check(output, expected, mode))
            ep_group.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation()
            stream.synchronize()
            ep_group.barrier()
            observed, references = [], []
            modes = ("empty", "full", "remote", "local", "full", "remote", "empty", "full")
            for mode in modes:
                routes.copy_(routes_for(mode))
                references.append(oracle())
                graph.replay()
                observed.append(output.clone())
            stream.synchronize()

            def check_replays():
                for actual, expected, mode in zip(observed, references, modes):
                    check(actual, expected, mode)

            collective_check(check_replays)
