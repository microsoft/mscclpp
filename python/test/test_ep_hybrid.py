# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Correctness coverage for the hybrid NVLink/IPC + GPUNetIO latency path.

Covers latency RANK_MAJOR_TOPK_EXPANDED (IPC and GPUNetIO) and, when the job spans
several CUDA IPC domains with ``MSCCLPP_EP_ENABLE_GPUNETIO=1``, latency RANK_MAJOR
with RANK_LOCAL_REDUCE over GPUNetIO, including batched dispatch with chunking and
deferred flush (enabled by that test itself). Example (two nodes, four GPUs each)::

    mpirun -np 8 --map-by ppr:4:node -x MSCCLPP_EP_ENABLE_GPUNETIO=1 \\
        python3 -m pytest python/test/test_ep_hybrid.py -vx
"""

from contextlib import contextmanager
import importlib.util
import os

import cupy as cp
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

NUM_TOPK = 8
HIDDEN = 4096
NUM_LOCAL_EXPERTS = 16


def _gpunetio_requested():
    try:
        return int(os.environ.get("MSCCLPP_EP_ENABLE_GPUNETIO", "0")) != 0
    except ValueError:
        return False


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
    if group.nranks_per_ipc_domain < group.nranks and not _gpunetio_requested():
        pytest.skip("Ranks span CUDA IPC domains; set MSCCLPP_EP_ENABLE_GPUNETIO=1")
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


def test_topk_expanded_config_validation(ep_group):
    base = dict(
        comm=ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR_TOPK_EXPANDED,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=4,
    )
    with pytest.raises(ValueError, match="RANK_LOCAL_REDUCE"):
        MoECommunicatorConfig(**base, combine_mode=CombineMode.DIRECT_SEND)
    with pytest.raises(ValueError, match="BF16"):
        MoECommunicatorConfig(**base, quant=QuantConfig(format=DispatchDataType.FP8_E4M3_STATIC, scale=1.0))
    with pytest.raises(ValueError, match="output_layout"):
        MoECommunicatorConfig(**{**base, "mode": MoEMode.THROUGHPUT})


def _expert_output(layout, result, active_capacity):
    """Identity experts: TOPK_EXPANDED returns unweighted rows; RANK_MAJOR returns weighted local sums."""
    if layout == DispatchLayout.RANK_MAJOR_TOPK_EXPANDED:
        return result.tokens
    counts = result.layout.num_tokens_per_rank
    valid_rows = torch.arange(active_capacity, device="cuda")[None, :] < counts[:, None]
    local_weights = torch.where(valid_rows[..., None], result.weights, 0).sum(dim=-1, keepdim=True)
    return torch.where(valid_rows[..., None], result.tokens.float() * local_weights, 0).to(torch.bfloat16)


def _check_dispatch_metadata(ep_group, layout, result, num_tokens):
    num_experts = NUM_LOCAL_EXPERTS * ep_group.nranks
    expected_counts = []
    for source in range(ep_group.nranks):
        routes = [token_routes(source, ep_group.nranks, token, num_tokens) for token in range(num_tokens)]
        local = [[e >= 0 and e // NUM_LOCAL_EXPERTS == ep_group.my_rank for e in r] for r in routes]
        if layout != DispatchLayout.RANK_MAJOR_TOPK_EXPANDED:
            expected_counts.append(sum(any(row) for row in local))
            continue
        # Expanded counts one row per locally routed (token, top-k slot) pair.
        expected_counts.append(sum(sum(row) for row in local))
        # Expanded rows keep their [source rank, token, top-k slot] position.
        ids = result.topk_ids[source, :num_tokens].cpu()
        weights = result.weights[source, :num_tokens].cpu()
        for token in range(num_tokens):
            for slot in range(NUM_TOPK):
                if local[token][slot]:
                    assert ids[token, slot].item() == routes[token][slot]
                    assert weights[token, slot].item() == 1.0 / NUM_TOPK
                else:
                    assert ids[token, slot].item() < 0 or ids[token, slot].item() >= num_experts
    assert result.layout.num_tokens_per_rank.cpu().tolist() == expected_counts


@pytest.mark.parametrize("hidden", [HIDDEN, 2048])
@pytest.mark.parametrize("layout", [DispatchLayout.RANK_MAJOR, DispatchLayout.RANK_MAJOR_TOPK_EXPANDED])
def test_latency_hybrid_dispatch_combine(ep_group, layout, hidden):
    num_tokens, capacity = 5, 8
    expanded = layout == DispatchLayout.RANK_MAJOR_TOPK_EXPANDED
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
            if expanded:

                def check_capacity_rejected():
                    with pytest.raises(ValueError, match="runtime_max_tokens_per_rank"):
                        runtime.dispatch(input, routes, weights, runtime_max_tokens_per_rank=capacity - 1)

                collective_check(check_capacity_rejected)

            result, handle = runtime.dispatch(input, routes, weights, stream=stream)

            def check_dispatch():
                shape = (ep_group.nranks, capacity) + ((NUM_TOPK,) if expanded else ()) + (hidden,)
                assert tuple(result.tokens.shape) == shape
                assert tuple(result.combine_input_buffer.shape) == shape
                assert tuple(result.topk_ids.shape) == shape[:-1] + (() if expanded else (NUM_TOPK,))
                assert result.layout.kind == layout
                assert result.quant is None
                stream.synchronize()
                _check_dispatch_metadata(ep_group, layout, result, num_tokens)

            collective_check(check_dispatch)

            result.combine_input_buffer.copy_(_expert_output(layout, result, capacity))
            combined = runtime.combine(result.combine_input_buffer, handle, stream=stream)
            expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
            stream.synchronize()
            torch.testing.assert_close(combined.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", [DispatchLayout.RANK_MAJOR, DispatchLayout.RANK_MAJOR_TOPK_EXPANDED])
def test_latency_hybrid_multi_iteration_graph_replay(ep_group, layout):
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
                result.combine_input_buffer.copy_(_expert_output(layout, result, capacity))
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


@pytest.mark.parametrize("layout", [DispatchLayout.RANK_MAJOR, DispatchLayout.RANK_MAJOR_TOPK_EXPANDED])
def test_latency_hybrid_single_round_graph_changing_routes(ep_group, layout):
    """A graph holding one dispatch/combine round must not reuse the previous replay's metadata."""
    hybrid_rank_major = _gpunetio_requested() and ep_group.nranks_per_ipc_domain < ep_group.nranks
    if layout == DispatchLayout.RANK_MAJOR and not hybrid_rank_major:
        pytest.skip("Single-domain RANK_MAJOR uses the IPC path, whose graphs need two captured rounds")
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
                result.combine_input_buffer.copy_(_expert_output(layout, result, capacity))
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


def _all_rank_routes(rank, world_size, num_tokens):
    """Every token but the last reaches NUM_TOPK distinct experts on min(NUM_TOPK, world_size) ranks."""
    routes = [
        [
            ((rank + slot) % world_size) * NUM_LOCAL_EXPERTS + (token + slot) % NUM_LOCAL_EXPERTS
            for slot in range(NUM_TOPK)
        ]
        for token in range(num_tokens - 1)
    ]
    return routes + [[-1] * NUM_TOPK]


@pytest.mark.parametrize("defer_flush", ["0", "1"])
def test_latency_hybrid_batched_dispatch_chunked_graph_replay(ep_group, defer_flush, monkeypatch):
    """Batched GPUNetIO dispatch with several chunks plus a partial tail, deferred flush on and off,
    and single-round graph replays alternating populated and empty destinations."""
    if not (_gpunetio_requested() and ep_group.nranks_per_ipc_domain < ep_group.nranks):
        pytest.skip("Batched dispatch applies only to cross-domain GPUNetIO RANK_MAJOR")
    # Read at initialize(); every rank sets the same values, which initialize() also verifies.
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_BATCHED_DISPATCH", "1")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", "2")
    monkeypatch.setenv("MSCCLPP_EP_GPUNETIO_DEFER_FLUSH", defer_flush)
    # Each populated destination receives num_tokens - 1 = 7 rows: three 2-row chunks and a 1-row tail.
    num_tokens, capacity = 8, 8
    layout = DispatchLayout.RANK_MAJOR
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
                _all_rank_routes(ep_group.my_rank, ep_group.nranks, num_tokens), dtype=torch.int64, device="cuda"
            )
            sparse = torch.tensor(_alternate_routes(ep_group.my_rank, num_tokens), dtype=torch.int64, device="cuda")
            routes = full.clone()
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            input = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(input)

            def operation():
                result, handle = runtime.dispatch(input, routes, weights)
                result.combine_input_buffer.copy_(_expert_output(layout, result, capacity))
                return runtime.combine(result.combine_input_buffer, handle, out=out)

            def expected():
                return input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)

            input.copy_(token_values(ep_group.my_rank, num_tokens, 1.0)[:, None].expand(-1, HIDDEN))
            output = operation()
            stream.synchronize()
            ep_group.barrier()
            torch.testing.assert_close(output.float(), expected(), rtol=0, atol=0)

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
                torch.testing.assert_close(output.float(), expected(), rtol=0, atol=0)


@pytest.mark.parametrize("fast_path", [True, False])
def test_topk_expanded_preweighted_zero_weight_routes(ep_group, fast_path, monkeypatch):
    """With apply_router_weights=False, zero-weight routes must contribute the caller's zero rows."""
    for name in ("MSCCLPP_EP_EXPANDED_IPC_FASTPATH", "MSCCLPP_EP_EXPANDED_GPUNETIO_FASTPATH"):
        monkeypatch.setenv(name, "1" if fast_path else "0")
    num_tokens, capacity = 5, 8
    layout = DispatchLayout.RANK_MAJOR_TOPK_EXPANDED
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
            input = token_values(ep_group.my_rank, num_tokens, 1.0)[:, None].expand(-1, HIDDEN)
            input = input.to(torch.bfloat16).contiguous()
            zeroed = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            zeroed[:, ::2] = 0.0
            for weights in (torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda"), zeroed):
                result, handle = runtime.dispatch(input, routes, weights, stream=stream)
                num_experts = NUM_LOCAL_EXPERTS * ep_group.nranks
                local = ((result.topk_ids >= 0) & (result.topk_ids < num_experts))[..., None]
                preweighted = torch.where(local, result.tokens.float() * result.weights[..., None], 0)
                result.combine_input_buffer.copy_(preweighted.to(torch.bfloat16))
                combined = runtime.combine(
                    result.combine_input_buffer, handle, stream=stream, apply_router_weights=False
                )
                expected = input.float() * torch.where(routes >= 0, weights, 0).sum(dim=-1, keepdim=True)
                stream.synchronize()
                torch.testing.assert_close(combined.float(), expected, rtol=0, atol=0)
                ep_group.barrier()
