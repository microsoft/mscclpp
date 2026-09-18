# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Python bindings for the C++ EP runtime, exercised with real CUDA IPC peers."""

from contextlib import contextmanager
import gc
import importlib.util

import cupy as cp
from mpi4py import MPI
import pytest

torch = pytest.importorskip("torch")
if importlib.util.find_spec("mscclpp.mscclpp_ep_cpp") is None:
    pytest.skip("The CUDA EP extension was not built", allow_module_level=True)

from mscclpp import CommGroup
from mscclpp._mscclpp import Error as MscclppError
from mscclpp.ep import CombineMode, DispatchDataType, DispatchLayout, MoECommunicator, MoEMode, QuantConfig


@pytest.fixture(scope="module")
def ep_group():
    if not torch.cuda.is_available():
        pytest.skip("EP requires CUDA")
    torch.cuda.set_device(cp.cuda.runtime.getDevice())
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("EP requires SM90 or newer")
    group = CommGroup(mpi_comm=MPI.COMM_WORLD)
    if not 1 <= group.nranks <= 64 or group.nranks_per_ipc_domain < group.nranks:
        pytest.skip("EP requires 1-64 ranks in one CUDA IPC domain")
    yield group
    torch.cuda.synchronize()
    group.barrier()


@contextmanager
def initialized_runtime(group, **kwargs):
    runtime = MoECommunicator(comm=group, device=torch.cuda.current_device(), **kwargs)
    assert runtime.is_available()
    runtime.initialize()
    runtime.initialize()
    assert runtime.is_initialized()
    try:
        yield runtime
    finally:
        torch.cuda.synchronize()
        group.barrier()


def token_routes(rank, world_size, token, num_tokens):
    if token == num_tokens - 1:
        return [-1, -1, -1]
    first = (rank + token) % world_size
    second = (rank + world_size - 1 - token) % world_size
    return [first * 4, -1 if token == 1 else first * 4 + 1, second * 4 + 2]


CASES = [
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.BF16, CombineMode.DIRECT_SEND),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.DIRECT_SEND),
    (MoEMode.LATENCY, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE),
    (MoEMode.LATENCY, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.DIRECT_SEND),
]


@pytest.mark.parametrize("mode,layout,data_type,combine_mode", CASES)
def test_dispatch_combine(ep_group, mode, layout, data_type, combine_mode):
    num_tokens, capacity, active_capacity = 5, 8, 6
    fp8 = data_type == DispatchDataType.FP8_E4M3
    hidden = 4096 if mode == MoEMode.LATENCY else 256 if fp8 else 136
    with initialized_runtime(
        ep_group,
        mode=mode,
        output_layout=layout,
        combine_mode=combine_mode,
        num_experts=4 * ep_group.nranks,
        hidden_size=hidden,
        topk=3,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            values = (torch.arange(num_tokens, device="cuda") + ep_group.my_rank * num_tokens + 1).float() * 4
            input = values[:, None].expand(-1, hidden).to(torch.bfloat16).contiguous()
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, token, num_tokens) for token in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.tensor([0.25, 0.5, 0.25], device="cuda").expand(num_tokens, -1).contiguous()
            quant = QuantConfig(format=data_type)
            dispatch_input = input
            preparation = None
            if fp8 and mode == MoEMode.THROUGHPUT:
                dispatch_input = torch.ones_like(input, dtype=torch.float8_e4m3fn)
                quant.block_scales = values[:, None].expand(-1, hidden // 128).contiguous()
                preparation = runtime.prepare(routes, stream=stream, runtime_max_tokens_per_rank=active_capacity)
            result, handle = runtime.dispatch(
                dispatch_input,
                routes,
                weights,
                quant,
                stream=stream,
                prepare_handle=preparation,
                runtime_max_tokens_per_rank=active_capacity,
            )
            assert result.tokens.data_ptr() == runtime.get_dispatch_output_buffer().data_ptr()
            assert result.tokens.dtype == (torch.float8_e4m3fn if fp8 else torch.bfloat16)
            if layout == DispatchLayout.EXPERT_MAJOR:
                counts = result.layout.num_tokens_per_expert
                valid_rows = torch.arange(ep_group.nranks * active_capacity, device="cuda")[None, :] < counts[:, None]
            elif layout == DispatchLayout.RANK_MAJOR:
                counts = result.layout.num_tokens_per_rank
                valid_rows = torch.arange(active_capacity, device="cuda")[None, :] < counts[:, None]
            else:
                counts = result.layout.num_tokens_per_expert
                valid_rows = (
                    torch.arange(ep_group.nranks * active_capacity, device="cuda") < result.layout.num_recv_tokens
                )
            tokens = result.tokens.float()
            if fp8:
                assert result.quant.block_scales.dtype == torch.float32
                assert result.quant.block_scales.shape == (*result.tokens.shape[:-1], hidden // 128)
                tokens = tokens * result.quant.block_scales.repeat_interleave(128, dim=-1)
            tokens = torch.where(valid_rows[..., None], tokens, 0)
            if layout == DispatchLayout.EXPERT_MAJOR:
                expert_output = tokens.to(torch.bfloat16)
            else:
                local_weights = torch.where(valid_rows[..., None], result.weights, 0)
                if mode == MoEMode.LATENCY and combine_mode == CombineMode.DIRECT_SEND:
                    expert_output = (tokens[..., None, :] * local_weights[..., None]).to(torch.bfloat16)
                else:
                    expert_output = (tokens * local_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
            out = torch.empty_like(input)
            combined_weights = torch.empty_like(weights) if mode == MoEMode.THROUGHPUT else None
            combined = runtime.combine(
                expert_output, handle, out=out, stream=stream, output_topk_weights=combined_weights
            )
            assert combined is out
            expected_weights = torch.where(routes >= 0, weights, 0)
            expected = input.float() * expected_weights.sum(dim=-1, keepdim=True)
        stream.synchronize()
        torch.testing.assert_close(combined.float(), expected, rtol=0, atol=1 if fp8 else 0)
        if combined_weights is not None:
            torch.testing.assert_close(combined_weights, expected_weights, rtol=0, atol=0)

        expected_counts = [0] * (ep_group.nranks if layout == DispatchLayout.RANK_MAJOR else 4)
        expected_rows = 0
        for source in range(ep_group.nranks):
            for token in range(num_tokens):
                local_routes = [
                    expert
                    for expert in token_routes(source, ep_group.nranks, token, num_tokens)
                    if expert >= 0 and expert // 4 == ep_group.my_rank
                ]
                if local_routes:
                    expected_rows += 1
                    if layout == DispatchLayout.RANK_MAJOR:
                        expected_counts[source] += 1
                if layout != DispatchLayout.RANK_MAJOR:
                    for expert in local_routes:
                        expected_counts[expert % 4] += 1
        assert counts.cpu().tolist() == expected_counts
        if mode == MoEMode.THROUGHPUT:
            assert result.layout.num_recv_tokens.item() == expected_rows


def test_preparation_reuse_and_invalidation(ep_group):
    tokens, hidden = 5, 136
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=hidden,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=tokens,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.full((tokens, hidden), ep_group.my_rank + 1, device="cuda", dtype=torch.bfloat16)
            routes = torch.full((tokens, 1), (ep_group.my_rank + 1) % ep_group.nranks, device="cuda", dtype=torch.int64)
            preparation = runtime.prepare(routes, stream=stream)
            for value in (2, 4):
                input.fill_(value)
                result, previous = runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
                output = runtime.combine(result.tokens, previous, stream=stream)
                stream.synchronize()
                torch.testing.assert_close(output, input, rtol=0, atol=0)
            result, latest = runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
            with pytest.raises(MscclppError, match="(?i)stale"):
                runtime.combine(result.tokens, previous, stream=stream)
            runtime.combine(result.tokens, latest, stream=stream)
            refreshed = runtime.prepare(routes, stream=stream)
            with pytest.raises(MscclppError, match="(?i)stale"):
                runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
            runtime.dispatch(input, routes, stream=stream, prepare_handle=refreshed)
            routes.fill_(-1)
            empty = runtime.prepare(routes, stream=stream)
            result, handle = runtime.dispatch(input, routes, stream=stream, prepare_handle=empty)
            output = runtime.combine(result.tokens, handle, stream=stream)
        stream.synchronize()
        assert result.layout.num_recv_tokens.item() == 0
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_tensor_stream_and_handle_validation(ep_group):
    tokens, hidden = 3, 136
    config = dict(
        mode=MoEMode.THROUGHPUT,
        hidden_size=hidden,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=tokens,
    )
    with initialized_runtime(ep_group, **config) as runtime, initialized_runtime(ep_group, **config) as other:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.ones((tokens, hidden), dtype=torch.bfloat16, device="cuda")
            routes = torch.full((tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            preparation = runtime.prepare(routes, stream=stream)
            for invalid in (
                input.float(),
                torch.empty((tokens, hidden * 2), device="cuda", dtype=torch.bfloat16)[:, ::2],
                torch.empty(tokens * hidden + 1, device="cuda", dtype=torch.bfloat16)[1:].reshape(tokens, hidden),
                torch.empty((tokens, hidden), dtype=torch.bfloat16),
            ):
                with pytest.raises((TypeError, ValueError)):
                    runtime.dispatch(invalid, routes, stream=stream, prepare_handle=preparation)
            with pytest.raises((TypeError, ValueError)):
                runtime.dispatch(input, routes.int(), stream=stream, prepare_handle=preparation)
            with pytest.raises((TypeError, ValueError)):
                runtime.dispatch(
                    input, routes, stream=stream, prepare_handle=preparation, runtime_max_tokens_per_rank=tokens + 1
                )
            with pytest.raises((TypeError, ValueError, RuntimeError), match="(?i)stream"):
                runtime.dispatch(input, routes, stream=torch.cuda.Stream(), prepare_handle=preparation)
            with pytest.raises((ValueError, RuntimeError)):
                other.dispatch(input, routes, stream=stream, prepare_handle=preparation)
            result, handle = runtime.dispatch(
                input, routes, quant=QuantConfig(), stream=stream, prepare_handle=preparation
            )
            with pytest.raises((ValueError, RuntimeError)):
                other.combine(result.tokens, handle, stream=stream)
            output = runtime.combine(result.tokens, handle, stream=stream)
        stream.synchronize()
        torch.testing.assert_close(output, input, rtol=0, atol=0)


def test_device_only_preparation(ep_group, monkeypatch):
    tokens, hidden = 2, 136
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=hidden,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=tokens,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.ones((tokens, hidden), dtype=torch.bfloat16, device="cuda")
            routes = torch.full((tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")

            def forbidden(*args, **kwargs):
                raise AssertionError("EP must not synchronize or copy device counts to the host")

            with monkeypatch.context() as patch:
                for name in ("item", "cpu", "tolist"):
                    patch.setattr(torch.Tensor, name, forbidden)
                patch.setattr(torch.cuda, "synchronize", forbidden)
                patch.setattr(torch.cuda.Stream, "synchronize", forbidden)
                preparation = runtime.prepare(routes, stream=stream)
                result, handle = runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
                output = runtime.combine(result.tokens, handle, stream=stream)
        stream.synchronize()
        torch.testing.assert_close(output, input, rtol=0, atol=0)


def test_empty_dispatch(ep_group):
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=136,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=2,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.empty((0, 136), dtype=torch.bfloat16, device="cuda")
            routes = torch.empty((0, 1), dtype=torch.int64, device="cuda")
            preparation = runtime.prepare(routes, stream=stream)
            result, handle = runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
            output = runtime.combine(result.tokens, handle, stream=stream)
        stream.synchronize()
        assert output.shape == (0, 136)
        assert result.layout.num_recv_tokens.item() == 0


def test_runtime_buffer_view_lifetime(ep_group):
    runtime = MoECommunicator(
        comm=ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=136,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=2,
    )
    runtime.initialize()
    tensor = runtime.get_dispatch_output_buffer()
    view = tensor.reshape(-1)[1:17]
    view.fill_(3)
    torch.cuda.synchronize()
    ep_group.barrier()
    del tensor, runtime
    gc.collect()
    torch.testing.assert_close(view, torch.full_like(view, 3), rtol=0, atol=0)
    torch.cuda.synchronize()
    ep_group.barrier()
    del view
    gc.collect()
    ep_group.barrier()


def test_external_throughput_output(ep_group):
    tokens, hidden = 3, 136
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=hidden,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=tokens,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.full((tokens, hidden), ep_group.my_rank + 1, dtype=torch.bfloat16, device="cuda")
            routes = torch.full((tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            external = torch.empty_like(runtime.get_dispatch_output_buffer())
            result, handle = runtime.dispatch(input, routes, output_buffer=external, stream=stream)
            assert result.tokens.data_ptr() == external.data_ptr()
            output = runtime.combine(result.tokens, handle, stream=stream)
        stream.synchronize()
        torch.testing.assert_close(output, input, rtol=0, atol=0)


def test_prepared_cuda_graph(ep_group):
    tokens, hidden = 3, 136
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        hidden_size=hidden,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=tokens,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.ones((tokens, hidden), dtype=torch.bfloat16, device="cuda")
            routes = torch.full((tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            preparation = runtime.prepare(routes, stream=stream)
            received = runtime.get_dispatch_output_buffer()
            output = torch.empty_like(input)
            result, handle = runtime.dispatch(
                input, routes, output_buffer=received, stream=stream, prepare_handle=preparation
            )
            runtime.combine(result.tokens, handle, out=output, stream=stream)
        stream.synchronize()
        ep_group.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result, handle = runtime.dispatch(
                input, routes, output_buffer=received, stream=stream, prepare_handle=preparation
            )
            runtime.combine(result.tokens, handle, out=output, stream=stream)
        for value in (2, 3):
            with torch.cuda.stream(stream):
                input.fill_(value)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output, input, rtol=0, atol=0)
        ep_group.barrier()
        del graph


@pytest.mark.parametrize("combine_mode", [CombineMode.RANK_LOCAL_REDUCE, CombineMode.DIRECT_SEND])
def test_stale_rank_major_combine_preserves_buffer(ep_group, combine_mode):
    with initialized_runtime(
        ep_group,
        mode=MoEMode.LATENCY,
        output_layout=DispatchLayout.RANK_MAJOR,
        combine_mode=combine_mode,
        hidden_size=4096,
        num_experts=ep_group.nranks,
        topk=1,
        max_tokens_per_rank=2,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.full((2, 4096), 4, dtype=torch.bfloat16, device="cuda")
            routes = torch.full((2, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            _, stale = runtime.dispatch(input, routes, stream=stream)
            result, current = runtime.dispatch(input, routes, stream=stream)
            result.combine_input_buffer.fill_(4)
            invalid_expert_output = torch.full_like(result.combine_input_buffer, 9)
            with pytest.raises(MscclppError, match="(?i)stale"):
                runtime.combine(invalid_expert_output, stale, stream=stream)
            output = runtime.combine(result.combine_input_buffer, current, stream=stream)
        stream.synchronize()
        torch.testing.assert_close(output, input, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,first_operation",
    [
        (MoEMode.LATENCY, "get_dispatch_output_buffer"),
        (MoEMode.THROUGHPUT, "prepare"),
        (MoEMode.THROUGHPUT, "dispatch"),
    ],
)
def test_lazy_initialization_device_scope(ep_group, monkeypatch, mode, first_operation):
    device = torch.cuda.current_device()
    other_device = (device + 1) % torch.cuda.device_count()
    hidden = 4096 if mode == MoEMode.LATENCY else 136
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        input = torch.ones((1, hidden), dtype=torch.bfloat16, device=device)
        routes = torch.full((1, 1), ep_group.my_rank, dtype=torch.int64, device=device)

    def forbidden():
        raise AssertionError("The Python wrapper must not query graph capture state")

    with torch.cuda.device(other_device), monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
        runtime = MoECommunicator(
            comm=ep_group,
            device=device,
            mode=mode,
            hidden_size=hidden,
            num_experts=ep_group.nranks,
            topk=1,
            max_tokens_per_rank=2,
        )
        assert runtime.is_available()
        assert not runtime.is_initialized()
        assert torch.cuda.current_device() == other_device
        preparation = None
        if first_operation == "get_dispatch_output_buffer":
            runtime.get_dispatch_output_buffer()
        elif first_operation == "prepare":
            preparation = runtime.prepare(routes, stream=stream)
        result, handle = runtime.dispatch(input, routes, stream=stream, prepare_handle=preparation)
        output = runtime.combine(result.tokens, handle, stream=stream)
        assert runtime.is_initialized()
        runtime.initialize()
        assert torch.cuda.current_device() == other_device
    stream.synchronize()
    ep_group.barrier()
    torch.testing.assert_close(output, input, rtol=0, atol=0)
