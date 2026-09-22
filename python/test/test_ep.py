# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end correctness coverage for the C++ EP Python interface."""

from contextlib import contextmanager
from dataclasses import replace
import gc
import importlib.util

import cupy as cp
from mpi4py import MPI
import pytest

torch = pytest.importorskip("torch")
if importlib.util.find_spec("mscclpp.mscclpp_ep_cpp") is None:
    pytest.skip("The CUDA EP extension was not built", allow_module_level=True)

from mscclpp import CommGroup
from mscclpp._mscclpp import Error
from mscclpp.ep import CombineMode, DispatchDataType, DispatchLayout, MoECommunicator, MoEMode, QuantConfig

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
        return [-1] * NUM_TOPK
    num_experts = NUM_LOCAL_EXPERTS * world_size
    base = (rank * NUM_LOCAL_EXPERTS + token * 7) % num_experts
    routes = [(base + topk * 5) % num_experts for topk in range(NUM_TOPK)]
    if token == 1:
        routes[1] = -1
    return routes


CASES = [
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, True),
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE, True),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, True),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE, True),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.BF16, CombineMode.DIRECT_SEND, False),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.LATENCY, DispatchLayout.EXPERT_MAJOR, DispatchDataType.FP8_E4M3, CombineMode.DIRECT_SEND, False),
    (MoEMode.LATENCY, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.RANK_LOCAL_REDUCE, False),
    (MoEMode.LATENCY, DispatchLayout.RANK_MAJOR, DispatchDataType.BF16, CombineMode.DIRECT_SEND, False),
]


def _run_dispatch_combine_case(ep_group, mode, layout, data_type, combine_mode, prepared):
    num_tokens, capacity, active_capacity = 5, 8, 6
    num_experts = NUM_LOCAL_EXPERTS * ep_group.nranks
    fp8 = data_type == DispatchDataType.FP8_E4M3
    with initialized_runtime(
        ep_group,
        mode=mode,
        output_layout=layout,
        combine_mode=combine_mode,
        num_experts=num_experts,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.tensor(
                [token_routes(ep_group.my_rank, ep_group.nranks, token, num_tokens) for token in range(num_tokens)],
                dtype=torch.int64,
                device="cuda",
            )
            weights = torch.full((num_tokens, NUM_TOPK), 1.0 / NUM_TOPK, device="cuda")
            preparation = (
                runtime.prepare(routes, stream=stream, runtime_max_tokens_per_rank=active_capacity)
                if prepared
                else None
            )

            values = (
                (torch.arange(num_tokens, device="cuda") + ep_group.my_rank * num_tokens) % 15 + 1
            ).float() * NUM_TOPK
            input = values[:, None].expand(-1, HIDDEN).to(torch.bfloat16).contiguous()
            quant = QuantConfig(format=data_type)
            dispatch_input = input
            if fp8 and mode == MoEMode.THROUGHPUT:
                dispatch_input = torch.ones_like(input, dtype=torch.float8_e4m3fn)
                quant.block_scales = values[:, None].expand(-1, HIDDEN // 128).contiguous()

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
            assert result.layout is handle.output_info.layout
            if layout != DispatchLayout.EXPERT_MAJOR:
                assert result.topk_ids.data_ptr() == runtime._runtime.output_topk_ids_buffer_ptr()
                assert result.weights.data_ptr() == runtime._runtime.output_topk_weights_buffer_ptr()
                assert result.combine_input_buffer.data_ptr() == runtime._runtime.combine_input_buffer_ptr()
            if mode == MoEMode.THROUGHPUT:
                assert result.combine_input_buffer.data_ptr() != result.tokens.data_ptr()
                dispatched_bytes = result.tokens.view(torch.uint8).clone()
                if fp8:
                    assert result.quant.block_scales.data_ptr() == runtime._runtime.output_scales_buffer_ptr()

            if layout == DispatchLayout.EXPERT_MAJOR:
                counts = result.layout.num_tokens_per_expert
                valid_rows = torch.arange(ep_group.nranks * active_capacity, device="cuda")[None, :] < counts[:, None]
            elif layout == DispatchLayout.RANK_MAJOR:
                counts = result.layout.num_tokens_per_rank
                valid_rows = torch.arange(active_capacity, device="cuda")[None, :] < counts[:, None]
            else:
                counts = result.layout.num_tokens_per_expert
                valid_rows = None

            tokens = result.tokens.float()
            if fp8:
                assert result.quant.block_scales.dtype == torch.float32
                assert result.quant.block_scales.shape == (*result.tokens.shape[:-1], HIDDEN // 128)
                tokens = tokens * result.quant.block_scales.repeat_interleave(128, dim=-1)
            if valid_rows is not None:
                tokens = torch.where(valid_rows[..., None], tokens, 0)

            if layout == DispatchLayout.EXPERT_MAJOR:
                expert_output = tokens.to(torch.bfloat16)
            else:
                local_weights = (
                    result.weights if valid_rows is None else torch.where(valid_rows[..., None], result.weights, 0)
                )
                if mode == MoEMode.LATENCY and combine_mode == CombineMode.DIRECT_SEND:
                    expert_output = (tokens[..., None, :] * local_weights[..., None]).to(torch.bfloat16)
                else:
                    expert_output = (tokens * local_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
            if result.combine_input_buffer is not None:
                result.combine_input_buffer.copy_(expert_output)
                expert_output = result.combine_input_buffer

            combined = runtime.combine(expert_output, handle, stream=stream)
            expected_weights = torch.where(routes >= 0, weights, 0)
            expected = input.float() * expected_weights.sum(dim=-1, keepdim=True)
            stream.synchronize()
            torch.testing.assert_close(combined.float(), expected, rtol=0, atol=1 if fp8 else 0)
            if mode == MoEMode.THROUGHPUT:
                torch.testing.assert_close(result.tokens.view(torch.uint8), dispatched_bytes, rtol=0, atol=0)

            expected_counts = [0] * (ep_group.nranks if layout == DispatchLayout.RANK_MAJOR else NUM_LOCAL_EXPERTS)
            received_rows, expected_ids, expected_recv_weights = [], [], []
            for source in range(ep_group.nranks):
                rank_row = 0
                for token in range(num_tokens):
                    all_routes = token_routes(source, ep_group.nranks, token, num_tokens)
                    local_routes = [
                        expert
                        for expert in all_routes
                        if expert >= 0 and expert // NUM_LOCAL_EXPERTS == ep_group.my_rank
                    ]
                    if layout == DispatchLayout.RANK_MAJOR and local_routes:
                        expected_counts[source] += 1
                    elif layout != DispatchLayout.RANK_MAJOR:
                        for expert in local_routes:
                            expected_counts[expert % NUM_LOCAL_EXPERTS] += 1
                    if mode == MoEMode.THROUGHPUT and local_routes:
                        row = (
                            source * active_capacity + rank_row
                            if layout == DispatchLayout.RANK_MAJOR
                            else len(received_rows)
                        )
                        rank_row += 1
                        received_rows.append(row)
                        expected_ids.append(
                            [expert % NUM_LOCAL_EXPERTS if expert in local_routes else -1 for expert in all_routes]
                        )
                        expected_recv_weights.append(
                            [1.0 / NUM_TOPK if expert in local_routes else 0.0 for expert in all_routes]
                        )
            assert counts.cpu().tolist() == expected_counts
            if received_rows:
                rows = torch.tensor(received_rows, device="cuda")
                torch.testing.assert_close(
                    result.topk_ids.reshape(-1, NUM_TOPK)[rows],
                    torch.tensor(expected_ids, dtype=torch.int32, device="cuda"),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    result.weights.reshape(-1, NUM_TOPK)[rows],
                    torch.tensor(expected_recv_weights, device="cuda"),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.nranks(8)
def test_dispatch_combine_correctness(ep_group):
    for case in CASES:
        _run_dispatch_combine_case(ep_group, *case)
        gc.collect()
        ep_group.barrier()


def test_preparation_validation(ep_group):
    num_tokens, capacity, active_capacity = 2, 4, 3
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.arange(NUM_TOPK, device="cuda").repeat(num_tokens, 1)
            input = torch.ones((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            preparation = runtime.prepare(routes, runtime_max_tokens_per_rank=active_capacity)

            with pytest.raises(TypeError, match="PrepareHandle"):
                runtime.dispatch(input, routes, prepare_handle=object())
            with pytest.raises(ValueError, match="belongs to another MoECommunicator"):
                runtime.dispatch(input, routes, prepare_handle=replace(preparation, _runtime=object()))
            with pytest.raises(TypeError):
                runtime.dispatch(input, routes, prepare_handle=replace(preparation, _native=object()))

            for payload, ids, active in (
                (input, routes.clone(), active_capacity),
                (input[:1], routes[:1], active_capacity),
                (input, routes, capacity),
            ):
                with pytest.raises(Error, match="must match the preparation"):
                    runtime.dispatch(payload, ids, prepare_handle=preparation, runtime_max_tokens_per_rank=active)

            result, handle = runtime.dispatch(
                input, routes, prepare_handle=preparation, runtime_max_tokens_per_rank=active_capacity
            )
            assert handle._runtime is runtime._runtime
            assert result.tokens.shape == (ep_group.nranks * active_capacity, HIDDEN)
            assert preparation._topk_ids is routes
            result.combine_input_buffer.copy_(result.tokens)
            runtime.combine(result.combine_input_buffer, handle)

            runtime.prepare(routes, runtime_max_tokens_per_rank=active_capacity)
            with pytest.raises(Error, match="Stale preparation handle"):
                runtime.dispatch(input, routes, prepare_handle=preparation, runtime_max_tokens_per_rank=active_capacity)


def test_dispatch_handle_validation(ep_group):
    num_tokens, capacity = 2, 3
    config = dict(
        mode=MoEMode.LATENCY,
        num_experts=NUM_LOCAL_EXPERTS * ep_group.nranks,
        hidden_size=HIDDEN,
        topk=NUM_TOPK,
        max_tokens_per_rank=capacity,
    )
    with initialized_runtime(ep_group, **config) as runtime:
        other = MoECommunicator(comm=ep_group, device=torch.cuda.current_device(), **config)
        stream = torch.cuda.Stream()
        other_stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            routes = torch.arange(NUM_TOPK, device="cuda").repeat(num_tokens, 1)
            input = torch.ones((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            with pytest.raises(ValueError, match="dtype"):
                runtime.dispatch(input.float(), routes, stream=other_stream)
            assert runtime._stream is None

            result, handle = runtime.dispatch(input, routes)
            assert runtime._stream.cuda_stream == stream.cuda_stream
            with pytest.raises(TypeError, match="DispatchHandle"):
                runtime.combine(result.tokens, object())
            with pytest.raises(ValueError, match="belongs to another MoECommunicator"):
                other.combine(result.tokens, handle)
            for invalid in (
                dict(_num_tokens=-1),
                dict(_num_tokens=capacity + 1),
                dict(_num_tokens=True),
                dict(_active_capacity=0),
                dict(_active_capacity=capacity + 1),
                dict(_active_capacity=1),
            ):
                with pytest.raises(ValueError, match="invalid token count or capacity"):
                    runtime.combine(result.tokens, replace(handle, **invalid))
            with pytest.raises(TypeError):
                runtime.combine(result.tokens, replace(handle, _native=object()))
            with pytest.raises(ValueError, match="another CUDA stream"):
                runtime.combine(result.tokens, handle, stream=other_stream)

            combined = runtime.combine(result.tokens, handle)
            stream.synchronize()
            torch.testing.assert_close(combined, input * NUM_TOPK, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,layout",
    [
        (MoEMode.LATENCY, DispatchLayout.RANK_MAJOR),
        (MoEMode.THROUGHPUT, DispatchLayout.TOKEN_MAJOR),
        (MoEMode.THROUGHPUT, DispatchLayout.RANK_MAJOR),
    ],
)
def test_runtime_owned_buffers(ep_group, mode, layout):
    num_tokens, capacity, active_capacity = 2, 4, 3
    with initialized_runtime(
        ep_group,
        mode=mode,
        output_layout=layout,
        num_experts=ep_group.nranks,
        hidden_size=HIDDEN,
        topk=1,
        max_tokens_per_rank=capacity,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.ones((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            routes = torch.full((num_tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            buffer = runtime.get_dispatch_output_buffer(runtime_max_tokens_per_rank=active_capacity)
            with pytest.raises(ValueError, match="runtime-owned dispatch buffer"):
                runtime.dispatch(
                    input,
                    routes,
                    output_buffer=torch.empty_like(buffer),
                    runtime_max_tokens_per_rank=active_capacity,
                )
            assert runtime._stream is None
            result, handle = runtime.dispatch(
                input, routes, output_buffer=buffer, runtime_max_tokens_per_rank=active_capacity
            )
            assert result.tokens.data_ptr() == buffer.data_ptr()
            dispatched = result.tokens.clone()
            combine_input = result.combine_input_buffer
            combine_input.fill_(7)
            with pytest.raises(ValueError, match="DispatchOutput.combine_input_buffer"):
                runtime.combine(torch.empty_like(combine_input), handle)
            torch.testing.assert_close(combine_input, torch.full_like(combine_input, 7), rtol=0, atol=0)
            with pytest.raises(TypeError, match="output_topk_weights"):
                runtime.combine(combine_input, handle, output_topk_weights=torch.empty((num_tokens, 1), device="cuda"))

            combine_input.copy_(dispatched)
            output = torch.empty_like(input)
            combined = runtime.combine(combine_input, handle, out=output)
            stream.synchronize()
            assert combined is output
            torch.testing.assert_close(combined, input, rtol=0, atol=0)


@pytest.mark.parametrize("layout", [DispatchLayout.TOKEN_MAJOR, DispatchLayout.RANK_MAJOR])
@pytest.mark.parametrize("num_tokens", [0, 2])
def test_throughput_graph_replay(ep_group, layout, num_tokens):
    with initialized_runtime(
        ep_group,
        mode=MoEMode.THROUGHPUT,
        output_layout=layout,
        num_experts=ep_group.nranks,
        hidden_size=128,
        topk=1,
        max_tokens_per_rank=4,
    ) as runtime:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input = torch.ones((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
            routes = torch.full((num_tokens, 1), ep_group.my_rank, dtype=torch.int64, device="cuda")
            preparation = runtime.prepare(routes, runtime_max_tokens_per_rank=3)

            def operation():
                result, handle = runtime.dispatch(
                    input, routes, prepare_handle=preparation, runtime_max_tokens_per_rank=3
                )
                result.combine_input_buffer.copy_(result.tokens)
                return result, handle, runtime.combine(result.combine_input_buffer, handle)

            operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result, handle, output = operation()
            for value in (2, 3):
                input.fill_(value)
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(output, input, rtol=0, atol=0)
            if num_tokens == 0:
                counts = (
                    result.layout.num_tokens_per_rank
                    if layout == DispatchLayout.RANK_MAJOR
                    else result.layout.num_tokens_per_expert
                )
                torch.testing.assert_close(counts, torch.zeros_like(counts), rtol=0, atol=0)
