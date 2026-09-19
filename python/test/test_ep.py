# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end correctness coverage for the C++ EP Python interface."""

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
            if mode == MoEMode.LATENCY and layout == DispatchLayout.RANK_MAJOR:
                result.combine_input_buffer.copy_(expert_output)
                expert_output = result.combine_input_buffer

            combined_weights = torch.empty_like(weights) if mode == MoEMode.THROUGHPUT else None
            combined = runtime.combine(
                expert_output,
                handle,
                stream=stream,
                output_topk_weights=combined_weights,
            )
            expected_weights = torch.where(routes >= 0, weights, 0)
            expected = input.float() * expected_weights.sum(dim=-1, keepdim=True)
            stream.synchronize()
            torch.testing.assert_close(combined.float(), expected, rtol=0, atol=1 if fp8 else 0)
            if combined_weights is not None:
                torch.testing.assert_close(combined_weights, expected_weights, rtol=0, atol=0)

            expected_counts = [0] * (ep_group.nranks if layout == DispatchLayout.RANK_MAJOR else NUM_LOCAL_EXPERTS)
            for source in range(ep_group.nranks):
                for token in range(num_tokens):
                    local_routes = [
                        expert
                        for expert in token_routes(source, ep_group.nranks, token, num_tokens)
                        if expert >= 0 and expert // NUM_LOCAL_EXPERTS == ep_group.my_rank
                    ]
                    if layout == DispatchLayout.RANK_MAJOR and local_routes:
                        expected_counts[source] += 1
                    elif layout != DispatchLayout.RANK_MAJOR:
                        for expert in local_routes:
                            expected_counts[expert % NUM_LOCAL_EXPERTS] += 1
            assert counts.cpu().tolist() == expected_counts


def test_dispatch_combine_correctness(ep_group):
    for case in CASES:
        _run_dispatch_combine_case(ep_group, *case)
        gc.collect()
        ep_group.barrier()
