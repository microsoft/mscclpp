# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routing regressions, with native tests gated by MSCCLPP_TEST_MEGAMOE_ROUTING=1.

Run with ``python -m pytest --noconftest python/test/test_megamoe_routing.py``.
For two or four NVLink-connected SM100-family GPUs, use ``torchrun --nnodes=1
--master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 --module pytest --noconftest``
with the same file. Gloo is
only used for multi-rank rendezvous and the CPU oracle; MSCCL++ uses MASTER_PORT+1.
Set MSCCLPP_TEST_MEGAMOE_JIT=1 as well to exercise non-builtin specializations.
The configuration-selection tests also support EP32 across eight four-GPU nodes;
select them with ``-k configuration_selection``.
"""

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import os
import subprocess
from types import SimpleNamespace

import pytest

_JIT_ONLY = pytest.mark.skipif(
    os.environ.get("MSCCLPP_TEST_MEGAMOE_JIT") != "1",
    reason="set MSCCLPP_TEST_MEGAMOE_JIT=1 as well to compile routed specializations",
)


def _routing_reference(config, sample, weights):
    """Match the configured quantization and FP32 slot sum, entirely on CPU."""
    from mscclpp.ext.megamoe.benchmark import _reference

    assert all(tensor.device.type == "cpu" for tensor in (*sample, *weights))
    return _reference(config, *sample, weights)


def _sample(config, tokens, pattern, *, valid_tokens=None):
    import torch

    generator = torch.Generator().manual_seed(4813 + config.rank)
    inputs = (torch.randn((tokens, config.hidden), generator=generator) * 0.125).to(torch.bfloat16)
    rows = torch.arange(tokens).view(-1, 1)
    slots = torch.arange(config.top_k).view(1, -1)
    ordinal = rows * config.top_k + slots
    phases = {"hot_first": 0, "masked": 1, "hot_last": 2, "balanced": 3, "mixed": 4, "random": 5, "sweep": 6}
    scores = torch.tensor([-1.25, 0.0, 0.375, 0.7, 1.875], dtype=torch.float32)
    scores = scores[(ordinal + phases[pattern]) % scores.numel()].contiguous()
    owner = (rows + slots + config.rank) % config.world_size
    ids = (owner * config.local_experts + ordinal % config.local_experts).to(torch.int32)
    if pattern == "hot_first":
        ids.zero_()
    elif pattern == "hot_last":
        ids.fill_(config.num_experts - 1)
    elif pattern == "masked":
        ids.fill_(-1)
    elif pattern == "random":
        ids = torch.randint(config.num_experts, ids.shape, generator=generator, dtype=torch.int32)
    elif pattern == "mixed":
        if config.top_k > 1:
            ids[::3, 1] = ids[::3, 0]
        ids[(rows + 2 * slots) % 5 == 0] = -1
        ids[torch.arange(tokens) % 11 == 3] = -1
    elif pattern == "sweep":
        # Across ranks, every local expert is reached, including expert 128.
        scores.fill_(0.75)
    if tokens and pattern in ("balanced", "mixed", "random"):
        # Keep the last capacity slot live, including a thread's second cached ID.
        ids[-1] = config.num_experts - 1
        scores[-1] = 0.75
    if valid_tokens is not None:
        ids[valid_tokens:] = -1
    scores[ids == -1] = float("nan")
    inputs[(ids == -1).all(dim=1)] = float("nan")
    return inputs, ids.contiguous(), scores


def _host_weights(config):
    import torch
    from mscclpp.ext.megamoe.benchmark import _weights

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3701 + config.rank)
        return _weights(config, torch.device("cpu"))


class _Runtime:
    def __init__(self, torch, dist, rank, world, device, bootstrap, communicator):
        self.torch, self.dist = torch, dist
        self.rank, self.world, self.device = rank, world, device
        self.bootstrap, self.communicator = bootstrap, communicator
        self.next_tag = 23100

    def barrier(self):
        self.bootstrap.barrier()

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)
        self.barrier()

    def collective(self, operation):
        """Do not let one rank's assertion leave other ranks launching collectives."""
        result, error = None, None
        try:
            result = operation()
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            MemoryError,
            subprocess.SubprocessError,
        ) as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [error]
        if self.world > 1:
            errors = [None] * self.world
            self.dist.all_gather_object(errors, error)
        if any(errors):
            pytest.fail("\n".join(message for message in errors if message), pytrace=False)
        return result

    def config(self, capacity, *, local_experts=8, top_k=2, clamp=-1.0, hidden=4096, intermediate=6656):
        from mscclpp.ext.megamoe import MegaMoEConfig

        sms = self.torch.cuda.get_device_properties(self.device).multi_processor_count
        return MegaMoEConfig(
            rank=self.rank,
            world_size=self.world,
            max_tokens=capacity,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=local_experts * self.world,
            top_k=top_k,
            sm_margin=max(0, sms - 8),
            gate_up_clamp=clamp,
        )


@pytest.fixture(scope="module")
def _native_runtime():
    if os.environ.get("MSCCLPP_TEST_MEGAMOE_ROUTING") != "1":
        pytest.skip("set MSCCLPP_TEST_MEGAMOE_ROUTING=1 to opt into native SM100-family routing tests")
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import is_available

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world not in (1, 2, 4, 32):
        pytest.skip("native tests support one, two, four, or 32 ranks")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("run routing tests without pytest-xdist")
    owned_group = False
    bootstrap = communicator = runtime = None
    try:
        if world > 1:
            if dist.is_initialized():
                pytest.fail("use a dedicated torchrun process group for these tests")
            dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(minutes=5))
            owned_group = True
        reason = None
        if not is_available() or not torch.cuda.is_available() or torch.version.hip:
            reason = "requires native MegaMoE and NVIDIA CUDA"
        elif local_rank >= torch.cuda.device_count():
            reason = f"LOCAL_RANK={local_rank} has no visible CUDA device"
        else:
            torch.cuda.set_device(local_rank)
            if torch.cuda.get_device_capability(local_rank) not in ((10, 0), (10, 3), (10, 7)):
                reason = "requires the SM100 family on every participating rank"
        reasons = [reason]
        if world > 1:
            reasons = [None] * world
            dist.all_gather_object(reasons, reason)
        if any(reasons):
            pytest.skip("; ".join(message for message in reasons if message))
        device = torch.device("cuda", local_rank)
        bootstrap = TcpBootstrap.create(rank, world)
        if world == 1:
            bootstrap.initialize(TcpBootstrap.create_unique_id())
        else:
            port = int(os.environ["MASTER_PORT"]) + 1
            if not 1 <= port <= 65535:
                pytest.fail("MASTER_PORT must leave MASTER_PORT+1 available for MSCCL++")
            bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
        communicator = Communicator(bootstrap)
        runtime = _Runtime(torch, dist, rank, world, device, bootstrap, communicator)
        yield runtime
    finally:
        if runtime is not None:
            runtime.synchronize()
            runtime.communicator = None
            communicator = None
            gc.collect()
            runtime.barrier()
            runtime.bootstrap = None
            bootstrap = None
            gc.collect()
        if owned_group:
            dist.barrier()
            dist.destroy_process_group()


@pytest.fixture
def routing_runtime(_native_runtime):
    if _native_runtime.world not in (1, 2, 4):
        pytest.skip("routing tests support one, two, or four ranks")
    return _native_runtime


@pytest.fixture
def configuration_runtime(_native_runtime):
    if _native_runtime.world not in (4, 32):
        pytest.skip("configuration-selection tests require EP4 or EP32")
    return _native_runtime


@contextmanager
def _native_case(runtime):
    """Keep graphs and native allocations alive through every rank's peer accesses."""
    from mscclpp.ext.megamoe import MegaMoE

    contexts, graphs = [], []
    stream = runtime.torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(runtime.torch.cuda.current_stream(runtime.device))

    def create(config, weights, kernel=None):
        tag = runtime.next_tag
        runtime.next_tag += 1
        with runtime.torch.cuda.stream(stream):
            device_weights = [weight.to(runtime.device) for weight in weights]
            context = MegaMoE(config, runtime.communicator, *device_weights, stream=stream, tag=tag, kernel=kernel)
        contexts.append(context)
        return context

    try:
        yield SimpleNamespace(create=create, graphs=graphs, stream=stream)
    finally:
        runtime.synchronize()
        for graph in graphs:
            graph.reset()
        graphs.clear()
        contexts.clear()
        gc.collect()
        runtime.barrier()


def _buffers(runtime, context, direct=False):
    torch, config = runtime.torch, context.config
    inputs = (
        context.input_view()
        if direct
        else torch.empty((config.max_tokens, config.hidden), dtype=torch.bfloat16, device=runtime.device)
    )
    ids = torch.empty((config.max_tokens, config.top_k), dtype=torch.int32, device=runtime.device)
    scores = torch.empty_like(ids, dtype=torch.float32)
    guard = torch.empty((config.max_tokens + 2, config.hidden), dtype=torch.bfloat16, device=runtime.device)
    return SimpleNamespace(
        inputs=inputs,
        ids=ids,
        scores=scores,
        guard=guard,
        pointers=tuple(tensor.data_ptr() for tensor in (inputs, ids, scores, guard)),
    )


def _stage(buffers, sample):
    tokens = sample[0].shape[0]
    buffers.inputs.fill_(float("nan"))
    buffers.ids.fill_(-1)
    buffers.scores.fill_(float("nan"))
    for destination, source in zip((buffers.inputs, buffers.ids, buffers.scores), sample):
        destination[:tokens].copy_(source)
    buffers.guard.fill_(7)
    buffers.guard[1 : tokens + 1].fill_(float("nan"))


def _launch(context, buffers, tokens, stream):
    return context(
        buffers.inputs[:tokens],
        buffers.ids[:tokens],
        buffers.scores[:tokens],
        output=buffers.guard[1 : tokens + 1],
        stream=stream,
        validate_routing=False,
    )


def _check_output(runtime, buffers, sample, expected):
    torch = runtime.torch
    tokens = sample[0].shape[0]
    host = buffers.guard.cpu()
    output = host[1 : tokens + 1]

    def check():
        assert output.dtype == torch.bfloat16 and output.shape == expected.shape
        assert torch.isfinite(output).all(), "masked NaN inputs/scores leaked into the output"
        assert torch.all(host[:1] == 7) and torch.all(host[tokens + 1 :] == 7), "output guard overwritten"
        assert tuple(tensor.data_ptr() for tensor in (buffers.inputs, buffers.ids, buffers.scores, buffers.guard)) == (
            buffers.pointers
        )
        ids, scores = sample[1:]
        inactive = torch.where(ids >= 0, scores, 0).eq(0).all(dim=1)
        assert torch.count_nonzero(output[inactive]) == 0, "masked or zero-score routes retained stale output"
        if expected.numel():
            error = torch.linalg.vector_norm(output.float() - expected.float()).item()
            norm = torch.linalg.vector_norm(expected.float()).item()
            if norm == 0:
                assert torch.count_nonzero(output) == 0
            else:
                assert error / norm < 0.02, f"relative L2 error {error / norm:.6g} >= 0.02"

    runtime.collective(check)
    return output


@pytest.mark.parametrize("top_k", [1, 2])
def test_native_routing_capacity_boundary(routing_runtime, top_k):
    runtime, torch = routing_runtime, routing_runtime.torch
    routing_slots = 4096
    capacity = routing_slots // (runtime.world * top_k)
    configs = [runtime.config(capacity + extra, top_k=top_k) for extra in (0, 1)]
    assert configs[0].world_size * configs[0].max_tokens * top_k == routing_slots
    assert configs[1].world_size * configs[1].max_tokens * top_k > routing_slots
    weights = _host_weights(configs[0])
    with _native_case(runtime) as case:
        contexts = [case.create(config, weights) for config in configs]
        buffers = [_buffers(runtime, context) for context in contexts]
        for tokens, pattern in ((33, "mixed"), (capacity, "balanced")):
            sample = _sample(configs[0], tokens, pattern)
            expected = _routing_reference(configs[0], sample, weights)
            outputs = []
            for context, storage in zip(contexts, buffers):
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    _launch(context, storage, tokens, case.stream)
                case.stream.synchronize()
                outputs.append(_check_output(runtime, storage, sample, expected))
            runtime.collective(
                lambda: torch.testing.assert_close(
                    outputs[0].view(torch.int16), outputs[1].view(torch.int16), rtol=0, atol=0
                )
            )
        # Exercise the first capacity beyond the precision-specific planner limit.
        sample = _sample(configs[1], capacity + 1, "balanced")
        expected = _routing_reference(configs[1], sample, weights)
        with torch.cuda.stream(case.stream):
            _stage(buffers[1], sample)
            _launch(contexts[1], buffers[1], capacity + 1, case.stream)
        case.stream.synchronize()
        _check_output(runtime, buffers[1], sample, expected)


@pytest.mark.parametrize("local_experts", [128, 129])
def test_native_routing_local_expert_boundary(routing_runtime, local_experts):
    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(129, local_experts=local_experts, top_k=1)
    assert config.world_size * config.max_tokens * config.top_k <= 4096
    weights = _host_weights(config)
    sample = _sample(config, 129, "sweep")
    expected = _routing_reference(config, sample, weights)
    with _native_case(runtime) as case:
        context = case.create(config, weights)
        buffers = _buffers(runtime, context)
        with torch.cuda.stream(case.stream):
            _stage(buffers, sample)
            _launch(context, buffers, 129, case.stream)
        case.stream.synchronize()
        _check_output(runtime, buffers, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("cluster_size,num_warps,ctas", [(2, 16, 2), (1, 12, 1), (1, 16, 3)])
@pytest.mark.parametrize("fixed_token_count", [False, True])
def test_native_routing_reservation_boundary_graph_reuse(
    routing_runtime, fixed_token_count, cluster_size, num_warps, ctas
):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    policy = W4A8KernelConfig(
        cluster_size=cluster_size,
        num_warps=num_warps,
        load_stages=5 if cluster_size == 1 else 8,
        fixed_token_count=fixed_token_count,
        expert_major_tasks=True,
        adaptive_cache_weights=True,
    )
    # Sizes around four routes per resident thread make warps scan several 32-route groups.
    routing_slots = ctas * policy.num_warps * 32 * 4
    boundary = routing_slots // (runtime.world * 2)
    config = runtime.config(boundary + 1, local_experts=2, top_k=2)
    sms = torch.cuda.get_device_properties(runtime.device).multi_processor_count
    config = replace(config, sm_margin=sms - ctas)
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, module)

        def check_policy():
            assert context.cta_count == ctas
            assert context.effective_kernel_config["adaptive_cache_weights"] is True

        runtime.collective(check_policy)
        storage = _buffers(runtime, context)
        graphs = {}
        for tokens in (0, boundary - 1, boundary, boundary + 1):
            sample = _sample(config, tokens, "masked")
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            graphs[tokens] = graph
        for tokens in (boundary + 1, 0, boundary, boundary - 1, 0, boundary + 1):
            for pattern in ("mixed", "masked", "hot_last"):
                valid_tokens = policy.tile_n + 2 if pattern == "hot_last" else 3
                sample = _sample(config, tokens, pattern, valid_tokens=valid_tokens)
                if tokens and pattern != "masked":
                    sample[0][-1].fill_(0.125)
                    sample[1][-1].fill_(config.num_experts - 1)
                    sample[2][-1].fill_(0.75)
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    for _ in range(3):
                        graphs[tokens].replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("fixed_token_count", [False, True])
def test_native_routing_scratch_overflow_graph_reuse(routing_runtime, fixed_token_count):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    # Balanced 2048-token samples exceed one CTA's N128 two-stage route cache, as do 1024-token hot samples on
    # multiple ranks. Eight CTAs keep every sample cached, including received routes on the hot expert's rank.
    policy = W4A8KernelConfig(cluster_size=1, load_stages=2, fixed_token_count=fixed_token_count)
    config = runtime.config(2048, local_experts=8, top_k=8)
    sms = torch.cuda.get_device_properties(runtime.device).multi_processor_count
    configs = [replace(config, sm_margin=sms - 1), config]
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        contexts = [case.create(config, weights, module) for config in configs]

        def check_ctas():
            assert [context.cta_count for context in contexts] == [1, 8]

        runtime.collective(check_ctas)
        buffers = [_buffers(runtime, context) for context in contexts]
        graphs = {}
        for tokens in (1024, 2048):
            sample = _sample(config, tokens, "masked")
            for index, (context, storage) in enumerate(zip(contexts, buffers)):
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    _launch(context, storage, tokens, case.stream)
                runtime.synchronize()
                graph = torch.cuda.CUDAGraph()
                case.graphs.append(graph)
                with torch.cuda.graph(graph, stream=case.stream):
                    _launch(context, storage, tokens, case.stream)
                graphs[index, tokens] = graph
        references = {}
        for tokens, patterns in ((2048, ("balanced", "mixed")), (1024, ("hot_first", "random")), (2048, ("balanced",))):
            for pattern in patterns:
                sample = _sample(config, tokens, pattern)
                if (tokens, pattern) not in references:
                    references[tokens, pattern] = _routing_reference(config, sample, weights)
                outputs = []
                for index, storage in enumerate(buffers):
                    with torch.cuda.stream(case.stream):
                        _stage(storage, sample)
                        for _ in range(2):
                            graphs[index, tokens].replay()
                    case.stream.synchronize()
                    outputs.append(_check_output(runtime, storage, sample, references[tokens, pattern]))
                runtime.collective(
                    lambda: torch.testing.assert_close(
                        outputs[0].view(torch.int16), outputs[1].view(torch.int16), rtol=0, atol=0
                    )
                )


@pytest.mark.parametrize("capacity,direct", [(33, False), (129, True), (512, True)])
@pytest.mark.parametrize("top_k", [1, 2])
def test_native_w4a8_chunk_readiness_graph_reuse(routing_runtime, capacity, direct, top_k):
    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(capacity, local_experts=2, top_k=top_k, hidden=4096, intermediate=6656)
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights)
        storage = _buffers(runtime, context, direct)
        graphs = {}
        counts = {count: max(0, count - runtime.rank) for count in (0, 1, capacity)}
        for nominal, tokens in counts.items():
            with torch.cuda.stream(case.stream):
                _stage(storage, _sample(config, tokens, "hot_first"))
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            graphs[nominal] = graph
        for nominal in (capacity, 1, 0, capacity):
            tokens = counts[nominal]
            for pattern in ("hot_first", "masked", "hot_last", "mixed", "random"):
                sample = _sample(config, tokens, pattern)
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    graphs[nominal].replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)


@pytest.mark.parametrize("use_jit", [False, pytest.param(True, marks=_JIT_ONLY)], ids=["native", "jit-split-two-stage"])
def test_native_w4a8_runtime_geometry_ragged_graph_reuse(routing_runtime, use_jit):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(128, local_experts=2, top_k=2, hidden=8192, intermediate=4096)
    kernel = None
    if use_jit:
        policy = W4A8KernelConfig(load_stages=5, split_pipelines=True, dispatch_chunk=2048, dispatch_stages=2)
        kernel = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, kernel)
        storage = _buffers(runtime, context)
        counts = {count: max(0, count - runtime.rank) for count in (0, 1, 5)}
        graphs = {}
        for nominal, tokens in counts.items():
            with torch.cuda.stream(case.stream):
                _stage(storage, _sample(config, tokens, "hot_first"))
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            graphs[nominal] = graph

        def check_selected_kernel():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.cuda.stream(case.stream):
                    graphs[5].replay()
                case.stream.synchronize()
            kernels = [
                event.name.replace(" ", "")
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and "megaMoeW4A8" in event.name
            ]
            assert len(kernels) == 1 and "megaMoeW4A8<8192,4096,false>" in kernels[0], kernels
            copies = [
                event.name
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and "memcpy" in event.name.lower()
            ]
            assert not copies, f"caller inputs must be staged inside the launch: {copies}"
            if kernel is not None:
                assert context.kernel_config == kernel.config

        runtime.collective(check_selected_kernel)
        for nominal in (5, 1, 0, 5):
            tokens = counts[nominal]
            for pattern in ("hot_first", "masked", "hot_last", "mixed"):
                sample = _sample(config, tokens, pattern)
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    for _ in range(3):
                        graphs[nominal].replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)


@pytest.mark.parametrize(
    "hidden,capacity,intermediate,clamp,local_experts,top_k,remaining_sms,expected_tile_n,expected_load_stages",
    [
        (9216, 32, 4096, -1.0, 16, 8, 8, 128, 8),
        (9216, 64, 4096, -1.0, 16, 8, 8, 128, 8),
        (9216, 128, 4096, -1.0, 16, 8, 8, 128, 8),
        (9216, 64, 4608, -7.0, 16, 8, 8, 128, 8),
        (9216, 64, 4096, -1.0, 16, 8, 2, 128, 8),
        (9216, 65, 4096, -1.0, 16, 8, 8, 128, 8),
        (9216, 64, 4096, 0.125, 16, 8, 8, 128, 8),
        (9216, 64, 4096, -1.0, 8, 8, 8, 128, 8),
        (9216, 64, 4096, -1.0, 16, 7, 8, 128, 8),
        (8192, 16, 4096, -1.0, 16, 8, 8, 128, 8),
        (8192, 32, 4096, -1.0, 16, 8, 8, 128, 8),
        (8192, 64, 4096, -1.0, 16, 8, 8, 128, 8),
        (8192, 128, 4096, -1.0, 16, 8, 8, 128, 8),
        (8192, 128, 4096, -1.0, 4, 3, 8, 128, 8),
        (4096, 128, 6656, -1.0, 32, 8, 8, 128, 8),
        (4096, 190, 6656, -1.0, 32, 8, 8, 128, 8),
    ],
    ids=[
        "specialized-capacity-32",
        "specialized-4k",
        "specialized-capacity-128",
        "specialized-4k5",
        "specialized-low-ctas",
        "generic-capacity",
        "generic-clamp",
        "generic-experts",
        "generic-top-k",
        "h8192-specialized-capacity-16",
        "h8192-specialized-capacity-32",
        "h8192-specialized-capacity-64",
        "h8192-specialized-capacity-128",
        "h8192-runtime-experts-top-k",
        "h4096-i6656-capacity-128",
        "h4096-i6656-capacity-190",
    ],
)
def test_native_w4a8_configuration_selection_graph_reuse(
    configuration_runtime,
    hidden,
    capacity,
    intermediate,
    clamp,
    local_experts,
    top_k,
    remaining_sms,
    expected_tile_n,
    expected_load_stages,
):
    runtime, torch = configuration_runtime, configuration_runtime.torch
    config = runtime.config(
        capacity,
        local_experts=local_experts,
        top_k=top_k,
        clamp=clamp,
        hidden=hidden,
        intermediate=intermediate,
    )
    sms = torch.cuda.get_device_properties(runtime.device).multi_processor_count
    config = replace(config, sm_margin=max(0, sms - remaining_sms))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights)
        storage = _buffers(runtime, context)
        graphs = {}
        # Keep H8192 and EP32 CPU oracles small while retaining the specialized capacity.
        sample_capacity = (
            4 if hidden in (4096, 8192) else ((8 if remaining_sms == 2 else 4) if runtime.world == 32 else capacity)
        )
        if remaining_sms == 2:

            def check_planner_fallback():
                assert context.cta_count == 2
                assert config.world_size * sample_capacity * config.top_k > context.cta_count * 512

            runtime.collective(check_planner_fallback)
        reference_config = replace(config, max_tokens=sample_capacity)
        counts = {count: count for count in (0, 1, sample_capacity)}
        for nominal, tokens in counts.items():
            with torch.cuda.stream(case.stream):
                _stage(storage, _sample(config, tokens, "hot_first"))
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            graphs[nominal] = graph

        def check_selected_kernel():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.cuda.stream(case.stream):
                    graphs[sample_capacity].replay()
                case.stream.synchronize()
            kernels = [
                event.name.replace(" ", "")
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and "megaMoeW4A8" in event.name
            ]
            expected = f"megaMoeW4A8<{hidden},{intermediate},false>"
            assert len(kernels) == 1 and expected in kernels[0], (expected, kernels)
            assert context.effective_kernel_config["tile_n"] == expected_tile_n
            assert context.effective_kernel_config["load_stages"] == expected_load_stages

        runtime.collective(check_selected_kernel)
        for nominal in (sample_capacity, 1, 0, sample_capacity):
            tokens = counts[nominal]
            for pattern in ("hot_first", "masked", "hot_last", "mixed"):
                sample = _sample(config, tokens, pattern)
                expected = _routing_reference(reference_config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    for _ in range(3):
                        graphs[nominal].replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("remaining_sms", [2, 8])
def test_native_w4a8_decode_fixed_rows_masked_graph_reuse(routing_runtime, remaining_sms):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(129, local_experts=2, top_k=2, hidden=4096, intermediate=6656)
    sms = torch.cuda.get_device_properties(runtime.device).multi_processor_count
    config = replace(config, sm_margin=max(0, sms - remaining_sms))
    policy = W4A8KernelConfig(
        load_stages=5, fixed_token_count=True, split_pipelines=True, dispatch_chunk=2048, dispatch_stages=2
    )
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, module)
        storage = _buffers(runtime, context)
        graphs = {}

        def check_fixed_policy():
            assert context.effective_kernel_config["fixed_token_count"] is True

        runtime.collective(check_fixed_policy)
        for tokens in (0, 129):
            sample = _sample(config, tokens, "hot_first", valid_tokens=max(0, tokens - runtime.rank))
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            graphs[tokens] = graph

        def check_selected_kernel():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.cuda.stream(case.stream):
                    graphs[129].replay()
                case.stream.synchronize()
            kernels = [
                event.name.replace(" ", "")
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and "megaMoeW4A8" in event.name
            ]
            assert len(kernels) == 1 and "megaMoeW4A8<4096,6656,true>" in kernels[0], kernels
            assert context.cta_count <= remaining_sms

        runtime.collective(check_selected_kernel)
        for tokens in (129, 0, 129):
            for pattern in ("hot_first", "mixed", "masked", "hot_last"):
                sample = _sample(config, tokens, pattern, valid_tokens=min(tokens, runtime.rank + 1))
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    for _ in range(3):
                        graphs[tokens].replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)
            # Eager prefill can still submit different row counts on each rank.
            eager_tokens = max(0, 5 - runtime.rank)
            sample = _sample(config, eager_tokens, "mixed")
            expected = _routing_reference(config, sample, weights)
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                _launch(context, storage, eager_tokens, case.stream)
            case.stream.synchronize()
            _check_output(runtime, storage, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("clamp", [-1.0, 0.125])
@pytest.mark.parametrize("cluster_size", [1, 2], ids=["cta1", "cta2"])
@pytest.mark.parametrize(
    "hidden,intermediate,expert_major_tasks,cache_weights,adaptive_cache_weights",
    [
        (4096, 6656, False, False, False),
        (4096, 6656, True, False, False),
        (4096, 6656, False, True, False),
        (4096, 6656, True, True, False),
        (8192, 4096, True, True, False),
        (9216, 4096, True, True, False),
        (9216, 4608, True, True, False),
        (4096, 6656, False, False, True),
        (4096, 6656, True, False, True),
        (8192, 4096, True, False, True),
        (9216, 4096, True, False, True),
    ],
    ids=[
        "h4096-i6656-default",
        "h4096-i6656-order",
        "h4096-i6656-cache",
        "h4096-i6656-both",
        "h8192-i4096-n128",
        "h9216-i4096-n128",
        "h9216-i4608-n128",
        "h4096-i6656-adaptive",
        "h4096-i6656-order-adaptive",
        "h8192-i4096-n128-adaptive",
        "h9216-i4096-n128-adaptive",
    ],
)
def test_native_w4a8_expert_task_order_graph_reuse(
    routing_runtime,
    clamp,
    cluster_size,
    hidden,
    intermediate,
    expert_major_tasks,
    cache_weights,
    adaptive_cache_weights,
):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    capacity = 129
    config = runtime.config(capacity, local_experts=3, top_k=2, clamp=clamp, hidden=hidden, intermediate=intermediate)
    policy = W4A8KernelConfig(
        load_stages=5 if cluster_size == 1 else 7,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
        expert_major_tasks=expert_major_tasks,
        cache_weights=cache_weights,
        adaptive_cache_weights=adaptive_cache_weights,
        cluster_size=cluster_size,
    )
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, module)

        def check_policy():
            effective = context.effective_kernel_config
            assert effective["expert_major_tasks"] is expert_major_tasks
            assert effective["cache_weights"] is cache_weights
            assert effective["adaptive_cache_weights"] is adaptive_cache_weights
            assert effective["cluster_size"] == cluster_size
            assert effective["tile_m"] == 128 * cluster_size
            assert context.cta_count >= cluster_size and context.cta_count % cluster_size == 0

        runtime.collective(check_policy)
        storage = _buffers(runtime, context)
        with torch.cuda.stream(case.stream):
            _stage(storage, _sample(config, capacity, "hot_first"))
            _launch(context, storage, capacity, case.stream)
        runtime.synchronize()
        graph = torch.cuda.CUDAGraph()
        case.graphs.append(graph)
        with torch.cuda.graph(graph, stream=case.stream):
            _launch(context, storage, capacity, case.stream)
        peer = (runtime.rank + 1) % runtime.world
        for active in (policy.tile_n // 2, policy.tile_n // 2 + 1, capacity, 0, policy.tile_n // 2):
            sample = _sample(config, capacity, "hot_first", valid_tokens=active)
            sample[1][:active].fill_(peer * config.local_experts)
            # Duplicated slots reach the next rank with exactly N or N+2 expert rows.
            assert active * config.top_k == int((sample[1] >= 0).sum())
            expected = _routing_reference(config, sample, weights)
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                for _ in range(3):
                    graph.replay()
            case.stream.synchronize()
            _check_output(runtime, storage, sample, expected)
        for pattern in ("hot_first", "hot_last", "mixed", "masked", "random", "two_hot_peers"):
            sample = _sample(config, capacity, "random" if pattern == "two_hot_peers" else pattern)
            if pattern == "two_hot_peers":
                local_ids = (torch.arange(capacity) % 2 * 2).to(torch.int32)
                sample[1].copy_((peer * config.local_experts + local_ids[:, None]).expand_as(sample[1]))
                counts = torch.bincount(local_ids, minlength=config.local_experts) * config.top_k
                assert counts.tolist() == [policy.tile_n + 2, 0, policy.tile_n]
            expected = _routing_reference(config, sample, weights)
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                for _ in range(3):
                    graph.replay()
            case.stream.synchronize()
            _check_output(runtime, storage, sample, expected)
        # Eager execution retains unequal per-rank token counts after fixed-shape graph replays.
        tokens = max(0, capacity - runtime.rank)
        sample = _sample(config, tokens, "mixed")
        expected = _routing_reference(config, sample, weights)
        with torch.cuda.stream(case.stream):
            _stage(storage, sample)
            _launch(context, storage, tokens, case.stream)
        case.stream.synchronize()
        _check_output(runtime, storage, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("flag", ["expert_major_tasks", "cache_weights", "adaptive_cache_weights", "cluster_size"])
def test_native_w4a8_optimization_flags_must_match_across_ranks(routing_runtime, flag):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime = routing_runtime
    if runtime.world == 1:
        pytest.skip("policy agreement requires multiple ranks")
    config = runtime.config(1, local_experts=1, top_k=1)
    policy = W4A8KernelConfig(
        tile_n=128,
        load_stages=5,
        transfer_registers=64,
        epilogue_warps=8,
        epilogue_registers=208,
        dispatch_chunk=4096,
        **{flag: 1 + runtime.rank % 2 if flag == "cluster_size" else bool(runtime.rank % 2)},
    )
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        with pytest.raises(ValueError, match="kernel IDs, specialization metadata, and workspace sizes must match"):
            case.create(config, weights, module)


@_JIT_ONLY
@pytest.mark.parametrize("cluster_size,remaining_sms", [(1, 1), (1, 3), (2, 2), (2, 3)])
@pytest.mark.parametrize("adaptive_cache_weights", [False, True], ids=["static-cache", "adaptive-cache"])
def test_native_w4a8_cluster_occupancy_and_fixed_graph_replay(
    routing_runtime, cluster_size, remaining_sms, adaptive_cache_weights
):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(65, local_experts=2, top_k=2)
    sms = torch.cuda.get_device_properties(runtime.device).multi_processor_count
    config = replace(config, sm_margin=sms - remaining_sms)
    policy = W4A8KernelConfig(
        cluster_size=cluster_size,
        load_stages=5,
        fixed_token_count=True,
        split_pipelines=True,
        dispatch_chunk=2048,
        dispatch_stages=2,
        expert_major_tasks=True,
        cache_weights=not adaptive_cache_weights,
        adaptive_cache_weights=adaptive_cache_weights,
    )
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, module)

        def check_cluster():
            assert context.cta_count == remaining_sms // cluster_size * cluster_size
            assert context.effective_kernel_config["cluster_size"] == cluster_size

        runtime.collective(check_cluster)
        storage = _buffers(runtime, context)
        for tokens in (0, 65):
            with torch.cuda.stream(case.stream):
                _stage(storage, _sample(config, tokens, "hot_first"))
                _launch(context, storage, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                _launch(context, storage, tokens, case.stream)
            for pattern in ("mixed", "hot_last", "masked", "hot_first"):
                sample = _sample(config, tokens, pattern)
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    for _ in range(3):
                        graph.replay()
                case.stream.synchronize()
                _check_output(runtime, storage, sample, expected)


@_JIT_ONLY
@pytest.mark.parametrize("adaptive_cache_weights", [False, True], ids=["static-cache", "adaptive-cache"])
@pytest.mark.parametrize(
    "tile_k,load_stages,load_warps,epilogue_tokens",
    [(128, 5, 1, 16), (256, 3, 2, 32), (512, 2, 1, 16)],
    ids=["n128-k128-loader1", "n128-k256-loader2", "n128-k512-loader1"],
)
def test_native_w4a8_cluster_variants_match(
    routing_runtime, tile_k, load_stages, load_warps, epilogue_tokens, adaptive_cache_weights
):
    from mscclpp.ext.megamoe import W4A8KernelConfig, compile_kernel

    runtime, torch = routing_runtime, routing_runtime.torch
    config = runtime.config(129, local_experts=2, top_k=2)
    modules = [
        runtime.collective(
            lambda cluster_size=cluster_size: compile_kernel(
                W4A8KernelConfig(
                    cluster_size=cluster_size,
                    tile_k=tile_k,
                    load_stages=load_stages,
                    load_warps=load_warps,
                    num_warps=12,
                    epilogue_tokens=epilogue_tokens,
                    dispatch_chunk=2048,
                    expert_major_tasks=True,
                    cache_weights=not adaptive_cache_weights,
                    adaptive_cache_weights=adaptive_cache_weights,
                )
            )
        )
        for cluster_size in (1, 2)
    ]
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        contexts = [case.create(config, weights, module) for module in modules]
        buffers = [_buffers(runtime, context) for context in contexts]
        for pattern in ("mixed", "hot_last", "masked", "hot_first"):
            sample = _sample(config, config.max_tokens, pattern)
            expected = _routing_reference(config, sample, weights)
            outputs = []
            for context, storage in zip(contexts, buffers):
                with torch.cuda.stream(case.stream):
                    _stage(storage, sample)
                    _launch(context, storage, config.max_tokens, case.stream)
                case.stream.synchronize()
                outputs.append(_check_output(runtime, storage, sample, expected))
            runtime.collective(
                lambda: torch.testing.assert_close(
                    outputs[0].view(torch.int16), outputs[1].view(torch.int16), rtol=0, atol=0
                )
            )


@_JIT_ONLY
@pytest.mark.parametrize("capacity", [32, 64, 128, 256, 512, 1024])
def test_native_w4a8_jit_profile_ep32_graph_correctness(configuration_runtime, capacity):
    from mscclpp.ext.megamoe import compile_kernel, resolve_w4a8_kernel_config

    runtime, torch = configuration_runtime, configuration_runtime.torch
    if runtime.world != 32:
        pytest.skip("packaged W4A8 JIT graph correctness requires EP32")
    config = runtime.config(
        capacity,
        local_experts=32,
        top_k=8,
        hidden=4096,
        intermediate=6656,
    )
    policy = resolve_w4a8_kernel_config(config)
    module = runtime.collective(lambda: compile_kernel(policy))
    weights = _host_weights(config)
    with _native_case(runtime) as case:
        context = case.create(config, weights, module)
        storage = _buffers(runtime, context)

        def check_policy():
            assert context.kernel_id == module.key
            assert context.kernel_config == policy
            effective = context.effective_kernel_config
            assert {name: effective[name] for name in asdict(policy)} == asdict(policy)

        runtime.collective(check_policy)
        tokens = 4
        first = _sample(config, tokens, "hot_first")
        with torch.cuda.stream(case.stream):
            _stage(storage, first)
            _launch(context, storage, tokens, case.stream)
        runtime.synchronize()
        graph = torch.cuda.CUDAGraph()
        case.graphs.append(graph)
        with torch.cuda.graph(graph, stream=case.stream):
            _launch(context, storage, tokens, case.stream)
        for pattern in ("hot_first", "mixed", "masked", "hot_last"):
            sample = _sample(config, tokens, pattern)
            expected = _routing_reference(config, sample, weights)
            with torch.cuda.stream(case.stream):
                _stage(storage, sample)
                for _ in range(3):
                    graph.replay()
            case.stream.synchronize()
            _check_output(runtime, storage, sample, expected)
