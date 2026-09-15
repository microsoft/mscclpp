# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routing regressions, with native tests gated by MSCCLPP_TEST_MEGAMOE_ROUTING=1.

Run with ``python -m pytest --noconftest python/test/test_megamoe_routing.py``.
For two or four NVLink-connected SM100 GPUs, use ``torchrun --nnodes=1
--master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 --module pytest --noconftest``
with the same file. Gloo is
only used for multi-rank rendezvous and the CPU oracle; MSCCL++ uses MASTER_PORT+1.
Set MSCCLPP_TEST_MEGAMOE_JIT=1 as well to exercise non-builtin specializations.
"""

from contextlib import contextmanager
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
    """Match the benchmark's BF16 handoffs and FP32 slot sum, entirely on CPU."""
    import torch
    from mscclpp.ext.megamoe import dequantize_mxfp8

    inputs, ids, scores = sample
    assert all(tensor.device.type == "cpu" for tensor in (*sample, *weights))
    if config.world_size > 1:
        from mscclpp.ext.megamoe.autotune import _ragged_reference

        return _ragged_reference(config, inputs, ids, scores, weights)

    partial = torch.zeros((inputs.shape[0], config.top_k, config.hidden), dtype=torch.float32)
    fc1, sf1, fc2, sf2 = weights
    for expert in range(config.local_experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        first = dequantize_mxfp8(fc1[expert], sf1[expert], dtype=torch.float32)
        second = dequantize_mxfp8(fc2[expert], sf2[expert], dtype=torch.float32)
        gate, up = (inputs[rows].float() @ first.T).chunk(2, dim=-1)
        if config.gate_up_clamp >= 0:
            gate = gate.clamp(max=config.gate_up_clamp)
            up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
        activation = (torch.nn.functional.silu(gate) * up * scores[rows, slots, None]).to(torch.bfloat16)
        partial[rows, slots] = (activation.float() @ second.T).to(torch.bfloat16).float()
    return partial.sum(dim=1).to(torch.bfloat16)


def _sample(config, tokens, pattern):
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

    def config(self, capacity, *, local_experts=8, top_k=2, e5m2=False, clamp=-1.0, intermediate=128):
        from mscclpp.ext.megamoe import MegaMoEConfig

        sms = self.torch.cuda.get_device_properties(self.device).multi_processor_count
        return MegaMoEConfig(
            rank=self.rank,
            world_size=self.world,
            max_tokens=capacity,
            hidden=128,
            intermediate=intermediate,
            num_experts=local_experts * self.world,
            top_k=top_k,
            sm_margin=max(0, sms - 8),
            weight_e5m2=e5m2,
            gate_up_clamp=clamp,
        )


@pytest.fixture(scope="module")
def routing_runtime():
    if os.environ.get("MSCCLPP_TEST_MEGAMOE_ROUTING") != "1":
        pytest.skip("set MSCCLPP_TEST_MEGAMOE_ROUTING=1 to opt into native SM100 routing tests")
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from mscclpp.ext.megamoe import is_available

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world not in (1, 2, 4):
        pytest.skip("routing tests support one, two, or four ranks")
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
            if torch.cuda.get_device_capability(local_rank) != (10, 0):
                reason = "requires SM100 on every participating rank"
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


@pytest.mark.parametrize("e5m2", [False, True])
@pytest.mark.parametrize("clamp", [-1.0, 0.125])
def test_routing_cpu_oracle_duplicate_slots_and_masked_nan(e5m2, clamp):
    torch = pytest.importorskip("torch")
    from mscclpp.ext.megamoe import MegaMoEConfig, quantize_mxfp8

    config = MegaMoEConfig(0, 1, 4, 128, 128, 2, 2, weight_e5m2=e5m2, gate_up_clamp=clamp)
    first = torch.empty((2, 256, 128))
    first[0, :128], first[0, 128:] = 1 / 128, 0.5 / 128
    first[1, :128], first[1, 128:] = 0.5 / 128, 1 / 128
    second = torch.stack((torch.full((128, 128), 1 / 128), torch.full((128, 128), 2 / 128)))
    weights = [*quantize_mxfp8(first, e5m2=e5m2), *quantize_mxfp8(second, e5m2=e5m2)]
    values = torch.tensor([-2.0, 0.5, float("nan"), 3.0])
    inputs = values[:, None].expand(-1, 128).to(torch.bfloat16).contiguous()
    ids = torch.tensor([[0, 0], [1, -1], [-1, -1], [0, 1]], dtype=torch.int32)
    scores = torch.tensor([[-1.25, 0.375], [0, float("nan")], [float("nan"), float("nan")], [1.875, -0.5]])
    output = _routing_reference(config, (inputs, ids, scores), weights)
    expected = torch.zeros((4, 2), dtype=torch.float32)
    for row, slot in ((0, 0), (0, 1), (1, 0), (3, 0), (3, 1)):
        expert = ids[row, slot].item()
        gate = values[row] * (1.0 if expert == 0 else 0.5)
        up = values[row] * (0.5 if expert == 0 else 1.0)
        if clamp >= 0:
            gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        activation = (torch.nn.functional.silu(gate) * up * scores[row, slot]).to(torch.bfloat16)
        expected[row, slot] = (activation.float() * (expert + 1)).to(torch.bfloat16).float()
    expected = expected.sum(dim=1).to(torch.bfloat16)[:, None].expand(-1, 128)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[1:3]) == 0


@pytest.mark.parametrize("top_k", [1, 2])
def test_native_routing_capacity_boundary(routing_runtime, top_k):
    runtime, torch = routing_runtime, routing_runtime.torch
    capacity = 1024 // (runtime.world * top_k)
    configs = [runtime.config(capacity + extra, top_k=top_k) for extra in (0, 1)]
    assert configs[0].world_size * configs[0].max_tokens * top_k == 1024
    assert configs[1].world_size * configs[1].max_tokens * top_k > 1024
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
        # On one GPU these are 1025 slots (K=1) and 1026 slots (K=2).
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
    assert config.world_size * config.max_tokens * config.top_k <= 1024
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


@pytest.mark.parametrize(
    "kernel_values,e5m2,clamp,direct",
    [
        pytest.param(None, False, -1.0, False, id="N32L8T7-e4-staged"),
        pytest.param(None, True, 0.125, True, id="N32L8T7-e5-clamp-direct"),
        pytest.param((32, 6, 7), False, 0.125, False, marks=_JIT_ONLY, id="N32L6T7-e4-clamp"),
        pytest.param((64, 6, 6), True, -1.0, True, marks=_JIT_ONLY, id="N64L6T6-e5-direct"),
        pytest.param((128, 4, 4), False, -1.0, False, marks=_JIT_ONLY, id="N128L4T4-e4-staged"),
    ],
)
def test_native_routing_graph_reuse(routing_runtime, kernel_values, e5m2, clamp, direct):
    from mscclpp.ext.megamoe import jit

    runtime, torch = routing_runtime, routing_runtime.torch
    # Keep the 129-token graph inside the fast path even with four ranks.
    top_k = 1 if runtime.world == 4 else 2
    config = runtime.config(129, top_k=top_k, e5m2=e5m2, clamp=clamp, intermediate=256)
    assert config.world_size * config.max_tokens * config.top_k <= 1024
    weights = _host_weights(config)
    kernel = None
    if kernel_values is not None:
        kernel = runtime.collective(lambda: jit.compile_kernel(jit.KernelConfig(*kernel_values)))
    with _native_case(runtime) as case:
        context = case.create(config, weights, kernel)
        buffers = _buffers(runtime, context, direct)

        def check_kernel():
            assert context.kernel_config == jit.KernelConfig(*(kernel_values or (32, 8, 7)))
            if kernel_values is None:
                assert context.kernel_id == "builtin"

        runtime.collective(check_kernel)
        graphs = {}
        counts = {count: max(0, count - runtime.rank) for count in (0, 1, 33, 65, 129)}
        for nominal, tokens in counts.items():
            sample = _sample(config, tokens, "hot_first")
            with torch.cuda.stream(case.stream):
                _stage(buffers, sample)
                _launch(context, buffers, tokens, case.stream)
            runtime.synchronize()
            graph = torch.cuda.CUDAGraph()
            case.graphs.append(graph)
            with torch.cuda.graph(graph, stream=case.stream):
                for _ in range(2):
                    _launch(context, buffers, tokens, case.stream)
            graphs[nominal] = graph

        # Reuse the same allocations and captured shapes, including a separate
        # empty graph. No eager forward may refresh native routing between replays.
        for nominal in (129, 1, 33, 65, 0, 129):
            tokens = counts[nominal]
            for pattern in ("hot_first", "masked", "hot_last", "balanced", "mixed", "random"):
                sample = _sample(config, tokens, pattern)
                expected = _routing_reference(config, sample, weights)
                with torch.cuda.stream(case.stream):
                    _stage(buffers, sample)
                    for _ in range(3):
                        graphs[nominal].replay()
                case.stream.synchronize()
                _check_output(runtime, buffers, sample, expected)
