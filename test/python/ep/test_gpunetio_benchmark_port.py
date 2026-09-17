# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU-only execution tests for the GPUNetIO Python benchmark port.

Extract real functions with AST instead of importing Torch, MPI, or native EP.
Only their external dependencies are faked; benchmark control flow is executed.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import product
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BENCH = "test/python/ep/ep_bench_mscclpp.py"
HARNESS = "test/python/ep/run_ep_bench_python.py"


@lru_cache(None)
def _source(path):
    return ast.parse((ROOT / path).read_text(), filename=path)


def _load(path, names, namespace, owner=None):
    body = _source(path).body
    if owner is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    nodes = [node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + nodes, type_ignores=[])
    namespace.setdefault("__name__", __name__)
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace


def _api_types():
    namespace = {
        "dataclass": dataclass,
        "CombineMode": Enum("CombineMode", "RANK_LOCAL_REDUCE DIRECT_SEND"),
        "DispatchLayout": Enum("DispatchLayout", "EXPERT_MAJOR RANK_MAJOR TOKEN_MAJOR RANK_MAJOR_TOPK_EXPANDED"),
        "DispatchDataType": Enum("DispatchDataType", "BF16 FP8_E4M3"),
        "MoEMode": Enum("MoEMode", "LATENCY THROUGHPUT"),
    }
    return _load(
        "python/mscclpp/ep/types.py",
        (
            "MoECommunicatorConfig",
            "QuantConfig",
            "DispatchOutput",
            "DispatchOutputInfo",
            "DispatchLayoutInfo",
            "DispatchHandle",
            "_ExpertMajorCombineContext",
            "_RankMajorCombineContext",
        ),
        namespace,
    )


class FakeTensor:
    def __init__(self, shape, dtype="bf16", device="cuda", trace=None):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.trace = trace

    def data_ptr(self):
        return id(self)

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def normal_(self):
        self.trace.append("normal")
        return self


def _args(**kwargs):
    values = dict(
        mode="latency",
        backend="mscclpp",
        ep_layout="rank_major",
        num_sms=0,
        num_tokens=2,
        hidden=16,
        num_experts=4,
        num_topk=2,
        num_warmup=2,
        num_iters=2,
        combine_mode="rank_local_reduce",
        dispatch_dtype="bf16",
        cuda_graph=True,
        iters_per_graph=50,
        validate=False,
        seed=0,
    )
    return NS(**(values | kwargs))


class CpuPortTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.trace = []

    def log(self, text, *, flush=False):
        self.assertTrue(flush, text)
        self.trace.append(text)

    def torch_tensors(self):
        def empty(shape, *, dtype, device):
            self.trace.append("allocate")
            return FakeTensor(shape, dtype, device, self.trace)

        return NS(bfloat16="bf16", float8_e4m3fn="fp8", empty=empty)

    def setup_benchmark(self, args):
        api = _api_types()
        trace = self.trace
        tensors = self.torch_tensors()
        instances = []

        class FakeMoE:
            def __init__(self, **kwargs):
                # Use the target's real config: obsolete keyword fields fail.
                self.config = api["MoECommunicatorConfig"](**kwargs)
                instances.append(self)
                self.tokens = FakeTensor((4, 16), trace=trace)
                self.combined = []

            def is_available(self):
                return True

            def is_internode(self):
                return True

            def dispatch(self, *args, **kwargs):
                trace.append("dispatch")
                rank_major = self.config.output_layout in (
                    api["DispatchLayout"].RANK_MAJOR,
                    api["DispatchLayout"].RANK_MAJOR_TOPK_EXPANDED,
                )
                return (
                    api["DispatchOutput"](
                        tokens=self.tokens,
                        quant=None,
                        layout=api["DispatchLayoutInfo"](self.config.output_layout),
                        combine_input_buffer=self.tokens if rank_major else None,
                    ),
                    object(),
                )

            def combine(self, tensor, handle, **kwargs):
                trace.append("combine")
                self.combined.append((tensor, handle, kwargs))

        package = ModuleType("mscclpp")
        ep = ModuleType("mscclpp.ep")
        ep.__dict__.update(api)
        ep.MoECommunicator = FakeMoE
        package.ep = ep
        namespace = dict(os=os, torch=tensors, print=self.log, _make_comm_group=lambda comm: comm)
        _load("test/python/ep/ep_bench_common.py", ("simulated_gemm_output",), namespace)
        _load(BENCH, ("setup_mscclpp", "_setup_mscclpp_latency", "_setup_mscclpp_throughput"), namespace)
        with patch.dict(sys.modules, {"mscclpp": package, "mscclpp.ep": ep}):
            ops = namespace["setup_mscclpp"](args, object(), 0, 2, (object(), object(), object(), 4))
        return ops, instances[0], namespace

    def test_latency_block_precedence_and_resolved_config(self):
        for cli, env, expected in ((0, None, 130), (0, "76", 76), (64, "76", 64), (64, "bad", 64)):
            with self.subTest(cli=cli, env=env), patch.dict(os.environ, {}, clear=True):
                if env is not None:
                    os.environ["MSCCLPP_EP_LL_BLOCKS"] = env
                self.trace.clear()
                _, moe, _ = self.setup_benchmark(_args(num_sms=cli))
                self.assertEqual(moe.config.num_blocks, expected)
                self.assertTrue(any(f"num_blocks={expected} " in line for line in self.trace))
                self.assertTrue(any(f"ll_blocks={expected} " in line for line in self.trace))
        os.environ["MSCCLPP_EP_LL_BLOCKS"] = "bad"
        with self.assertRaises(ValueError):
            self.setup_benchmark(_args())

    def test_throughput_ignores_latency_blocks(self):
        os.environ["MSCCLPP_EP_LL_BLOCKS"] = "bad"
        for cli, expected in ((0, None), (32, 32)):
            with self.subTest(cli=cli):
                _, moe, _ = self.setup_benchmark(_args(mode="throughput", num_sms=cli))
                self.assertEqual(moe.config.num_blocks, expected)
                self.assertEqual(moe.config.mode.name, "THROUGHPUT")

    def test_eager_and_graph_keep_bf16_alias_without_weighting_or_copies(self):
        for layout, debug in product(("rank_major", "expert_major"), (None, "0", "1", "true")):
            with self.subTest(layout=layout, debug=debug), patch.dict(os.environ, {}, clear=True):
                if debug is not None:
                    os.environ["MSCCLPP_EP_DEBUG_COMBINE"] = debug
                ops, moe, _ = self.setup_benchmark(_args(ep_layout=layout))
                self.trace.clear()
                rank_major = layout == "rank_major"
                enabled = debug == "1"
                expected = []
                for iteration in range(2):
                    dout = ops["dispatch"]()
                    ops["combine"](dout)
                    self.assertIs(moe.combined[-1][0], dout[0].tokens)
                    self.assertIs(moe.combined[-1][1], dout[1])
                    expected.append("dispatch")
                    if enabled:
                        expected.append("[combfn][rank 0] enter")
                        if rank_major:
                            expected.append("[rank_major_input][rank 0] enter")
                    if rank_major and iteration == 0:
                        expected.append("normal")
                        if enabled:
                            expected.append("[rank_major_input][rank 0] initialized")
                    if enabled and rank_major:
                        expected.append("[rank_major_input][rank 0] buffer ready")
                    expected.append("combine")
                    if enabled:
                        expected.append("[combfn][rank 0] exit")
                self.assertEqual(self.trace, expected)

                self.trace.clear()
                for _ in range(2):
                    ops["graph"]["dispatch"]()
                    ops["graph"]["combine"]()
                    self.assertIs(moe.combined[-1][0], moe.tokens)
                expected = ["dispatch"]
                if enabled:
                    expected.append("[graph_combfn][rank 0] enter")
                    if rank_major:
                        expected += ["[rank_major_input][rank 0] enter", "[rank_major_input][rank 0] simulated"]
                expected.append("combine")
                if enabled:
                    expected.append("[graph_combfn][rank 0] exit")
                self.assertEqual(self.trace, expected * 2)

    def run_fake_backend(self, name, args, group=1, kineto=False):
        trace = self.trace
        trace.clear()
        stream = NS(synchronize=lambda: trace.append("stream sync"))
        event_count = 0

        def event(*, enable_timing):
            nonlocal event_count
            self.assertTrue(enable_timing)
            index = event_count
            event_count += 1

            def record(on_stream):
                self.assertIs(on_stream, stream)
                trace.append(f"record {index}")

            return NS(record=record, elapsed_time=lambda other: 1.0)

        def stats(comm, avg, lo, hi, ranks):
            trace.append(("stats", avg, lo, hi))
            return avg, lo, hi

        def profile(*args, **kwargs):
            trace.append("profile")
            self.assertEqual(kwargs["barrier"], "bench barrier")
            self.assertEqual(kwargs["mid_barrier"], "nccl barrier")
            return 3.0, 4.0

        def dispatch():
            trace.append("dispatch")
            return "dispatch result"

        def combine(dout):
            self.assertEqual(dout, "dispatch result")
            trace.append("combine")

        torch = NS(cuda=NS(current_stream=lambda: stream, Event=event, synchronize=lambda: trace.append("final sync")))
        comm = NS(
            Barrier=lambda: trace.append("barrier"),
            gather=lambda value, root: [value],
            allreduce=lambda value, op: value,
        )
        namespace = dict(
            os=os, torch=torch, print=self.log, _mpi_stats=stats, MPI=NS(MIN="min"), torch_profiler_kernel_us=profile
        )
        _load(HARNESS, ("run_backend",), namespace)
        with patch.dict(os.environ, {"EP_KERNEL_TIMER": "kineto" if kineto else "events"}):
            namespace["run_backend"](
                name,
                args,
                comm,
                1,
                2,
                (None, None, None, 4),
                dispatch,
                combine,
                nccl_barrier="nccl barrier",
                bench_barrier="bench barrier",
                graph_group_size=group,
            )
        first_stats = next(i for i, item in enumerate(trace) if isinstance(item, tuple) and item[0] == "stats")
        self.assertEqual(trace[first_stats][1:], (1000.0 / group,) * 3)
        return trace[:first_stats]

    def expected_pair_trace(self, name, args, sync, debug):
        trace = []

        def log(message):
            if debug:
                trace.append(f"[pairdbg][{name}][rank 1] {message}")

        for i in range(args.num_warmup):
            log(f"warmup {i} pair start")
            trace.append("dispatch")
            log(f"warmup {i} dispatch returned")
            trace.append("combine")
            log(f"warmup {i} combine returned")
            trace.append("stream sync")
            log(f"warmup {i} stream synced")
            trace.append("barrier")
            log(f"warmup {i} barrier done")
        for i in range(args.num_iters):
            log(f"iter {i} pair start")
            trace += [f"record {i}", "dispatch"]
            log(f"iter {i} dispatch returned")
            trace += [f"record {args.num_iters + i}", f"record {2 * args.num_iters + i}", "combine"]
            log(f"iter {i} combine returned")
            trace.append(f"record {3 * args.num_iters + i}")
            if sync:
                trace += ["barrier", "stream sync"]
                log(f"iter {i} stream synced")
                trace.append("barrier")
                log(f"iter {i} barrier done")
        log("final stream sync start")
        trace.append("final sync")
        log("final stream sync done")
        return trace

    def test_default_sync_and_debug_are_latency_gpunetio_rank_major_only(self):
        cases = product(
            ("mscclpp", "nccl", "deepep", "flashinfer"),
            ("latency", "throughput"),
            (None, "rank_major", "expert_major", "token_major"),
            (None, "0", "1", "true"),
        )
        for name, mode, layout, gpunetio in cases:
            with (
                self.subTest(name=name, mode=mode, layout=layout, gpunetio=gpunetio),
                patch.dict(os.environ, {}, clear=True),
            ):
                if gpunetio is not None:
                    os.environ["MSCCLPP_EP_ENABLE_GPUNETIO"] = gpunetio
                args = _args(mode=mode, ep_layout=layout)
                enabled = (name, mode, layout, gpunetio) == ("mscclpp", "latency", "rank_major", "1")
                actual = self.run_fake_backend(name, args)
                self.assertEqual(actual, self.expected_pair_trace(name, args, enabled, enabled))

    def test_explicit_sync_and_debug_overrides_and_ordering(self):
        cases = product(
            (("mscclpp", "latency", True), ("mscclpp", "throughput", False), ("nccl", "latency", False)),
            (None, "0", "1", "", "true"),
            (None, "0", "1", "", "true"),
        )
        for (name, mode, default), sync_env, debug_env in cases:
            with (
                self.subTest(name=name, mode=mode, sync=sync_env, debug=debug_env),
                patch.dict(os.environ, {}, clear=True),
            ):
                os.environ["MSCCLPP_EP_ENABLE_GPUNETIO"] = "1"
                for key, value in (("EP_SYNC_EACH_ITER", sync_env), ("EP_DEBUG_PAIR", debug_env)):
                    if value is not None:
                        os.environ[key] = value
                sync = default if sync_env is None else sync_env == "1"
                debug = sync if debug_env is None else debug_env == "1"
                args = _args(mode=mode)
                actual = self.run_fake_backend(name, args, group=50)
                self.assertEqual(actual, self.expected_pair_trace(name, args, sync, debug))

    def test_zero_warmup_single_iteration_and_profiler_order(self):
        os.environ.update(EP_SYNC_EACH_ITER="1", EP_DEBUG_PAIR="1")
        args = _args(num_warmup=0, num_iters=1)
        actual = self.run_fake_backend("mscclpp", args, kineto=True)
        self.assertEqual(actual, self.expected_pair_trace("mscclpp", args, True, True) + ["barrier", "profile"])

    def test_graph_group_size_reaches_capture_and_timing_unchanged(self):
        os.environ.update(MSCCLPP_EP_ENABLE_GPUNETIO="1", EP_KERNEL_TIMER="events")
        ops, _, namespace = self.setup_benchmark(_args())
        trace = self.trace

        class FakeGraph:
            def replay(self):
                trace.append("replay")

        namespace["torch"].cuda = NS(
            CUDAGraph=FakeGraph, graph=lambda graph: nullcontext(), synchronize=lambda: trace.append("sync")
        )
        _load(HARNESS, ("parse_args", "_capture_paired_graph", "main"), namespace)
        namespace["argparse"] = argparse
        with patch.object(sys, "argv", ["bench", "--backend", "mscclpp", "--cuda-graph", "--iters-per-graph", "50"]):
            parsed = namespace["parse_args"]()
        self.assertEqual(parsed.iters_per_graph, 50)
        captured = []

        def run_backend(*args, graph_group_size, **kwargs):
            captured.append(graph_group_size)
            args[6]()
            args[7](None)

        comm = NS(Barrier=lambda: None, allreduce=lambda value, op: value)
        namespace.update(
            parse_args=lambda: _args(),
            init_mpi=lambda: (comm, 0, 2, 0),
            make_inputs=lambda *args: (None, None, None, 4),
            _SETUP={"mscclpp": lambda *args: ops | {"teardown": lambda: None}},
            _PARSE_KINETO={"mscclpp": None},
            MPI=NS(MIN="min"),
            run_backend=run_backend,
        )
        trace.clear()
        namespace["main"]()
        self.assertEqual(captured, [50])
        self.assertEqual(trace.count("dispatch"), 51)  # one prime plus all 50 capture pairs
        self.assertEqual(trace.count("combine"), 51)
        self.assertEqual(trace.count("replay"), 1)
        self.assertNotIn("normal", trace)

    def runtime_combine(self, *, rank_major, allocate, debug, direct=False, failure=None):
        namespace = _api_types()
        namespace.update(
            os=os,
            torch=self.torch_tensors(),
            print=self.log,
            # Isolate the combine body from deferred collective initialization.
            requires_initialized=lambda method: method,
            cuda_stream_ptr=lambda stream: stream.cuda_stream,
        )
        _load("python/mscclpp/ep/latency.py", ("combine", "_validate_combine"), namespace, owner="LatencyRuntime")
        runtime_type = type("LatencyRuntime", (), {name: namespace[name] for name in ("combine", "_validate_combine")})
        runtime = runtime_type()
        layout = namespace["DispatchLayout"].RANK_MAJOR if rank_major else namespace["DispatchLayout"].EXPERT_MAJOR
        mode = namespace["CombineMode"].DIRECT_SEND if direct else namespace["CombineMode"].RANK_LOCAL_REDUCE
        shape = (6, 2, 16) if direct and rank_major else ((6, 16) if rank_major else (2, 8, 16))
        expert = FakeTensor(shape)
        context = dict(topk_ids=FakeTensor((2, 2)), num_experts=4, num_tokens=2, hidden_size=16)
        if rank_major:
            context = namespace["_RankMajorCombineContext"](**context, max_tokens_per_rank=3)
        else:
            context = namespace["_ExpertMajorCombineContext"](
                **context, weights=FakeTensor((2, 2)), src_info=FakeTensor((2, 8)), layout_range=FakeTensor((2, 2))
            )
        handle = namespace["DispatchHandle"](
            namespace["DispatchOutputInfo"](namespace["DispatchLayoutInfo"](layout)), context
        )
        runtime.context = NS(
            rank=1,
            world_size=2,
            num_local_experts=2,
            num_experts=4,
            hidden_size=16,
            topk=2,
            max_tokens_per_rank=4,
            num_blocks=12,
            output_layout=layout,
            dispatch_data_type=namespace["DispatchDataType"].BF16,
            combine_mode=mode,
            combine_input_buffer=expert,
        )
        native_args = []

        def native(*args):
            self.trace.append("native")
            native_args.append(args)
            if failure == "native":
                raise RuntimeError("native failure")

        runtime.cpp_runtime = NS(combine=native)
        comm_namespace = dict(os=os, print=self.log)
        _load("python/mscclpp/ep/communicator.py", ("combine",), comm_namespace, owner="MoECommunicator")
        communicator = NS(_runtime=runtime)
        out = None if allocate else FakeTensor((2, 16))
        if failure == "validation":
            expert.dtype = "invalid"
        env = {} if debug is None else {"MSCCLPP_EP_DEBUG_COMBINE": debug}
        self.trace.clear()
        with patch.dict(os.environ, env, clear=True):
            invoke = lambda: comm_namespace["combine"](communicator, expert, handle, out=out, stream=NS(cuda_stream=42))
            if failure:
                with self.assertRaises(ValueError if failure == "validation" else RuntimeError):
                    invoke()
                return
            result = invoke()
        self.assertEqual(result.shape, (2, 16))
        if not allocate:
            self.assertIs(result, out)
        self.assertEqual(
            native_args,
            [
                (
                    expert.data_ptr(),
                    context.topk_ids.data_ptr(),
                    0 if rank_major else context.weights.data_ptr(),
                    0 if rank_major else context.src_info.data_ptr(),
                    0 if rank_major else context.layout_range.data_ptr(),
                    result.data_ptr(),
                    2,
                    16,
                    2,
                    3 if rank_major else 4,
                    4,
                    layout,
                    namespace["DispatchDataType"].BF16,
                    mode,
                    10,
                    42,
                )
            ],
        )

    def test_runtime_context_logs_and_native_arguments_without_sync(self):
        for rank_major, allocate, debug, direct in product(
            (False, True), (False, True), (None, "0", "1", "true"), (False, True)
        ):
            with self.subTest(rank_major=rank_major, allocate=allocate, debug=debug, direct=direct):
                self.runtime_combine(rank_major=rank_major, allocate=allocate, debug=debug, direct=direct)
                expected = []
                if debug == "1":
                    expected += [
                        "[py_comm_combine] enter runtime=LatencyRuntime",
                        "[py_ll_combine][rank 1] enter handle=DispatchHandle",
                        "[py_ll_combine][rank 1] validated",
                        (
                            "[py_ll_combine][rank 1] rank-major context capacity=3"
                            if rank_major
                            else "[py_ll_combine][rank 1] expert-major context"
                        ),
                    ]
                if allocate:
                    if debug == "1":
                        expected.append("[py_ll_combine][rank 1] allocate output")
                    expected.append("allocate")
                if debug == "1":
                    expected.append("[py_ll_combine][rank 1] before combine tokens=2 hidden=16 topk=2")
                expected.append("native")
                if debug == "1":
                    expected += ["[py_ll_combine][rank 1] after combine", "[py_comm_combine] exit"]
                self.assertEqual(self.trace, expected)

    def test_failure_logs_stop_at_validation_or_native_call(self):
        self.runtime_combine(rank_major=True, allocate=False, debug="1", failure="validation")
        self.assertEqual(
            self.trace,
            [
                "[py_comm_combine] enter runtime=LatencyRuntime",
                "[py_ll_combine][rank 1] enter handle=DispatchHandle",
            ],
        )
        self.runtime_combine(rank_major=True, allocate=False, debug="1", failure="native")
        self.assertEqual(
            self.trace[-2:], ["[py_ll_combine][rank 1] before combine tokens=2 hidden=16 topk=2", "native"]
        )
        self.assertNotIn("[py_comm_combine] exit", self.trace)


if __name__ == "__main__":
    unittest.main()
