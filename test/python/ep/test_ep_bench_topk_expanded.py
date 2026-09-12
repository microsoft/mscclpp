# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU-only adapter/CLI tests, adapted from a40b27e and 7747372.

NumPy tensor doubles exercise real benchmark functions, not CUDA/MPI kernels.
Expanded dispatch/expert storage aliases; compact rank-major storage does not.
"""

import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


def _bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return np.where(np.isfinite(values), rounded.view(np.float32), values)


class _Tensor(np.ndarray):
    """Small tensor double with write tracking shared across alias views."""

    def __new__(cls, values, dtype=np.float32):
        values = _bf16(values) if dtype == "bf16" else np.asarray(values, dtype=dtype)
        result = np.ndarray.view(values, cls)
        result._is_bf16 = dtype == "bf16"
        result.writes = []
        return result

    def __array_finalize__(self, source):
        self._is_bf16 = getattr(source, "_is_bf16", False)
        self.writes = getattr(source, "writes", [])

    def data_ptr(self):
        return self.__array_interface__["data"][0]

    def float(self):
        return _Tensor(self)

    def to(self, dtype):
        return _Tensor(self, dtype)

    def view(self, *shape):
        return self.reshape(*shape)

    def normal_(self):
        self.writes.append("normal")
        self.fill(1)
        return self

    def copy_(self, other):
        self.writes.append("copy")
        self[...] = _bf16(other) if self._is_bf16 else other
        return self

    def mul_(self, other):
        self.writes.append("mul")
        return self.copy_(np.asarray(self) * np.asarray(other))

    def masked_fill(self, mask, value):
        result = self.copy()
        result[mask] = value
        return result

    def sum(self, dim=None, **kwargs):
        return _Tensor(np.asarray(self).sum(axis=dim, **kwargs))

    def abs(self):
        return np.abs(self)


def _torch_stub():
    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float8_e4m3fn = "fp8"
    torch.float32 = np.float32
    torch.int = np.int32
    torch.empty = lambda shape, dtype, device: _Tensor(np.zeros(shape), dtype)
    torch.empty_like = lambda x: _Tensor(np.zeros(x.shape), "bf16")
    torch.zeros_like = lambda x, dtype: _Tensor(np.zeros(x.shape), dtype)
    # Model a single FP32 multiply-add, without BF16 intermediate rounding.
    torch.addcmul = lambda a, b, c: _Tensor(
        np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64) * np.asarray(c, dtype=np.float64)
    )
    torch.isfinite = np.isfinite
    torch.equal = np.array_equal
    torch.cuda = SimpleNamespace(synchronize=Mock(), CUDAGraph=Mock, graph=lambda graph: nullcontext())
    return torch


class _MoeStub:
    """Registered-buffer boundary double; no transport or kernel emulation."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.layout = kwargs["output_layout"]
        self.ranks = kwargs["comm"].nranks
        rows = self.ranks * kwargs["max_tokens_per_rank"]
        if self.layout == "expanded":
            rows *= kwargs["topk"]
        self.expert_output = _Tensor(np.zeros((rows, kwargs["hidden_size"])), "bf16")
        self.calls = []
        self.combined_inputs = []
        self.registered_reads = 0

    def is_available(self):
        return True

    def is_internode(self):
        return False

    def get_expert_output_buffer(self):
        self.registered_reads += 1
        return self.expert_output

    def dispatch(self, x, ids, weights, *, output_buffer):
        self.calls.append("dispatch")
        self.output_buffer = output_buffer
        e = self.config["num_experts"]
        local = (ids >= 0) & (ids < e // self.ranks)
        recv_ids = np.tile(np.where(local, ids, e), (self.ranks, 1))
        recv_weights = np.tile(np.where(local, 1.0 if weights is None else weights, 0), (self.ranks, 1))
        tokens = np.tile(x, (self.ranks, 1))
        if self.layout == "expanded":
            tokens = np.repeat(tokens, ids.shape[1], axis=0)
            recv_ids, recv_weights = recv_ids.ravel(), recv_weights.ravel()
            tokens[recv_ids == e] = np.nan
            self.expert_output[...] = tokens  # Native dispatch, not a Python adapter write.
            tokens = self.expert_output
        elif self.layout == "expert":
            tokens = _Tensor(np.zeros(output_buffer.shape), "bf16")
        else:
            tokens = _Tensor(tokens, "bf16")
        self.pristine = tokens.copy()
        self.dispatch_out = SimpleNamespace(
            tokens=tokens,
            topk_ids=_Tensor(recv_ids, np.int32),
            weights=_Tensor(recv_weights),
            # Separate Python view objects sharing the same registered pointer.
            combine_input_buffer=self.expert_output[:] if self.layout == "expanded" else None,
            quant=None,
            layout=SimpleNamespace(kind=self.layout),
        )
        return self.dispatch_out, SimpleNamespace(x=x, ids=ids, weights=weights)

    def combine(self, expert_input, handle, *, out):
        self.calls.append("combine")
        if self.layout != "expert":
            assert expert_input.data_ptr() == self.expert_output.data_ptr(), "combine requires registered storage"
        else:
            assert expert_input is self.dispatch_out.tokens
        self.combined_inputs.append(expert_input.copy())
        # Independent scalar identity oracle for adapter validation, not a
        # simulation of native payload-read safety, synchronization or transport.
        result = np.zeros(handle.x.shape, dtype=np.float32)
        for t in range(handle.x.shape[0]):
            for k in range(handle.ids.shape[1]):
                expert = handle.ids[t, k]
                weight = 1.0 if handle.weights is None else float(handle.weights[t, k])
                if 0 <= expert < self.config["num_experts"] and weight != 0:
                    result[t] = result[t].astype(np.float64) + handle.x[t].astype(np.float64) * weight
        out.copy_(result)


class TopkExpandedBenchTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        stack.enter_context(patch.object(sys, "dont_write_bytecode", True))
        self.torch = _torch_stub()
        ep = ModuleType("mscclpp.ep")
        ep.DispatchLayout = SimpleNamespace(
            EXPERT_MAJOR="expert", RANK_MAJOR="rank", RANK_MAJOR_TOPK_EXPANDED="expanded"
        )
        ep.CombineMode = SimpleNamespace(RANK_LOCAL_REDUCE="reduce", DIRECT_SEND="direct")
        ep.MoEMode = SimpleNamespace(LOW_LATENCY="ll")
        ep.DispatchDataType = SimpleNamespace(FP8_E4M3="fp8")
        ep.QuantConfig = lambda **kw: SimpleNamespace(**kw)
        self.instances = []

        def make_moe(**kwargs):
            instance = _MoeStub(**kwargs)
            self.instances.append(instance)
            return instance

        ep.MoECommunicator = Mock(side_effect=make_moe)
        self.ep = ep
        mscclpp = ModuleType("mscclpp")
        mscclpp.ep = ep
        mscclpp.CommGroup = Mock(side_effect=lambda **kw: SimpleNamespace(nranks=kw["mpi_comm"].size))
        self.make_group = mscclpp.CommGroup
        mpi4py = ModuleType("mpi4py")
        mpi4py.MPI = SimpleNamespace(MIN=object(), MAX=object(), SUM=object())
        modules = {"torch": self.torch, "mscclpp": mscclpp, "mscclpp.ep": ep, "mpi4py": mpi4py}
        for name in ("nccl", "deepep", "flashinfer"):
            module = ModuleType(f"ep_bench_{name}")
            setattr(module, f"setup_{name}", Mock(side_effect=AssertionError("unsupported backend ran")))
            module.parse_kineto_kernels = Mock()
            modules[module.__name__] = module
        stack.enter_context(patch.dict(sys.modules, modules))

        def load(name):
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        self.common = load("ep_bench_common")
        self.adapter = load("ep_bench_mscclpp")
        self.runner = load("run_ep_bench_python")
        self.comm = SimpleNamespace(size=2, Barrier=Mock(), allreduce=lambda value, op: value)

    def args(self, *extra):
        with patch.object(sys, "argv", ["bench", "--backend", "mscclpp", *extra]):
            return self.runner.parse_args()

    def setup(self, *, layout="rank_major_topk_expanded", validate=False, graph=False, unit=False):
        flags = ["--ep-layout", layout, "-t", "2", "-d", "2", "-e", "4", "-k", "3"]
        flags += ["--validate"] if validate else []
        flags += ["--cuda-graph"] if graph else []
        args = self.args(*flags)
        x = _Tensor([[128, 256], [np.nan, np.nan]], "bf16")
        ids = _Tensor([[0, 0, 4], [4, -7, 5] if unit else [1, -7, 5]], np.int64)
        weights = None if unit else _Tensor([[0.125, -0.375, 9], [0, 3, 4]])
        ops = self.adapter.setup_mscclpp(args, self.comm, 0, 2, (x, ids, weights, 3))
        return ops, self.instances[-1]

    def test_cli_accepts_expanded_and_graph_group(self):
        for k in (1, 8, 9):
            args = self.args("--ep-layout", "rank_major_topk_expanded", "-k", str(k))
            self.assertEqual(args.iters_per_graph, 1)
        args = self.args("--ep-layout", "rank_major_topk_expanded", "--cuda-graph", "--iters-per-graph", "3")
        self.assertEqual(args.iters_per_graph, 3)

    def test_cli_rejects_other_backends_before_bootstrap(self):
        for backend in ("nccl", "deepep", "flashinfer", "all"):
            with (
                self.subTest(backend=backend),
                patch.object(sys, "argv", ["bench", "--ep-layout", "rank_major_topk_expanded", "--backend", backend]),
                patch.object(self.runner, "init_mpi", side_effect=AssertionError("bootstrap ran")),
            ):
                with self.assertRaisesRegex(SystemExit, "requires --backend mscclpp"):
                    self.runner.main()
        with patch.object(sys, "argv", ["bench", "--ep-layout", "rank_major_topk_expanded"]):
            with self.assertRaisesRegex(SystemExit, "requires --backend mscclpp"):
                self.runner.parse_args()

    def test_cli_rejects_expanded_dtype_mode_and_topk(self):
        for flags, message in (
            (("--dispatch-dtype", "fp8_e4m3"), "requires --dispatch-dtype bf16"),
            (("--combine-mode", "direct_send"), "requires --combine-mode rank_local_reduce"),
            (("-k", "10"), "requires --num-topk in"),
            (("-k", "0"), "--num-topk must be in"),
        ):
            with self.subTest(flags=flags), self.assertRaisesRegex(SystemExit, message):
                self.args("--ep-layout", "rank_major_topk_expanded", *flags)

    def test_direct_adapter_rejects_invalid_expanded_config_before_bootstrap(self):
        for name, value in (
            ("backend", "all"),
            ("dispatch_dtype", "fp8_e4m3"),
            ("combine_mode", "direct_send"),
            ("num_topk", 10),
            ("num_topk", 0),
        ):
            args = self.args("--ep-layout", "rank_major_topk_expanded")
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "rank_major_topk_expanded requires"):
                self.adapter.setup_mscclpp(args, self.comm, 0, 2, None)
        self.make_group.assert_not_called()
        self.ep.MoECommunicator.assert_not_called()

    def test_expanded_eager_and_validation_use_unweighted_registered_alias(self):
        for unit in (False, True):
            ops, moe = self.setup(validate=True, unit=unit)
            ops["combine"](ops["dispatch"]())
            self.assertEqual(moe.layout, "expanded")
            self.assertEqual(moe.expert_output.shape, (12, 2))
            self.assertEqual(moe.dispatch_out.topk_ids.shape, (12,))
            self.assertEqual(moe.dispatch_out.topk_ids.dtype, np.int32)
            self.assertEqual(moe.dispatch_out.weights.dtype, np.float32)
            self.assertEqual(moe.config["invalid_token_expert_id"], 4)
            self.assertEqual(moe.config["mode"], "ll")
            self.assertEqual(moe.config["low_latency_combine_mode"], "reduce")
            self.assertFalse(moe.config.get("enable_overlap", False))
            self.assertIsNone(moe.output_buffer)
            self.assertEqual(moe.registered_reads, 1)
            self.assertEqual(len(moe.combined_inputs), 2)
            self.assertIsNot(moe.dispatch_out.tokens, moe.dispatch_out.combine_input_buffer)
            self.assertEqual(moe.dispatch_out.tokens.data_ptr(), moe.dispatch_out.combine_input_buffer.data_ptr())
            self.assertEqual(moe.expert_output.writes, [])
            for payload in moe.combined_inputs:
                np.testing.assert_equal(payload, moe.pristine)

    def test_expanded_graph_uses_registered_alias_in_one_paired_loop(self):
        ops, moe = self.setup(graph=True)
        spec = ops["graph"]
        captured = self.runner._capture_paired_graph(spec["dispatch"], spec["combine"], graph_group_size=3)
        self.assertIsNotNone(captured)
        self.assertEqual(moe.calls, ["dispatch", "combine"] * 4)
        self.assertEqual(moe.expert_output.writes, [])
        for payload in moe.combined_inputs:
            np.testing.assert_equal(payload, moe.pristine)
        dispatch, combine, graph = captured
        combine(dispatch())
        graph.replay.assert_called_once_with()
        self.assertEqual(len(moe.combined_inputs), 4)

    def test_registered_buffer_required_and_separate_input_staged(self):
        ops, moe = self.setup()
        output, handle = ops["dispatch"]()
        output.combine_input_buffer = None
        with self.assertRaisesRegex(ValueError, "combine_input_buffer"):
            ops["combine"]((output, handle))
        output.combine_input_buffer = _Tensor(np.zeros((1, 2)), "bf16")
        with self.assertRaisesRegex(ValueError, "expanded token shape"):
            ops["combine"]((output, handle))
        registered = _Tensor(np.zeros(moe.expert_output.shape), "bf16")
        output.combine_input_buffer = registered
        with patch.object(moe, "combine") as combine:
            ops["combine"]((output, handle))
        self.assertIs(combine.call_args.args[0], registered)
        np.testing.assert_equal(registered, output.tokens)
        self.assertEqual(registered.writes, ["copy"])

    def test_reference_skips_invalid_and_zero_weight_nan_payloads(self):
        x = _Tensor([[2, 4], [np.nan, np.nan], [np.nan, np.nan]], "bf16")
        ids = _Tensor([[0, 0, 4, -7, 9], [4, -7, 9, 4, -1], [0, 1, 2, 3, 0]], np.int64)
        weights = _Tensor([[0.125, 0.375, 9, 9, 9], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        result = self.adapter._topk_expanded_reference(x, ids, weights, 4)
        np.testing.assert_equal(result, [[1, 2], [0, 0], [0, 0]])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_equal(self.adapter._topk_expanded_reference(x[:2], ids[:2], None, 4), [[4, 8], [0, 0]])

    def test_reference_does_not_sanitize_live_nan_rows(self):
        for weights in (None, _Tensor([[1]])):
            result = self.adapter._topk_expanded_reference(_Tensor([[np.nan]], "bf16"), _Tensor([[0]]), weights, 4)
            self.assertTrue(np.isnan(result[0, 0]))

    def test_reference_has_no_per_rank_bf16_rounding(self):
        x = _Tensor([[128]], "bf16")
        ids = _Tensor([[0, 2]], np.int64)
        weights = _Tensor([[1.0038, 1.007]])
        result = self.adapter._topk_expanded_reference(x, ids, weights, 4)
        np.testing.assert_allclose(result, [[257.3824]], rtol=1e-7)
        np.testing.assert_equal(result.to("bf16"), [[258]])
        legacy = _bf16(np.asarray(x) * weights[0, 0]) + _bf16(np.asarray(x) * weights[0, 1])
        np.testing.assert_equal(_bf16(legacy), [[256]])

    def test_reference_preserves_topk_order_under_cancellation(self):
        result = self.adapter._topk_expanded_reference(
            _Tensor([[1]], "bf16"), _Tensor([[0, 2, 0]], np.int64), _Tensor([[2**24, 1, -(2**24)]]), 4
        )
        np.testing.assert_equal(result, [[0]])

    def test_adapter_validation_uses_source_fp32_reference(self):
        args = self.args(
            "--ep-layout", "rank_major_topk_expanded", "--validate", "-t", "1", "-d", "1", "-e", "4", "-k", "2"
        )
        x = _Tensor([[32768]], "bf16")
        ids = _Tensor([[0, 2]], np.int64)
        weights = _Tensor([[1.0038, 1.007]])
        with patch.object(
            self.adapter, "validate_combine_output_mpi", wraps=self.common.validate_combine_output_mpi
        ) as validate:
            self.adapter.setup_mscclpp(args, self.comm, 0, 2, (x, ids, weights, 2))
        validate.assert_called_once()
        np.testing.assert_equal(validate.call_args.args[1], [[66048]])
        self.assertEqual(validate.call_args.kwargs, {"exact": False})

    def test_validation_uses_existing_finiteness_and_tolerance(self):
        validate = self.common.validate_combine_output_mpi
        self.assertEqual(validate(_Tensor([8]), _Tensor([0]), self.comm, exact=False), 8)
        for actual, message in (([9], "mismatch"), ([np.nan], "NaN or Inf"), ([np.inf], "NaN or Inf")):
            with self.subTest(actual=actual), self.assertRaisesRegex(AssertionError, message):
                validate(_Tensor(actual), _Tensor([0]), self.comm, exact=False)

    def test_legacy_layouts_do_not_require_expanded_enum(self):
        del self.ep.DispatchLayout.RANK_MAJOR_TOPK_EXPANDED
        self.assertIsNone(self.args().ep_layout)
        for layout in ("rank_major", "expert_major"):
            with self.subTest(layout=layout):
                ops, moe = self.setup(layout=layout, graph=True)
                ops["combine"](ops["dispatch"]())
                ops["graph"]["dispatch"]()
                ops["graph"]["combine"]()
                self.assertIsNone(moe.dispatch_out.combine_input_buffer)
                if layout == "rank_major":
                    expected = moe.dispatch_out.tokens.copy()
                    sums = moe.dispatch_out.weights.masked_fill(moe.dispatch_out.topk_ids == 4, 0).sum(dim=1)
                    expected.mul_(sums.to("bf16").view(-1, 1))
                    for payload in moe.combined_inputs:
                        np.testing.assert_equal(payload, expected)
                else:
                    self.assertEqual(moe.registered_reads, 0)
                    self.assertEqual(moe.output_buffer.shape, (2, 4, 2))

    def test_compact_adapter_behaviors_match_base(self):
        root = Path(__file__).resolve().parents[3]
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", "4cc4276:test/python/ep/ep_bench_mscclpp.py"], text=True
        )
        old = ModuleType("base_adapter")
        exec(compile(original, "<base adapter>", "exec"), old.__dict__)
        for layout in ("rank_major", "expert_major"):
            args = self.args("--ep-layout", layout, "--cuda-graph", "-t", "1", "-d", "2", "-e", "4", "-k", "2")
            inputs = (_Tensor([[4, 8]], "bf16"), _Tensor([[0, 2]], np.int64), _Tensor([[0.5, 1]]), 2)
            outputs = []
            for adapter in (old, self.adapter):
                ops = adapter.setup_mscclpp(args, self.comm, 0, 2, inputs)
                moe = self.instances[-1]
                ops["combine"](ops["dispatch"]())
                ops["graph"]["dispatch"]()
                ops["graph"]["combine"]()
                outputs.append((moe.calls, moe.expert_output.writes, moe.combined_inputs))
            self.assertEqual(outputs[0][:2], outputs[1][:2])
            for before, after in zip(outputs[0][2], outputs[1][2]):
                np.testing.assert_equal(before, after)

    def test_timing_and_phase_helpers_unchanged(self):
        root = Path(__file__).resolve().parents[3]
        for filename, names in (
            ("ep_bench_mscclpp.py", ("parse_kineto_kernels",)),
            ("run_ep_bench_python.py", ("torch_profiler_kernel_us", "_capture_paired_graph", "run_backend", "main")),
        ):
            original = subprocess.check_output(
                ["git", "-C", str(root), "show", f"4cc4276:test/python/ep/{filename}"], text=True
            )
            current = Path(__file__).with_name(filename).read_text()
            before = {n.name: ast.dump(n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
            after = {n.name: ast.dump(n) for n in ast.parse(current).body if isinstance(n, ast.FunctionDef)}
            for name in names:
                self.assertEqual(before[name], after[name], name)


if __name__ == "__main__":
    unittest.main()
