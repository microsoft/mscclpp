# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU-only tests of the real Python API; tensor metadata and native calls only.

No CUDA allocations, transport, arithmetic, or GPU numerical claims are modeled.
Run directly with Python/unittest; NumPy is needed by the real API utilities.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
import importlib
from itertools import count, product
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

_POINTERS = count(4096, 4096)


@dataclass
class _Tensor:
    shape: tuple
    dtype: str = "bfloat16"
    device: object = field(default_factory=lambda: SimpleNamespace(type="cuda", index=0))
    pointer: int = field(default_factory=lambda: next(_POINTERS))
    contiguous: bool = True

    def dim(self):
        return len(self.shape)

    def size(self, axis):
        return self.shape[axis]

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def view(self, dtype):
        return replace(self, dtype=dtype)


class TopkExpandedPythonApiTests(unittest.TestCase):
    def setUp(self):
        importlib.import_module("numpy")  # Keep the real utils dependency loaded across isolated imports.
        self.enterContext(patch.dict(sys.modules))
        self.enterContext(patch.object(sys, "dont_write_bytecode", True))
        for name in tuple(sys.modules):
            if name == "mscclpp" or name.startswith("mscclpp."):
                del sys.modules[name]
        root = Path(__file__).resolve().parents[3] / "python" / "mscclpp"
        for name, path in (("mscclpp", root), ("mscclpp.ep", root / "ep")):
            module = ModuleType(name)
            module.__path__ = [str(path)]  # bypass native-loading package initializers only
            sys.modules[name] = module
        torch = ModuleType("torch")
        for dtype in ("bfloat16", "int32", "int64", "float32", "float8_e4m3fn"):
            setattr(torch, dtype, dtype)
        torch.Tensor, torch.empty = _Tensor, _Tensor
        torch.device = lambda kind, index: SimpleNamespace(type=kind, index=index)
        torch.cuda = SimpleNamespace(current_device=lambda: 0, current_stream=lambda: SimpleNamespace(cuda_stream=17))

        def as_tensor(view, *, device):
            interface = view.__cuda_array_interface__
            dtype = {"<u2": "uint16", "<i4": "int32", "<f4": "float32"}[interface["typestr"]]
            return _Tensor(interface["shape"], dtype, device, interface["data"][0])

        torch.as_tensor = as_tensor
        cpp = ModuleType("mscclpp.ep._cpp")
        for name, members in (
            ("DispatchLayout", "EXPERT_MAJOR TOKEN_MAJOR RANK_MAJOR RANK_MAJOR_TOPK_EXPANDED"),
            ("CombineMode", "RANK_LOCAL_REDUCE DIRECT_SEND"),
            ("DispatchDataType", "BF16 FP8_E4M3"),
            ("MoEMode", "LOW_LATENCY HIGH_THROUGHPUT"),
        ):
            setattr(cpp, name, Enum(name, members))
        getters = [
            f"{name}_buffer_ptr"
            for name in ("output_tokens", "expert_output", "output_topk_ids", "output_topk_weights")
        ]

        def runtime(*args, **kwargs):
            result = Mock(spec=getters + ["is_available", "is_internode_available", "ll_dispatch", "ll_combine"])
            result.is_available.return_value, result.is_internode_available.return_value = True, False
            for name in getters:
                getattr(result, name).return_value = next(_POINTERS)
            return result

        cpp._cpp, cpp.Config = cpp, Mock(side_effect=AssertionError("unexpected HT initialization"))
        self.factory = cpp.create_moe_runtime = Mock(side_effect=runtime)
        sys.modules.update({"torch": torch, "mscclpp.ep._cpp": cpp})
        self.api = importlib.import_module("mscclpp.ep.communicator")
        self.ll = importlib.import_module("mscclpp.ep.low_latency")
        self.layout = self.api.DispatchLayout
        self.comm = SimpleNamespace(my_rank=0, nranks=2, communicator=object())
        self.x, self.ids, self.weights = _Tensor((2, 128)), _Tensor((2, 3), "int64"), _Tensor((2, 3), "float32")

    def make(self, **changes):
        config = dict(
            comm=self.comm,
            num_experts=8,
            hidden_size=128,
            topk=3,
            max_tokens_per_rank=4,
            output_layout=self.layout.RANK_MAJOR_TOPK_EXPANDED,
            low_latency_num_blocks=6,
        )
        return self.api.MoECommunicator(self.api.MoECommunicatorConfig(**(config | changes)))

    def reject(self, call, native, cases, error=ValueError):
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(error):
                    call(**changes)
                native.assert_not_called()

    def test_expanded_buffers_and_original_weights_forwarded_once(self):
        for k, weighted, blocks, explicit in product((1, 3, 9), (False, True), (4, 130), (False, True)):
            with self.subTest(k=k, weighted=weighted, blocks=blocks, explicit=explicit):
                moe = self.make(topk=k, low_latency_num_blocks=blocks)
                backend, rows = moe._backend, 2 * 4 * k
                native = backend._runtime.cpp_runtime
                self.assertIsInstance(backend, self.ll.LowLatencyBackend)
                self.factory.assert_called_with(
                    self.comm.communicator,
                    self.api.MoEMode.LOW_LATENCY,
                    max_tokens_per_rank=4,
                    hidden=128,
                    num_experts=8,
                    num_topk=k,
                    output_layout=moe.output_layout,
                )
                ids, weights = _Tensor((2, k), "int64"), _Tensor((2, k), "float32") if weighted else None
                stream = SimpleNamespace(cuda_stream=29) if explicit else None
                output, handle = moe.dispatch(
                    self.x, ids, weights, stream=stream, output_buffer=backend._output_tokens if explicit else None
                )
                expert = moe.get_expert_output_buffer()
                for tensor, shape, dtype, getter in (
                    (output.tokens, (rows, 128), "bfloat16", native.output_tokens_buffer_ptr),
                    (expert, (rows, 128), "bfloat16", native.expert_output_buffer_ptr),
                    (output.topk_ids, (rows,), "int32", native.output_topk_ids_buffer_ptr),
                    (output.weights, (rows,), "float32", native.output_topk_weights_buffer_ptr),
                ):
                    self.assertEqual(
                        (tensor.shape, tensor.dtype, tensor.data_ptr()), (shape, dtype, getter.return_value)
                    )
                    self.assertIs(tensor._mscclpp_owner, backend._runtime)
                self.assertIs(expert, moe.get_expert_output_buffer())
                self.assertIsInstance(handle, self.api.RankMajorTopkExpandedDispatchHandle)
                self.assertIsInstance(handle.combine_context, self.api.RankMajorTopkExpandedCombineContext)
                self.assertIs(handle.combine_context.topk_ids, ids)
                self.assertIs(handle.combine_context.weights, weights)
                self.assertIsNot(output.weights, weights)
                self.assertIs(output.layout, handle.output_info.layout)
                self.assertIsNone(output.quant)
                counts = output.layout.num_tokens_per_rank
                self.assertEqual((counts.shape, counts.dtype), ((2,), "int32"))
                self.assertIsNone(output.layout.num_tokens_per_expert)
                wptr, sptr = 0 if weights is None else weights.data_ptr(), 29 if explicit else 17
                native.ll_dispatch.assert_called_once_with(
                    self.x.data_ptr(),
                    ids.data_ptr(),
                    wptr,
                    output.tokens.data_ptr(),
                    0,
                    0,
                    output.topk_ids.data_ptr(),
                    output.weights.data_ptr(),
                    0,
                    counts.data_ptr(),
                    2,
                    128,
                    k,
                    4,
                    8,
                    8,
                    moe.output_layout,
                    self.api.DispatchDataType.BF16,
                    blocks,
                    sptr,
                )
                supplied = _Tensor((2, 128)) if explicit else None
                combined = moe.combine(expert, handle, out=supplied, stream=stream)
                self.assertEqual((combined.shape, combined.dtype), ((2, 128), "bfloat16"))
                if explicit:
                    self.assertIs(combined, supplied)
                native.ll_combine.assert_called_once_with(
                    expert.data_ptr(),
                    ids.data_ptr(),
                    wptr,
                    0,
                    0,
                    combined.data_ptr(),
                    2,
                    128,
                    k,
                    4,
                    8,
                    moe.output_layout,
                    self.api.DispatchDataType.BF16,
                    self.api.CombineMode.RANK_LOCAL_REDUCE,
                    blocks - 2,
                    sptr,
                )
                self.assertEqual(moe.num_sms, blocks - 2)

    def test_invalid_configuration_rejected_before_runtime_creation(self):
        self.reject(
            self.make,
            self.factory,
            [dict(topk=k) for k in (0, 10, True, 1.5)]
            + [dict(low_latency_num_blocks=n) for n in (3, 131)]
            + [dict(low_latency_combine_mode=self.api.CombineMode.DIRECT_SEND)],
        )
        self.reject(
            self.make,
            self.factory,
            [
                dict(mode="LOW_LATENCY"),
                dict(output_layout="expanded"),
                dict(low_latency_combine_mode="RANK_LOCAL_REDUCE"),
            ],
            TypeError,
        )
        self.reject(
            self.make,
            self.factory,
            [
                dict(enable_overlap=True),
                dict(output_layout=self.layout.TOKEN_MAJOR),
                dict(mode=self.api.MoEMode.HIGH_THROUGHPUT),
                dict(quant=self.api.QuantConfig(format=self.api.DispatchDataType.FP8_E4M3)),
            ],
            NotImplementedError,
        )

    def test_runtime_capacity_guard(self):
        moe = self.make()
        native = moe._backend._runtime.cpp_runtime
        self.reject(
            lambda **kw: moe.dispatch(self.x, self.ids, **kw),
            native.ll_dispatch,
            [dict(runtime_max_tokens_per_rank=n) for n in (-1, 0, 1, 3, 5, True, 4.0, "4")],
        )
        with self.assertRaises(ValueError):
            moe.dispatch(_Tensor((5, 128)), _Tensor((5, 3), "int64"))
        native.ll_dispatch.assert_not_called()
        for capacity in (None, 4):
            _, handle = moe.dispatch(self.x, self.ids, runtime_max_tokens_per_rank=capacity)
            self.assertEqual(handle.combine_context.max_tokens_per_rank, 4)
            self.assertEqual(native.ll_dispatch.call_args.args[13], 4)

    def test_dispatch_tensor_and_registered_pointer_guards(self):
        moe = self.make()
        native = moe._backend._runtime.cpp_runtime
        inputs = dict(input=self.x, topk_ids=self.ids, weights=self.weights, output_buffer=moe._backend._output_tokens)
        for name, tensor in inputs.items():
            self.reject(
                lambda **kw: moe.dispatch(**(inputs | kw)),
                native.ll_dispatch,
                [
                    {name: replace(tensor, **change)}
                    for change in (
                        dict(shape=(1,)),
                        dict(shape=(2, 7)),
                        dict(dtype="wrong"),
                        dict(contiguous=False),
                        dict(device=SimpleNamespace(type="cpu", index=0)),
                    )
                ],
            )
        self.reject(
            lambda **kw: moe.dispatch(**(inputs | kw)),
            native.ll_dispatch,
            [
                dict(output_buffer=_Tensor((24, 128))),
                {
                    name: replace(tensor, device=SimpleNamespace(type="cuda", index=1))
                    for name, tensor in inputs.items()
                    if name != "output_buffer"
                },
            ],
        )
        self.reject(
            lambda **kw: moe.dispatch(**(inputs | kw)),
            native.ll_dispatch,
            [dict(quant=self.api.QuantConfig())],
            NotImplementedError,
        )

    def test_combine_tensor_handle_and_context_guards(self):
        moe = self.make()
        _, handle = moe.dispatch(self.x, self.ids, self.weights)
        native, expert = moe._backend._runtime.cpp_runtime, moe.get_expert_output_buffer()
        inputs = dict(expert_output=expert, handle=handle, out=_Tensor((2, 128)))
        for name in ("expert_output", "out"):
            self.reject(
                lambda **kw: moe.combine(**(inputs | kw)),
                native.ll_combine,
                [
                    {name: replace(inputs[name], **change)}
                    for change in (
                        dict(shape=(1,)),
                        dict(dtype="float32"),
                        dict(contiguous=False),
                        dict(device=SimpleNamespace(type="cuda", index=1)),
                    )
                ],
            )
        contexts = [dict(max_tokens_per_rank=n) for n in (None, -1, 0, 3, 5, True, 4.0)]
        contexts += [dict(num_tokens=n) for n in (-1, 5, True, 2.0)] + [dict(num_experts=16), dict(hidden_size=256)]
        for name, tensor in (("topk_ids", self.ids), ("weights", self.weights)):
            contexts += [
                {name: replace(tensor, **change)}
                for change in (
                    dict(shape=(2,)),
                    dict(dtype="int32"),
                    dict(contiguous=False),
                    dict(device=SimpleNamespace(type="cuda", index=1)),
                )
            ]
        self.reject(
            lambda **kw: moe.combine(**(inputs | kw)),
            native.ll_combine,
            [
                dict(handle=replace(handle, combine_context=replace(handle.combine_context, **change)))
                for change in contexts
            ]
            + [
                dict(handle=replace(handle, combine_context=object())),
                dict(handle=self.api.DispatchHandle(handle.output_info)),
                dict(expert_output=_Tensor((24, 128))),
                dict(
                    handle=replace(
                        handle,
                        output_info=replace(
                            handle.output_info, layout=self.api.DispatchLayoutInfo(self.layout.RANK_MAJOR)
                        ),
                    )
                ),
                dict(
                    handle=replace(
                        handle,
                        output_info=replace(
                            handle.output_info, quant=self.api.QuantConfig(format=self.api.DispatchDataType.FP8_E4M3)
                        ),
                    )
                ),
            ],
        )

    def test_legacy_parity_and_cross_layout_handles(self):
        expanded = self.make()
        _, expanded_handle = expanded.dispatch(self.x, self.ids)
        for layout in (None, self.layout.RANK_MAJOR):
            with self.subTest(layout=layout):
                moe = self.make(output_layout=layout)
                rank_major = layout == self.layout.RANK_MAJOR
                buffer = None if rank_major else _Tensor((4, 8, 128))
                output, handle = moe.dispatch(self.x, self.ids, self.weights, output_buffer=buffer)
                native = moe._backend._runtime.cpp_runtime
                self.assertEqual(output.tokens.shape, (8, 128) if rank_major else (4, 8, 128))
                if rank_major:
                    self.assertIsInstance(handle, self.api.RankMajorDispatchHandle)
                    self.assertFalse(hasattr(handle.combine_context, "weights"))
                    self.assertEqual((output.topk_ids.shape, output.weights.shape), ((8, 3), (8, 3)))
                    expert = moe.get_expert_output_buffer()
                else:
                    self.assertIsInstance(handle, self.api.ExpertMajorDispatchHandle)
                    self.assertIs(handle.combine_context.weights, self.weights)
                    self.assertIs(output.tokens, buffer)
                    self.assertIsNone(output.topk_ids)
                    self.assertIsNone(output.weights)
                    self.assertEqual(output.layout.num_tokens_per_expert.shape, (4,))
                    with self.assertRaises(RuntimeError):
                        moe.get_expert_output_buffer()
                    expert = output.tokens
                self.reject(lambda **kw: moe.combine(expert, **kw), native.ll_combine, [dict(handle=expanded_handle)])
                self.reject(
                    lambda **kw: expanded.combine(expanded.get_expert_output_buffer(), **kw),
                    expanded._backend._runtime.cpp_runtime.ll_combine,
                    [dict(handle=handle)],
                )
                moe.combine(expert, handle)
                native.ll_combine.assert_called_once()
                self.assertEqual(native.ll_combine.call_args.args[2], 0 if rank_major else self.weights.data_ptr())
                self.assertEqual(moe.create_overlap_config("dispatch").level, "op")
                if rank_major:
                    moe.dispatch(self.x, self.ids, runtime_max_tokens_per_rank=3)
                    self.assertEqual(native.ll_dispatch.call_args.args[13], 3)
                else:
                    with self.assertRaises(ValueError):
                        moe.dispatch(self.x, self.ids, output_buffer=buffer, runtime_max_tokens_per_rank=3)

    def test_expanded_overlap_helpers_reject_calls(self):
        moe = self.make()
        _, handle = moe.dispatch(self.x, self.ids)
        for op in ("dispatch", "combine"):
            with self.subTest(op=op), self.assertRaises(NotImplementedError):
                moe.create_overlap_config(op, handle=handle)
        for method in (moe.dispatch_async, moe.combine_async):
            with self.assertRaises(NotImplementedError):
                method()


if __name__ == "__main__":
    unittest.main()
