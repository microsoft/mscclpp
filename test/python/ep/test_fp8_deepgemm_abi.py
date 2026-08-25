# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU-only source-contract checks for the native FP8 DeepGEMM ABI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_compiled_extension_owns_abi_version():
    api = _read("src/ext/ep/include/api.cuh")
    bindings = _read("src/ext/ep/bindings.cpp")
    loader = _read("python/mscclpp/ep/_cpp.py")
    public = _read("python/mscclpp/ep/__init__.py")

    assert "inline constexpr int Fp8DeepGemmAbi = 1" in api
    assert "inline constexpr int Fp8DeepGemmScaleBlockSize = 128" in api
    assert 'm.attr("FP8_DEEPGEMM_ABI")' in bindings
    assert "FP8_DEEPGEMM_ABI = _cpp.FP8_DEEPGEMM_ABI" in loader
    assert '"FP8_DEEPGEMM_ABI"' in public


def test_deepgemm_ue8m0_rule_is_bit_exact_and_shared_by_payload_and_metadata():
    quantization = _read("src/ext/ep/include/quantization.cuh")
    assert "Fp8DeepGemmMinAmax = 1e-4f" in quantization
    assert "ceilToUe8m0(maxAbs / Fp8E4M3MaxValue)" in quantization
    assert "bits & 0x7fffffu" in quantization
    assert "const float quantScale = 1.0f / scale" in quantization
    assert "*scaleOut = scale" in quantization
    assert "log2f" not in quantization


def test_fp8_scale_layout_and_native_calls_fail_closed():
    frontend = _read("python/mscclpp/ep/low_latency.py")
    runtime = _read("src/ext/ep/ll_runtime.cc")
    assert "expected_shape = (self.num_local_experts, slots_per_expert, num_scales)" in frontend
    assert "expected_stride = (num_scales * slots_per_expert, 1, slots_per_expert)" in frontend
    assert "hidden == hidden_ && numTopk == numTopk_" in runtime
    assert "dispatchLayout == DispatchLayout::EXPERT_MAJOR" in runtime
    assert "outputScales != nullptr" in runtime


def test_device_receipts_cover_graph_replay_without_host_callbacks():
    config = _read("src/ext/ep/low_latency/config.cuh")
    dispatch = _read("src/ext/ep/low_latency/dispatch.cu")
    combine = _read("src/ext/ep/low_latency/combine.cu")
    frontend = _read("python/mscclpp/ep/low_latency.py")
    assert "ExecutionReceipt* executionReceipt_" in config
    assert "receipt->dispatches_ += 1" in dispatch
    assert "receipt->fp8Dispatches_ += 1" in dispatch
    assert "executionReceipt_->combines_ += 1" in combine
    assert "ll_execution_receipt" in frontend
