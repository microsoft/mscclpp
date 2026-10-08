# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native CUDA MegaMoE on an MSCCL++-registered NVLink workspace."""

from .api import MegaMoE, MegaMoEConfig, is_available
from .jit import CompiledKernel, KernelConfig, W4A8KernelConfig, compile_kernel, load_cached_kernel
from .quantization import dequantize_mxfp4, dequantize_mxfp8, quantize_mxfp4, quantize_mxfp8
from .resident_tuning import ResidentTuning, ResidentTuningMismatchError, resolve_resident_tuning
from .w4a8 import (
    W4A8KernelProfileMismatchError,
    load_w4a8_kernel_profiles,
    resolve_w4a8_kernel_config,
)

__all__ = [
    "MegaMoE",
    "MegaMoEConfig",
    "is_available",
    "quantize_mxfp8",
    "dequantize_mxfp8",
    "quantize_mxfp4",
    "dequantize_mxfp4",
    "KernelConfig",
    "W4A8KernelConfig",
    "CompiledKernel",
    "compile_kernel",
    "load_cached_kernel",
    "W4A8KernelProfileMismatchError",
    "load_w4a8_kernel_profiles",
    "resolve_w4a8_kernel_config",
    "ResidentTuning",
    "ResidentTuningMismatchError",
    "resolve_resident_tuning",
]
