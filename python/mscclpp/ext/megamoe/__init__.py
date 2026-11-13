# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native CUDA MegaMoE on an MSCCL++-registered NVLink workspace."""

from .api import BucketedMegaMoE, MegaMoE, MegaMoEConfig, SharedExpertConfig, is_available
from .jit import CompiledKernel, W4A8KernelConfig, compile_kernel, find_cached_kernel, load_cached_kernel
from .quantization import (
    dequantize_fp8_static,
    dequantize_mxfp4,
    dequantize_mxfp8,
    quantize_fp8_static,
    quantize_mxfp4,
    quantize_mxfp8,
)
from .policy import (
    MegaMoEPolicy,
    MegaMoEPolicyMismatchError,
    load_megamoe_policies,
    resolve_megamoe_policies,
    resolve_megamoe_policy,
)
from .resident_tuning import ResidentTuning, ResidentTuningMismatchError, resolve_resident_tuning
from .w4a8 import (
    W4A8KernelProfileMismatchError,
    load_w4a8_kernel_profiles,
    resolve_w4a8_kernel_config,
)

__all__ = [
    "MegaMoE",
    "BucketedMegaMoE",
    "MegaMoEConfig",
    "SharedExpertConfig",
    "is_available",
    "quantize_mxfp8",
    "dequantize_mxfp8",
    "quantize_mxfp4",
    "dequantize_mxfp4",
    "quantize_fp8_static",
    "dequantize_fp8_static",
    "W4A8KernelConfig",
    "CompiledKernel",
    "compile_kernel",
    "load_cached_kernel",
    "find_cached_kernel",
    "W4A8KernelProfileMismatchError",
    "load_w4a8_kernel_profiles",
    "resolve_w4a8_kernel_config",
    "MegaMoEPolicy",
    "MegaMoEPolicyMismatchError",
    "load_megamoe_policies",
    "resolve_megamoe_policies",
    "resolve_megamoe_policy",
    "ResidentTuning",
    "ResidentTuningMismatchError",
    "resolve_resident_tuning",
]
