# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native CUDA MegaMoE on an MSCCL++-registered NVLink workspace."""

from .api import MegaMoE, MegaMoEConfig, is_available
from .quantization import dequantize_mxfp8, quantize_mxfp8

__all__ = ["MegaMoE", "MegaMoEConfig", "is_available", "quantize_mxfp8", "dequantize_mxfp8"]
