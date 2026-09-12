# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Canonical MXFP8 K32 utilities; these do not apply a kernel-specific swizzle."""


def quantize_mxfp8(weights, *, e5m2=False):
    """Quantize finite floating-point [...,K] weights into FP8 plus uint8 E8M0.

    Each group of 32 consecutive K elements uses a power-of-two scale, rounded
    upward to avoid saturation. Zero groups use scale 1. This offline helper may
    synchronize and is not intended for CUDA Graph capture or timed inference.
    """
    import torch

    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        raise TypeError("weights must be a floating-point torch.Tensor")
    if weights.ndim < 2 or weights.shape[-1] == 0 or weights.shape[-1] % 32:
        raise ValueError("weights must have at least two dimensions and K divisible by 32")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("MXFP8 weights must be finite")
    dtype = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
    blocks = weights.float().reshape(*weights.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(maximum / torch.finfo(dtype).max))
    exponent = torch.where(maximum == 0, 0, exponent).clamp(-126, 127)
    scale = torch.exp2(exponent)
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max)
    return quantized.reshape(weights.shape).to(dtype).contiguous(), (exponent + 127).to(torch.uint8).contiguous()


def dequantize_mxfp8(weights, scales, *, dtype=None):
    """Decode canonical FP8 weights and row-major uint8 E8M0 K32 scales."""
    import torch

    if not isinstance(weights, torch.Tensor) or weights.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("weights must have dtype float8_e4m3fn or float8_e5m2")
    if weights.ndim < 2 or weights.shape[-1] == 0 or weights.shape[-1] % 32:
        raise ValueError("weights must have K divisible by 32")
    if (
        not isinstance(scales, torch.Tensor)
        or scales.dtype != torch.uint8
        or scales.device != weights.device
        or tuple(scales.shape) != (*weights.shape[:-1], weights.shape[-1] // 32)
    ):
        raise ValueError("scales must be uint8 [...,K//32] on the weight device")
    factors = torch.exp2(scales.float() - 127)
    factors = torch.where(scales == 255, float("nan"), factors)
    blocks = weights.float().reshape(*weights.shape[:-1], -1, 32)
    return (blocks * factors.unsqueeze(-1)).reshape(weights.shape).to(dtype or torch.bfloat16)
