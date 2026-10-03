# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Canonical MXFP8/MXFP4 K32 utilities without kernel-specific swizzles."""


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


def quantize_mxfp4(weights):
    """Quantize finite floating-point [...,K] weights into packed E2M1 plus E8M0.

    The returned uint8 tensor has shape [...,K//2]; its lower nibble is the even
    K element. Scales are canonical row-major uint8 [...,K//32].
    Scales round upward, values round to nearest even, and zero blocks use scale
    one. Like ``quantize_mxfp8``, this is an offline helper, not an inference kernel.
    """
    import torch

    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        raise TypeError("weights must be a floating-point torch.Tensor")
    if weights.ndim < 2 or weights.shape[-1] == 0 or weights.shape[-1] % 32:
        raise ValueError("weights must have at least two dimensions and K divisible by 32")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("MXFP4 weights must be finite")
    blocks = weights.float().reshape(*weights.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(maximum / 6.0))
    exponent = torch.where(maximum == 0, 0, exponent).clamp(-126, 127)
    normalized = (blocks / torch.exp2(exponent).unsqueeze(-1)).clamp(-6, 6)
    boundaries = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=weights.device)
    absolute = normalized.abs()
    magnitude = torch.bucketize(absolute, boundaries).to(torch.uint8)
    round_up_ties = (absolute == 0.75) | (absolute == 1.75) | (absolute == 3.5)
    magnitude += round_up_ties.to(torch.uint8)
    encoded = magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)
    encoded = encoded.reshape(*weights.shape[:-1], weights.shape[-1] // 2, 2)
    packed = encoded[..., 0] | (encoded[..., 1] << 4)
    return packed.contiguous(), (exponent + 127).to(torch.uint8).contiguous()


def dequantize_mxfp4(weights, scales, *, dtype=None):
    """Decode canonical packed E2M1 weights and row-major uint8 E8M0 K32 scales."""
    import torch

    if not isinstance(weights, torch.Tensor) or weights.dtype != torch.uint8:
        raise TypeError("weights must have dtype uint8")
    if weights.ndim < 2 or weights.shape[-1] == 0 or weights.shape[-1] % 16:
        raise ValueError("packed weights must represent K divisible by 32")
    logical_k = weights.shape[-1] * 2
    if (
        not isinstance(scales, torch.Tensor)
        or scales.dtype != torch.uint8
        or scales.device != weights.device
        or tuple(scales.shape) != (*weights.shape[:-1], logical_k // 32)
    ):
        raise ValueError("scales must be uint8 [...,K//32] on the weight device")
    low = weights & 0x0F
    high = weights >> 4
    encoded = torch.stack((low, high), dim=-1).reshape(*weights.shape[:-1], logical_k)
    levels = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=weights.device)
    values = levels[(encoded & 7).long()]
    values = torch.where((encoded & 8) != 0, -values, values)
    factors = torch.exp2(scales.float() - 127)
    factors = torch.where(scales == 255, float("nan"), factors)
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    return (blocks * factors.unsqueeze(-1)).reshape(values.shape).to(dtype or torch.bfloat16)
