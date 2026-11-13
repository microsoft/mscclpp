# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent MXFP8/MXFP4 reference for checking distributed MoE output.

Weights use the production FlashInfer quantizer, but scale/nibble decoding,
activation quantization, SwiGLU, routing, and combine are reimplemented here.
"""

import torch
import torch.nn.functional as F

if __package__:
    from .workload import make_expert_weights
else:
    from workload import make_expert_weights

# Element-wise |actual - expected| <= atol + rtol * |expected|, as in MegaMoE's check,
# plus ROUNDING_SLACK * sum_k |w_k * y_k| when the routed contributions are known.
CHECK_RTOL = 0.05
CHECK_ATOL = 0.01
# Expert outputs and each destination's weighted partial are rounded to BF16
# before the final sum. A one-ULP difference in either is at most 2**-7 of that
# value, so together they stay within 2**-6 of the summed contribution magnitudes,
# even when partials nearly cancel.
ROUNDING_SLACK = 2.0**-6
E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def unswizzle(sf, rows, k):
    """Undo the 128x4 MMA scale-factor swizzle."""
    return (
        sf.reshape(-1)
        .view((rows + 127) // 128, (k // 32 + 3) // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape((rows + 127) // 128 * 128, (k // 32 + 3) // 4 * 4)[:rows, : k // 32]
    )


def decode(q, sf, bits, swizzled=False):
    """Decode MX payload bytes and E8M0 scales without FlashInfer dequantization."""
    rows, cols = q.shape
    k = cols * (2 if bits == 4 else 1)
    codes = unswizzle(sf, rows, k) if swizzled else sf.reshape(rows, k // 32)
    scales = torch.exp2(codes.to(torch.int32).float() - 127)
    if bits == 4:
        lut = torch.tensor(E2M1_VALUES, device=q.device)
        values = lut[torch.stack((q & 15, q >> 4), dim=-1).reshape(rows, k).long()]
    else:
        values = q.float()
    return (values.reshape(rows, k // 32, 32) * scales[:, :, None]).reshape(rows, k)


def quantize_mxfp8(x):
    """Round to MXFP8 E4M3 with one power-of-two scale per 32 elements."""
    blocks = x.float().reshape(x.shape[0], -1, 32)
    maximum = blocks.abs().amax(-1)
    exponent = torch.ceil(torch.log2((maximum / 448).clamp_min(2.0**-126)))
    scale = torch.exp2(exponent)
    values = (blocks / scale[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (values * scale[:, :, None]).reshape_as(x)


def decoded_expert_weights(model, expert, seed, device):
    """Return decoded MXFP4 ``(fc1 [2I, H], fc2 [H, I])`` for one global expert."""
    if __package__:
        from .compute_backends.flashinfer import quantize_weights
    else:
        from compute_backends.flashinfer import quantize_weights

    gate, up, down = make_expert_weights(model, expert, seed, device)
    hidden, intermediate = gate.shape
    # Rebuild the kernel's interleaved 64-row up/gate groups independently.
    first = torch.empty(2 * intermediate, hidden, device=device, dtype=torch.bfloat16)
    for start in range(0, intermediate, 64):
        first[2 * start : 2 * start + 64] = up.T[start : start + 64]
        first[2 * start + 64 : 2 * start + 128] = gate.T[start : start + 64]
    return tuple(decode(*quantize_weights(weight), 4, True) for weight in (first, down.T))


def expert_output(x, weights):
    """MXFP8 activations x decoded MXFP4 weights, with SwiGLU between GEMMs."""
    first, down = weights
    projected = quantize_mxfp8(x) @ first.T
    chunks = projected.reshape(x.shape[0], -1, 2, 64)
    up = chunks[:, :, 0].reshape(x.shape[0], -1)
    gate = chunks[:, :, 1].reshape(x.shape[0], -1)
    return (quantize_mxfp8(F.silu(gate) * up) @ down.T).bfloat16()


def relative_error(actual, expected):
    """Relative L2 error; non-finite output always fails."""
    if not torch.isfinite(actual).all():
        return float("inf")
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def compare_outputs(actual, expected, rtol=CHECK_RTOL, atol=CHECK_ATOL, magnitude=None):
    """Element-wise comparison statistics, plus the first out-of-tolerance element.

    ``magnitude`` is ``sum_k |w_k * y_k|`` per element; it adds rounding slack for
    BF16 partial sums that nearly cancel.
    """
    actual, expected = actual.float(), expected.float()
    finite = torch.isfinite(actual)
    allowed = atol + rtol * expected.abs()
    if magnitude is not None:
        allowed = allowed + ROUNDING_SLACK * magnitude.float()
    difference = (actual - expected).abs()
    mismatched = ~finite | (difference > allowed)
    error, allowed = difference[finite], allowed[finite]
    # Fraction of the allowed error used by the worst element; above 1.0 fails.
    used = torch.where(allowed > 0, error / allowed, (error > 0).float() * float("inf"))
    stats = {
        "max_tolerance_fraction": used.max().item() if used.numel() else 0.0,
        "max_abs_error": error.max().item() if error.numel() else 0.0,
        "mean_abs_error": error.mean().item() if error.numel() else 0.0,
        "mismatched_elements": int(mismatched.sum()),
        "non_finite_elements": int((~finite).sum()),
        "total_elements": actual.numel(),
        "relative_l2": relative_error(actual, expected) if finite.all() else None,
    }
    first = None
    if stats["mismatched_elements"]:
        token, column = mismatched.nonzero()[0].tolist()
        first = {
            "token": token,
            "column": column,
            "actual": actual[token, column].item(),
            "expected": expected[token, column].item(),
        }
    return stats, first


@torch.inference_mode()
def reference_output(config, x, ids, weights, with_magnitude=False):
    """Compute this rank's routed layer output independently of dispatch/combine.

    With ``with_magnitude``, also return ``sum_k |w_k * y_k|`` for rounding slack.
    """
    model = config["model"]
    world = config["environment"]["num_gpus"]
    local_experts = model["num_experts"] // world
    routes = torch.zeros(x.shape[0], model["top_k"], model["d_model"], device=x.device, dtype=torch.bfloat16)
    for expert in torch.unique(ids).tolist():
        token, slot = (ids == expert).nonzero(as_tuple=True)
        decoded = decoded_expert_weights(model, expert, config["benchmark"]["seed"], x.device)
        routes[token, slot] = expert_output(x[token], decoded)
    expected = torch.zeros_like(x, dtype=torch.float32)
    if config["environment"]["communication_backend"] == "nccl":
        for slot in range(model["top_k"]):
            expected.addcmul_(routes[:, slot].float(), weights[:, slot].unsqueeze(-1))
    else:
        # MSCCL++ rounds each destination rank's weighted partial to BF16.
        for destination in range(world):
            partial = torch.zeros_like(expected)
            for slot in range(model["top_k"]):
                selected = ids[:, slot] // local_experts == destination
                weight = weights[:, slot].masked_fill(~selected, 0).unsqueeze(-1)
                partial.addcmul_(routes[:, slot].float(), weight)
            expected.add_(partial.to(torch.bfloat16).float())
    if not with_magnitude:
        return expected.to(torch.bfloat16)
    magnitude = torch.zeros_like(expected)
    for slot in range(model["top_k"]):
        magnitude.addcmul_(routes[:, slot].float().abs(), weights[:, slot].abs().unsqueeze(-1))
    return expected.to(torch.bfloat16), magnitude
