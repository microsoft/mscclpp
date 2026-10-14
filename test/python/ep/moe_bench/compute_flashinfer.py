# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional fixed-routing grouped experts. Communication stays BF16 expert-major."""

from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F

if __package__:
    from .compute import make_expert_weights
else:
    from compute import make_expert_weights


def preflight(config, device):
    """Reject untested ABI/hardware combinations before native collectives."""
    required = {"flashinfer-python": "0.7.1", "nvidia-cutlass-dsl": "4.8.0"}
    if config.get("compute", {}).get("weight_dtype") == "mxfp8":
        required.update({"cuda-tile": "1.6.0", "nvidia-cuda-tileiras": "13.4.92"})
    for package, expected in required.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"cutlass_flashinfer requires optional {package}=={expected}") from error
        if actual != expected:
            raise RuntimeError(f"cutlass_flashinfer requires tested {package}=={expected}; found {actual}")
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) != 13:
        raise RuntimeError("cutlass_flashinfer currently requires the tested CUDA 13 PyTorch runtime")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"cutlass_flashinfer requires Blackwell SM100/SM103; got {capability}")
    return required


def quantize(x, bits, swizzled):
    """MX E4M3/E2M1 payload and E8M0 scales for 32 contiguous reduction elements."""
    if bits not in (4, 8):
        raise ValueError("MX payload must use 4 or 8 bits")
    from flashinfer.quantization import mxfp4_quantize, mxfp8_quantize
    from flashinfer.quantization.fp4_quantization import SfLayout

    if bits == 8:
        return mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=swizzled, backend="cute-dsl", enable_pdl=False)
    return mxfp4_quantize(
        x.contiguous(),
        backend="cute-dsl",
        enable_pdl=False,
        sfLayout=SfLayout.layout_128x4 if swizzled else SfLayout.layout_linear,
    )


def fc1_weight(gate, up, mixed):
    """Convert Torch [K,N] weights to the kernel's [N,K] SwiGLU row ordering."""
    if mixed:
        # CuTe consumes alternating groups of 64 up rows and 64 gate rows.
        hidden, intermediate = gate.shape
        return torch.stack(
            (up.T.reshape(intermediate // 64, 64, hidden), gate.T.reshape(intermediate // 64, 64, hidden)), dim=1
        ).reshape(2 * intermediate, hidden)
    return torch.cat((up.T, gate.T), dim=0)


def make_schedule(counts, tile, sentinel, device):
    """Each compact row denotes one expert SLOT, never an original input token."""
    slots, experts, limits = [], [], []
    offset = token = 0
    for expert, count in enumerate(counts):
        padded = (count + tile - 1) // tile * tile
        slots.extend(range(token, token + count))
        slots.extend([sentinel] * (padded - count))
        experts.extend([expert] * (padded // tile))
        limits.extend([offset + count] * (padded // tile))
        token += count
        offset += padded
    return tuple(
        torch.tensor(values, dtype=torch.int32, device=device)
        for values in (slots, experts, limits, [len(experts)], [len(slots)])
    )


class FlashInferMLP:
    """Pack valid expert slots, compute unweighted outputs, and restore slot order."""

    @torch.inference_mode()
    def __init__(self, config, rank, counts, device):
        versions = preflight(config, device)
        self.counts = list(counts)
        self.mixed = config["compute"]["weight_dtype"] == "mxfp4"
        self.bits = 4 if self.mixed else 8
        model = config["model"]
        self.hidden, self.inter = model["d_model"], model["intermediate_size"]
        self.rows = sum(counts)
        capacity = config["environment"]["num_gpus"] * config["input"]["tokens_per_rank"]
        indices = [expert * capacity + row for expert, count in enumerate(counts) for row in range(count)]
        self.indices = torch.tensor(indices, device=device, dtype=torch.int64)
        self.schedule = make_schedule(counts, 128 if self.mixed else 16, -1 if self.mixed else self.rows, device)
        self.metadata = {
            "compute_backend": "cutlass_flashinfer",
            "compute_implementation": "cute_native_mxfp8_mxfp4" if self.mixed else "cutile_dequant_bf16",
            "compute_dependency_versions": versions,
            "communication_dtype": "bf16",
            "compute_activation_dtype": "mxfp8" if self.mixed else "bf16",
            "compute_weight_dtype": f"mxfp{self.bits}",
            "native_mixed_mma": self.mixed,
            "compute_input_layout": "compact_expert_assignment_K",
            "compute_output_layout": "bf16_expert_major_unweighted",
            "weight_payload": "packed_e2m1_low_nibble_first" if self.mixed else "e4m3fn",
            "scale_format": "e8m0",
            "scale_block_size": 32,
            "weight_layout": "expert_N_K (packed K/2 for MXFP4)",
            "weight_scale_layout": "128x4_mma_swizzled" if self.mixed else "linear_N_Kdiv32",
            "activation_scale_layout": "input_linear; intermediate_128x4_mma_swizzled" if self.mixed else "none",
            "fc1_row_order": "up64_gate64_interleaved" if self.mixed else "up_then_gate",
            "compute_routing": "topk1_unit_weights_per_compact_expert_slot; native_combine_applies_routing_weights",
            "mlp_timed_operations": "pack, input quantization (MXFP8 only), grouped GEMMs, SwiGLU, "
            "intermediate quantization (MXFP8 only), output zeroing/scatter, unpack, launch gaps",
            "weight_quantization_timed": False,
            "initial_activation_quantization_timed": self.mixed,
        }
        self.weights, self.scales = [], []
        for n, k in ((2 * self.inter, self.hidden), (self.hidden, self.inter)):
            self.weights.append(
                torch.empty(
                    len(counts),
                    n,
                    k // (2 if self.mixed else 1),
                    dtype=torch.uint8 if self.mixed else torch.float8_e4m3fn,
                    device=device,
                )
            )
            self.scales.append(torch.empty(len(counts), n, k // 32, dtype=torch.uint8, device=device))
        for expert in range(len(counts)):
            raw = make_expert_weights(model, rank * len(counts) + expert, config["benchmark"]["seed"], device)
            first = fc1_weight(raw[0], raw[1], self.mixed)
            for projection, matrix in enumerate((first, raw[2].T)):
                payload, scales = quantize(matrix, self.bits, self.mixed)
                self.weights[projection][expert].copy_(payload)
                self.scales[projection][expert].copy_(scales.reshape_as(self.scales[projection][expert]))
                del payload, scales
            del raw, first, matrix
        self.packed = torch.empty(self.rows, self.hidden, dtype=torch.bfloat16, device=device)
        self.result = torch.empty_like(self.packed)
        if self.mixed:
            from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
                blockscaled_contiguous_gather_grouped_gemm_act_fusion,
            )
            from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
                blockscaled_contiguous_grouped_gemm_finalize_fusion,
            )

            self.fc1 = blockscaled_contiguous_gather_grouped_gemm_act_fusion
            self.fc2 = blockscaled_contiguous_grouped_gemm_finalize_fusion
            padded = self.schedule[0].numel()
            self.activation = torch.empty(padded, self.inter, dtype=torch.float8_e4m3fn, device=device)
            self.activation_scales = torch.empty(padded * self.inter // 32, dtype=torch.uint8, device=device)
            self.qx = torch.empty(self.rows, self.hidden, dtype=torch.float8_e4m3fn, device=device)
            self.sx = torch.empty(self.rows * self.hidden // 32, dtype=torch.uint8, device=device)
            self.alpha = torch.ones(len(counts), device=device)
            self.unit_weights = torch.ones(self.rows, 1, device=device)
            self.common = dict(
                a_dtype="float8_e4m3fn",
                b_dtype="float4_e2m1fn",
                sf_dtype="float8_e8m0fnu",
                sf_vec_size=32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                enable_pdl=False,
            )
        else:
            from flashinfer.fused_moe.cutile.fp8 import GemmConfig, grouped_gemm

            self.grouped_gemm = grouped_gemm
            self.gemm1 = torch.empty(self.rows, 2 * self.inter, dtype=torch.bfloat16, device=device)
            self.activation = torch.empty(self.rows, self.inter, dtype=torch.bfloat16, device=device)
            self.empty = torch.empty(0, device=device)
            self.common = dict(
                top_k=1,
                block_size=16,
                config=GemmConfig(64, 128, 1),
                activation_fp8=False,
                block_scaled=True,
                weight_fp4=False,
            )

    @torch.inference_mode()
    def __call__(self, tokens, out):
        if not self.rows:
            # A rank can receive no assignments even when the global batch is nonempty.
            out.zero_()
            return out
        torch.index_select(tokens.view(-1, self.hidden), 0, self.indices, out=self.packed)
        slots, experts, limits, tiles, padded = self.schedule
        if self.mixed:
            # Allocations are graph-private during capture. Both quantization AND
            # the copies run on every replay, not just when the graph is constructed.
            qx, sx = quantize(self.packed, 8, False)
            self.qx.copy_(qx)
            self.sx.copy_(sx)
            self.fc1(
                self.qx,
                self.weights[0],
                self.sx,
                self.scales[0],
                self.alpha,
                experts,
                limits,
                slots,
                tiles,
                out=self.activation,
                out_scale=self.activation_scales,
                topk=1,
                c_dtype="float8_e4m3fn",
                quantize_output=True,
                **self.common,
            )
            self.result.zero_()
            self.fc2(
                self.activation,
                self.weights[1],
                self.activation_scales,
                self.scales[1],
                self.alpha,
                experts,
                tiles,
                limits,
                slots,
                self.unit_weights,
                out=self.result,
                **self.common,
            )
        else:
            self.grouped_gemm(
                self.packed,
                self.empty,
                self.weights[0],
                self.scales[0],
                slots,
                experts,
                padded,
                self.gemm1,
                **self.common,
            )
            self.activation.copy_(F.silu(self.gemm1[:, self.inter :].float()) * self.gemm1[:, : self.inter].float())
            self.grouped_gemm(
                self.activation,
                self.empty,
                self.weights[1],
                self.scales[1],
                slots,
                experts,
                padded,
                self.result,
                **self.common,
            )
        out.view(-1, self.hidden).index_copy_(0, self.indices, self.result)
        return out
