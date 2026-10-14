# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixed expert-major BF16 reference compute, without padded GEMMs."""

import math

import torch
import torch.nn.functional as F


def make_input(config, rank, device):
    generator = torch.Generator(device=device).manual_seed(config["benchmark"]["seed"] + rank)
    return torch.randn(
        config["input"]["tokens_per_rank"],
        config["model"]["d_model"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )


def make_expert_weights(model, expert_id, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + 100_000 + expert_id)
    hidden, intermediate = model["d_model"], model["intermediate_size"]
    weights = []
    for rows, cols in ((hidden, intermediate), (hidden, intermediate), (intermediate, hidden)):
        weight = torch.randn(rows, cols, dtype=torch.bfloat16, device=device, generator=generator)
        weight.mul_(1 / math.sqrt(rows))
        weights.append(weight)
    return tuple(weights)


class ExpertMLP:
    """Reuse weights/workspaces and write unweighted results in dispatch slot order."""

    def __init__(self, config, rank, counts, device):
        self.metadata = {
            "compute_backend": "torch",
            "compute_implementation": "torch_bf16",
            "communication_dtype": "bf16",
            "compute_activation_dtype": "bf16",
            "compute_weight_dtype": "bf16",
            "mlp_timed_operations": "valid-row GEMMs, SwiGLU, launch gaps",
        }
        self.counts = list(counts)
        self.weights = [
            make_expert_weights(config["model"], rank * len(counts) + expert, config["benchmark"]["seed"], device)
            for expert in range(len(counts))
        ]
        shape = (max(counts, default=0), config["model"]["intermediate_size"])
        self.gate = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.up = torch.empty_like(self.gate)

    @torch.inference_mode()
    def __call__(self, tokens, out):
        for expert, (count, weights) in enumerate(zip(self.counts, self.weights)):
            if count == 0:
                continue
            gate, up = self.gate[:count], self.up[:count]
            torch.mm(tokens[expert, :count], weights[0], out=gate)
            torch.mm(tokens[expert, :count], weights[1], out=up)
            F.silu(gate, inplace=True)
            gate.mul_(up)
            torch.mm(gate, weights[2], out=out[expert, :count])
        return out


def create_mlp(config, rank, counts, device):
    """Load optional compute dependencies only for the selected backend."""
    if config["environment"]["compute_backend"] == "torch":
        return ExpertMLP(config, rank, counts, device)
    if config["environment"]["compute_backend"] != "cutlass_flashinfer":
        raise ValueError("Unsupported compute backend")
    if __package__:
        from .compute_flashinfer import FlashInferMLP
    else:
        from compute_flashinfer import FlashInferMLP
    return FlashInferMLP(config, rank, counts, device)


def estimate_tensor_bytes(config, counts):
    """Conservative tensor budget, excluding native runtime and library allocations."""
    model, inputs, environment = config["model"], config["input"], config["environment"]
    hidden, intermediate = model["d_model"], model["intermediate_size"]
    local_experts = model["num_experts"] // environment["num_gpus"]
    weights = 6 * local_experts * hidden * intermediate
    capacity = environment["num_gpus"] * inputs["tokens_per_rank"]
    # Native runtime also owns a dispatch view; reserve one extra padded tensor.
    dispatch_and_output = 3 * 2 * local_experts * capacity * hidden
    mode, layout = environment.get("mode", "latency"), environment.get("layout", "expert_major")
    if mode == "throughput" or layout != "expert_major":
        rows = capacity
        if environment["communication_backend"] == "nccl" and layout == "expert_major":
            rows *= min(model["top_k"], local_experts)
        # Native receive/combine, FP32 reduction, gather/weighted temporaries,
        # IDs/weights, per-expert sort indices and graph-private allocations.
        adapter_bytes = 24 * rows * hidden + 48 * rows * model["top_k"] * local_experts
        if environment["cuda_graph"] == "mlp":
            adapter_bytes *= 2
        dispatch_and_output += adapter_bytes
        if environment["communication_backend"] == "mscclpp" and mode == "throughput":
            # RecvPoolConfig: fixed 65536 rows, 16384 payload + 768 metadata bytes.
            dispatch_and_output += 65536 * (16384 + 768) + 4 * 65536 * environment["num_gpus"] + 1024**2
    workspace = 4 * max(counts, default=0) * intermediate
    inputs_and_output = 4 * inputs["tokens_per_rank"] * hidden
    if environment["compute_backend"] == "cutlass_flashinfer":
        mixed = config.get("compute", {}).get("weight_dtype") == "mxfp4"
        # Both dimensions are multiples of 128, so weight scales need no padding.
        elements = 3 * local_experts * hidden * intermediate
        weights = elements // (2 if mixed else 1) + elements // 32
        # Only one expert's original BF16 matrices and packing/quantization
        # temporaries are live at once, never an expanded BF16 expert bank.
        staging = 14 * hidden * intermediate
        rows = sum(counts)
        padded = sum((count + 127) // 128 * 128 for count in counts)
        workspace = 4 * rows * hidden  # BF16 packed input and output
        if mixed:
            workspace += 2 * (rows * hidden + rows * hidden // 32)  # quant outputs + stable copies
            workspace += padded * intermediate + padded * intermediate // 32
        else:
            workspace += 18 * rows * intermediate  # BF16 FC1/activation plus FP32 SwiGLU temporaries
        workspace += 32 * (padded + rows + local_experts)  # schedules, indices, unit weights
        if environment["cuda_graph"] == "mlp":
            workspace *= 2  # graph-private quant/elementwise allocations and warmup pool
        # Reserve compiler/kernel and allocator overhead; native comm remains extra.
        workspace += 64 * 1024**2
        return weights + staging + dispatch_and_output + workspace + inputs_and_output
    return weights + dispatch_and_output + workspace + inputs_and_output
