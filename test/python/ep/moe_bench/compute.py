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


def estimate_tensor_bytes(config, counts):
    """Conservative tensor budget, excluding native runtime and library allocations."""
    model, inputs, environment = config["model"], config["input"], config["environment"]
    hidden, intermediate = model["d_model"], model["intermediate_size"]
    local_experts = model["num_experts"] // environment["num_gpus"]
    weights = 6 * local_experts * hidden * intermediate
    capacity = environment["num_gpus"] * inputs["tokens_per_rank"]
    # Native runtime also owns a dispatch view; reserve one extra padded tensor.
    dispatch_and_output = 3 * 2 * local_experts * capacity * hidden
    workspace = 4 * max(counts, default=0) * intermediate
    inputs_and_output = 4 * inputs["tokens_per_rank"] * hidden
    return weights + dispatch_and_output + workspace + inputs_and_output
