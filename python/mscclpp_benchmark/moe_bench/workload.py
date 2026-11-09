# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Seeded synthetic inputs and expert weights shared by compute backends."""

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


@torch.inference_mode()
def reference_output(config, x, ids, weights):
    """Compute routed BF16 experts locally, independently of dispatch/combine."""
    model = config["model"]
    world = config["environment"]["num_gpus"]
    local_experts = model["num_experts"] // world
    routes = torch.zeros(x.shape[0], model["top_k"], model["d_model"], device=x.device, dtype=torch.bfloat16)
    for expert in range(model["num_experts"]):
        gate, up, down = make_expert_weights(model, expert, config["benchmark"]["seed"], x.device)
        token, slot = (ids == expert).nonzero(as_tuple=True)
        selected = x[token]
        routes[token, slot] = (F.silu(selected @ gate) * (selected @ up)) @ down
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
    return expected.to(torch.bfloat16)
