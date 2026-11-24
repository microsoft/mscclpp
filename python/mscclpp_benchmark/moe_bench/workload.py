# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Seeded synthetic inputs and expert weights shared by compute backends."""

import math

import torch


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
