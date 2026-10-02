# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic CPU routing plans with placement-independent gate weights."""

import torch

if __package__:
    from .config import validate_config
else:
    from config import validate_config


def make_routing(config: dict, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distinct expert IDs (int64) and normalized weights (float32), [T, K]."""
    validate_config(config)
    if type(rank) is not int or not 0 <= rank < config["environment"]["num_gpus"]:
        raise ValueError("rank must be an integer in [0, environment.num_gpus)")
    tokens = config["input"]["tokens_per_rank"]
    experts = config["model"]["num_experts"]
    top_k = config["model"]["top_k"]
    seed = config["benchmark"]["seed"]
    route = config["routing"]
    placement_generator = torch.Generator(device="cpu").manual_seed((seed + rank) % 2**63)
    weight_generator = torch.Generator(device="cpu").manual_seed(((seed + rank) ^ 0x5DEECE66D) % 2**63)
    weights = torch.rand((tokens, top_k), dtype=torch.float32, device="cpu", generator=weight_generator) + 1.0
    weights /= weights.sum(dim=1, keepdim=True)

    if route["kind"] == "balanced":
        # Every rank uses the same permutation so global counts differ by at most one.
        global_generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(experts, generator=global_generator, device="cpu")
        global_slots = torch.arange(tokens * top_k, dtype=torch.int64, device="cpu") + rank * tokens * top_k
        ids = permutation[global_slots.remainder(experts)].reshape(tokens, top_k)
    else:
        probabilities = torch.full((experts,), 1.0 / experts, dtype=torch.float64, device="cpu")
        if route["kind"] == "hot_experts":
            hot = route["expert_ids"]
            mass = route["probability_mass"]
            probabilities.fill_((1 - mass) / (experts - len(hot)))
            probabilities[hot] = mass / len(hot)
        ids = torch.empty((tokens, top_k), dtype=torch.int64, device="cpu")
        # Bound the temporary multinomial matrix for large token counts.
        for start in range(0, tokens, 1024):
            stop = min(start + 1024, tokens)
            ids[start:stop] = torch.multinomial(
                probabilities.expand(stop - start, -1),
                top_k,
                replacement=False,
                generator=placement_generator,
            )
    return ids, weights
