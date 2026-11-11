# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic CPU routing plans with placement-independent gate weights.

``balanced``, ``zipf``, ``zipf_skewed``, ``lognormal``, and ``hot_experts``
without explicit expert IDs follow the MegaMoE benchmark's routing patterns
(``python/mscclpp/ext/megamoe/benchmark.py`` on ``feature/megamoe``). ``balanced``
draws each token's experts uniformly at random. The other four fix global route
counts per expert, capped so every token uses an expert at most once; all ranks
derive the same global plan and take their own tokens.
"""

import math
import random

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

    if route["kind"] in PROFILE_KINDS and not (route["kind"] == "hot_experts" and "expert_ids" in route):
        ids = profile_routing(config)[rank * tokens : (rank + 1) * tokens]
    elif route["kind"] == "locality":
        ids = locality_routing(config, rank, placement_generator)
    else:
        # balanced: each token picks top_k distinct experts uniformly at random,
        # like MegaMoE's balanced pattern; the other kinds reweight experts.
        probabilities = torch.full((experts,), 1.0 / experts, dtype=torch.float64, device="cpu")
        if route["kind"] == "hot_experts":
            # Explicit hot experts; without expert_ids this kind uses the MegaMoE profile above.
            hot = route["expert_ids"]
            mass = route["probability_mass"]
            probabilities.fill_((1 - mass) / (experts - len(hot)))
            probabilities[hot] = mass / len(hot)
        elif route["kind"] in ("hot_rank", "idle_ranks"):
            world = config["environment"]["num_gpus"]
            local_experts = experts // world
            owner = torch.arange(experts) // local_experts
            if route["kind"] == "hot_rank":
                # All of one rank's experts share probability_mass; the rest share the remainder.
                hot = owner == route.get("rank", DEFAULT_HOT_RANK) % world
                mass = route.get("probability_mass", DEFAULT_HOT_RANK_MASS)
                probabilities.fill_((1 - mass) / (experts - local_experts))
                probabilities[hot] = mass / local_experts
            else:
                # Only every stride-th rank owns experts that receive routes; the others stay idle.
                receiving = owner % route.get("stride", DEFAULT_IDLE_STRIDE) == 0
                probabilities.zero_()
                probabilities[receiving] = 1.0 / int(receiving.sum())
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


PROFILE_KINDS = ("zipf", "zipf_skewed", "hot_experts", "lognormal")
DEFAULT_HOT_RANK = -1
DEFAULT_HOT_RANK_MASS = 0.8
DEFAULT_IDLE_STRIDE = 2
DEFAULT_ZIPF_ALPHA = 1.2
DEFAULT_LOGNORMAL_SIGMA = 0.85


def expert_load_weights(route, world, local_experts, seed):
    """Relative global expert popularity, matching MegaMoE's routing patterns."""
    rng = random.Random(seed)
    kind = route["kind"]
    if kind in ("zipf", "zipf_skewed"):
        # zipf: hot, warm, tail, and unused experts on every rank.
        # zipf_skewed: the same, with rank r's load scaled by 1 / (r + 1)^0.7.
        alpha = route.get("alpha", DEFAULT_ZIPF_ALPHA)
        active = 0.85 if kind == "zipf" else 0.8
        weights = []
        for rank in range(world):
            factor = 1.0 if kind == "zipf" else 1.0 / ((rank + 1) ** 0.7)
            rank_weights = [
                0.0 if j >= max(1, int(local_experts * active)) else factor / ((j + 1) ** alpha)
                for j in range(local_experts)
            ]
            rng.shuffle(rank_weights)
            weights.extend(rank_weights)
        return weights
    if kind == "hot_experts":
        # Bimodal: about two hot experts per 32 take most routes, a quarter are unused.
        weights = []
        for _ in range(world):
            hot = max(1, local_experts // 16)
            empty = local_experts // 4
            rank_weights = []
            for j in range(local_experts):
                if j < hot:
                    rank_weights.append(35.0 * (1.0 + 0.3 * (hot - j)))
                elif j < hot + empty:
                    rank_weights.append(0.0)
                else:
                    rank_weights.append(1.0 + (j % 4) * 0.5)
            rng.shuffle(rank_weights)
            weights.extend(rank_weights)
        return weights
    if kind == "lognormal":
        # Global long-tail popularity, LogNormal(0, sigma) per expert.
        sigma = route.get("sigma", DEFAULT_LOGNORMAL_SIGMA)
        return [math.exp(rng.gauss(0.0, sigma)) for _ in range(world * local_experts)]
    raise ValueError(f"Unsupported routing profile: {kind}")


def scale_counts(weights, total_routes, max_per_expert):
    """Scale nonnegative weights to integer counts summing to total_routes (MegaMoE)."""
    n = len(weights)
    if total_routes > n * max_per_expert:
        raise ValueError("total_routes exceeds capacity n * max_per_expert")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        base, remainder = divmod(total_routes, n)
        return [base + (1 if i < remainder else 0) for i in range(n)]
    raw = [min(float(max_per_expert), float(w) * total_routes / total_weight) for w in weights]
    counts = [min(max_per_expert, int(math.floor(value))) for value in raw]
    # Keep small nonzero tail experts (up to 16 routes) when the budget allows.
    for i, weight in enumerate(weights):
        if weight > 0 and counts[i] == 0 and sum(counts) + 16 <= total_routes:
            counts[i] = min(16, max_per_expert)
    deficit = total_routes - sum(counts)
    if deficit > 0:
        order = sorted(range(n), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
        while deficit > 0:
            progressed = False
            for i in order:
                if deficit == 0:
                    break
                if counts[i] < max_per_expert and weights[i] > 0:
                    step = min(deficit, max_per_expert - counts[i], 16 if deficit >= 16 else deficit)
                    counts[i] += step
                    deficit -= step
                    progressed = True
            if not progressed:
                # Every weighted expert is full: spill the remainder onto unused experts.
                for i in order:
                    if deficit == 0:
                        break
                    if counts[i] < max_per_expert:
                        counts[i] += 1
                        deficit -= 1
    elif deficit < 0:
        excess = -deficit
        for i in sorted(range(n), key=lambda i: counts[i], reverse=True):
            if excess == 0:
                break
            take = min(excess, counts[i])
            counts[i] -= take
            excess -= take
    return counts


def profile_routing(config):
    """Global [world * tokens, top_k] expert IDs for a MegaMoE load profile."""
    world = config["environment"]["num_gpus"]
    tokens = world * config["input"]["tokens_per_rank"]
    top_k = config["model"]["top_k"]
    seed = config["benchmark"]["seed"]
    local_experts = config["model"]["num_experts"] // world
    weights = expert_load_weights(config["routing"], world, local_experts, seed)
    # At most one route per token and expert keeps every token's experts distinct.
    counts = scale_counts(weights, tokens * top_k, tokens)
    # Row-major fill of [top_k, tokens]: an expert's run of at most `tokens`
    # slots never covers the same token column twice.
    grid = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    ids = grid.view(top_k, tokens).t()
    permutation = torch.randperm(tokens, generator=torch.Generator(device="cpu").manual_seed(seed))
    return ids[permutation].contiguous()


def locality_groups(config, rank):
    """Return (local, remote) global expert IDs for a source rank's locality scope."""
    route = config["routing"]
    world = config["environment"]["num_gpus"]
    local_experts = config["model"]["num_experts"] // world
    if route["scope"] == "node" and "gpus_per_node" not in route:
        raise ValueError("routing.gpus_per_node is required for scope=node; the launcher sets it from the topology")
    group = 1 if route["scope"] == "rank" else route["gpus_per_node"]
    first = rank // group * group
    owner = torch.arange(config["model"]["num_experts"]) // local_experts
    local = (owner >= first) & (owner < first + group)
    return local.nonzero().flatten(), (~local).nonzero().flatten()


def locality_routing(config, rank, generator):
    """Route each slot locally with probability local_fraction, keeping experts distinct.

    "Local" means the source rank's experts (scope ``rank``) or its node's experts
    (scope ``node``). Each token draws its local route count, clamped to what the
    local and remote expert sets can supply, then picks distinct experts from each.
    """
    tokens = config["input"]["tokens_per_rank"]
    top_k = config["model"]["top_k"]
    local, remote = locality_groups(config, rank)
    draws = torch.rand((tokens, top_k), generator=generator) < config["routing"]["local_fraction"]
    local_count = draws.sum(dim=1).clamp(min=top_k - len(remote), max=len(local))
    local_pick = local[torch.rand((tokens, len(local)), generator=generator).argsort(dim=1)[:, :top_k]]
    slots = torch.arange(top_k).expand(tokens, -1)
    from_local = slots < local_count[:, None]
    ids = local_pick.gather(1, slots.clamp(max=local_pick.shape[1] - 1))
    if len(remote):
        remote_pick = remote[torch.rand((tokens, len(remote)), generator=generator).argsort(dim=1)[:, :top_k]]
        remote_slot = (slots - local_count[:, None]).clamp(min=0, max=remote_pick.shape[1] - 1)
        ids = torch.where(from_local, ids, remote_pick.gather(1, remote_slot))
    # Shuffle slots so local routes are not always first.
    order = torch.rand((tokens, top_k), generator=generator).argsort(dim=1)
    return ids.gather(1, order)
