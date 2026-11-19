# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Realized per-GPU send/receive traffic derived from the fixed routing.

``routes`` count token-to-expert assignments (expert-compute rows). ``tokens``
count token copies per destination GPU: a token routed to two experts on the
same GPU counts once there. Routing is fixed, so these are exact for every
timed iteration; receive-side route counts match the native dispatch counts
the worker verifies at startup.
"""

import torch


def send_counts(ids, local_experts, world):
    """Return one source rank's (routes, tokens) sent to each destination GPU."""
    destination = ids // local_experts
    routes = torch.bincount(destination.flatten(), minlength=world)
    tokens = torch.stack([(destination == rank).any(dim=1).sum() for rank in range(world)])
    return routes.tolist(), tokens.tolist()


def traffic_summary(route_matrix, token_matrix, gpus_per_node):
    """Summarize [source][destination] matrices into per-GPU and global traffic."""
    world = len(route_matrix)

    def split(matrix, rank, sent):
        values = [(peer, matrix[rank][peer] if sent else matrix[peer][rank]) for peer in range(world)]
        return {
            "total": sum(count for _, count in values),
            "remote": sum(count for peer, count in values if peer != rank),
            "cross_node": sum(count for peer, count in values if peer // gpus_per_node != rank // gpus_per_node),
        }

    per_rank = []
    for rank in range(world):
        entry = {"rank": rank}
        for direction, sent in (("sent", True), ("received", False)):
            for unit, matrix in (("tokens", token_matrix), ("routes", route_matrix)):
                counts = split(matrix, rank, sent)
                entry[f"{direction}_{unit}"] = counts["total"]
                entry[f"{direction}_remote_{unit}"] = counts["remote"]
                entry[f"{direction}_cross_node_{unit}"] = counts["cross_node"]
        per_rank.append(entry)
    total_routes = sum(map(sum, route_matrix))
    received = [entry["received_routes"] for entry in per_rank]
    mean = total_routes / world
    return {
        "gpus_per_node": gpus_per_node,
        "route_matrix": route_matrix,
        "token_matrix": token_matrix,
        "per_rank": per_rank,
        "same_gpu_route_fraction": sum(route_matrix[rank][rank] for rank in range(world)) / total_routes,
        "same_node_route_fraction": sum(
            route_matrix[source][destination]
            for source in range(world)
            for destination in range(world)
            if source // gpus_per_node == destination // gpus_per_node
        )
        / total_routes,
        "idle_receivers": sum(count == 0 for count in received),
        "busiest_receiver": max(range(world), key=received.__getitem__),
        "received_routes_max_over_mean": max(received) / mean,
    }
