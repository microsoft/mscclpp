# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""On-device native-row packing and weighted local reduction for fixed routing."""

import torch


class LayoutMLP:
    """Adapt native rows to the shared expert MLP without assuming row order.

    Counts are fixed, but dispatch may permute rows on every call. Fixed-size
    argsort selections keep all metadata processing on device and graph-safe.
    The native combine consumes BF16 rank partials (or weighted expert rows).
    """

    def __init__(self, mlp, counts, recv, expert_output, backend):
        self.mlp, self.counts = mlp, list(counts)
        self.recv, self.expert_output, self.backend = recv, expert_output, backend
        self.metadata = {
            **mlp.metadata,
            "mlp_timed_operations": "native-row selection/pack, "
            + mlp.metadata["mlp_timed_operations"]
            + ", FP32 routing-weight multiplication/local reduction, BF16 native-row write",
            "routing_weight_application": "local_mlp_adapter_once",
        }
        rows, hidden = backend.combine_buffer.reshape(-1, recv.shape[-1]).shape
        self.accumulator = torch.empty((rows, hidden), device=recv.device, dtype=torch.float32)

    @torch.inference_mode()
    def __call__(self, tokens, out):
        ids = local_route_ids(self.backend).reshape(-1)
        weights = self.backend.route_weights.reshape(-1)
        topk = self.backend.route_ids.shape[-1]
        native = tokens.reshape(-1, tokens.shape[-1])
        selections = []
        for expert, count in enumerate(self.counts):
            if not count:
                selections.append(None)
                continue
            indices = torch.argsort((ids != expert).to(torch.int32), stable=True)[:count]
            rows = indices // topk
            self.recv[expert, :count].copy_(native.index_select(0, rows))
            selections.append((indices, rows))
        self.mlp(self.recv, self.expert_output)
        self.accumulator.zero_()
        for expert, selection in enumerate(selections):
            if selection is None:
                continue
            indices, rows = selection
            values = self.expert_output[expert, : self.counts[expert]].float()
            values.mul_(weights.index_select(0, indices).unsqueeze(1))
            self.accumulator.index_add_(0, rows, values)
        self.backend.combine_buffer.reshape_as(self.accumulator).copy_(self.accumulator)
        return out


def assignment_counts(ids, local_experts):
    """Count valid local assignments, ignoring native padding/sentinel IDs."""
    ids = ids.reshape(-1)
    return torch.bincount(ids[(ids >= 0) & (ids < local_experts)].long(), minlength=local_experts)


def local_route_ids(backend):
    """Normalize IDs and ignore undefined metadata in unused native rank zones."""
    ids = backend.route_ids - backend.route_id_base
    valid = getattr(backend, "route_valid_rows", None)
    if valid is not None:
        ids = ids.reshape(valid.numel(), -1).masked_fill(~valid[:, None], -1)
    return ids
