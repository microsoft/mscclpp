# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""On-device native-row packing and weighted local reduction for fixed routing."""

import torch


class LayoutMLP:
    """Adapt native rows to the shared expert MLP without assuming row order.

    Counts are fixed, but dispatch may permute rows on every call. One stable
    sort by local expert per call orders all routes the way the expert-major
    slots expect, so a constant number of kernels runs regardless of how many
    local experts there are. All metadata processing stays on device and
    graph-safe. The native combine consumes BF16 rank partials.

    MLPs with ``compute_routed`` (FlashInfer) gather routes directly and apply
    routing weights in their final GEMM; others use the capacity-sized
    expert-major buffers and an FP32 reduction here.
    """

    def __init__(self, mlp, counts, recv, expert_output, backend):
        self.mlp, self.counts = mlp, list(counts)
        self.recv, self.expert_output, self.backend = recv, expert_output, backend
        self.routed = hasattr(mlp, "compute_routed")
        if self.routed:
            timed = "native-route expert sort, " + mlp.routed_timed_operations
            application = "fc2_fused_finalize_bf16_atomic"
        else:
            timed = (
                "native-route expert sort/pack, "
                + mlp.metadata["mlp_timed_operations"]
                + ", FP32 routing-weight multiplication/local reduction, BF16 native-row write"
            )
            application = "local_mlp_adapter_once"
        self.metadata = {**mlp.metadata, "mlp_timed_operations": timed, "routing_weight_application": application}
        self.total = sum(self.counts)
        if not self.routed:
            rows, hidden = backend.combine_buffer.reshape(-1, recv.shape[-1]).shape
            self.accumulator = torch.empty((rows, hidden), device=recv.device, dtype=torch.float32)
            capacity = recv.shape[1]
            slots = [expert * capacity + row for expert, count in enumerate(self.counts) for row in range(count)]
            self.slots = torch.tensor(slots, dtype=torch.int64, device=recv.device)

    @torch.inference_mode()
    def __call__(self, tokens, out):
        if not self.total:
            self.backend.combine_buffer.zero_()
            return out
        ids = local_route_ids(self.backend).reshape(-1)
        topk = self.backend.route_ids.shape[-1]
        hidden = tokens.shape[-1]
        native = tokens.reshape(-1, hidden)
        experts = len(self.counts)
        # Invalid routes sort last; valid routes are grouped by expert in route
        # order. Narrow keys sort fastest.
        key_dtype = torch.uint8 if experts < 255 else torch.int32
        keys = torch.where((ids >= 0) & (ids < experts), ids, experts).to(key_dtype)
        routes = torch.argsort(keys, stable=True)[: self.total]
        if self.routed:
            weights = self.backend.route_weights.reshape(-1, topk)
            self.mlp.compute_routed(native, routes, weights, self.backend.combine_buffer.view(-1, hidden))
            return out
        rows = routes // topk
        recv, expert_output = self.recv.view(-1, hidden), self.expert_output.view(-1, hidden)
        recv.index_copy_(0, self.slots, native.index_select(0, rows))
        self.mlp(self.recv, self.expert_output)
        values = expert_output.index_select(0, self.slots).float()
        values.mul_(self.backend.route_weights.reshape(-1).index_select(0, routes).unsqueeze(1))
        self.accumulator.zero_()
        self.accumulator.scatter_add_(0, rows.unsqueeze(1).expand(-1, hidden), values)
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
