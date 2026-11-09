# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixed expert-major BF16 reference compute, without padded GEMMs."""

import torch
import torch.nn.functional as F

if __package__ == "compute_backends":
    from workload import make_expert_weights
else:
    from ..workload import make_expert_weights


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
