# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compute backend selection with lazy optional dependencies."""


def create_mlp(config, rank, counts, device):
    """Load optional compute dependencies only for the selected backend."""
    if config["environment"]["compute_backend"] == "torch":
        from .torch_reference import ExpertMLP

        return ExpertMLP(config, rank, counts, device)
    if config["environment"]["compute_backend"] != "cutlass_flashinfer":
        raise ValueError("Unsupported compute backend")
    from .flashinfer import FlashInferMLP

    return FlashInferMLP(config, rank, counts, device)
