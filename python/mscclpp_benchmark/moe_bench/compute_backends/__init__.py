# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compute backend selection with lazy optional dependencies."""


def create_mlp(config, rank, counts, device):
    """Load compute dependencies only when the selected backend is constructed."""
    if config["environment"]["compute_backend"] != "cutlass_flashinfer":
        raise ValueError("Unsupported compute backend")
    from .flashinfer import FlashInferMLP

    return FlashInferMLP(config, rank, counts, device)
