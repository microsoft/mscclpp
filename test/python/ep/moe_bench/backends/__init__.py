# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixed-buffer communication adapters; optional runtimes are imported on demand.

Adapters expose dispatch() -> (tokens, expert_counts), combine(), metadata, and
close(). Both operations use the current Torch CUDA stream. Routing and all
input/output buffer addresses remain fixed for the adapter's lifetime.
"""


def create_backend(config, rank, x, ids, weights, recv, expert_output, output):
    backend = config["environment"]["communication_backend"]
    if backend == "mscclpp":
        from .mscclpp import MscclppBackend

        adapter = MscclppBackend
    elif backend == "nccl":
        from .nccl import NcclBackend

        adapter = NcclBackend
    else:
        raise ValueError(f"Unsupported communication backend: {backend!r}")
    return adapter(config, rank, x, ids, weights, recv, expert_output, output)
