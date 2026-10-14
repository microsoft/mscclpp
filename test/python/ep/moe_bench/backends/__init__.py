# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixed-buffer communication adapters; optional runtimes are imported on demand.

Adapters expose dispatch() -> (tokens, expert_counts), combine(), metadata, and
close(). Both operations use the current Torch CUDA stream. Routing and all
input/output buffer addresses remain fixed for the adapter's lifetime.
Non-LL-expert-major adapters set adapt_layout, expose combine_buffer, route_ids,
route_weights, route_id_base and optional route_valid_rows, and return None for
expert_counts; the pipeline derives fixed counts from those native assignments.
"""


def preflight_backend(config):
    """Import/check only the selected runtime, without entering collectives."""
    if config["environment"]["communication_backend"] == "nccl":
        from .nccl import preflight

        preflight(config)
    elif config["environment"]["communication_backend"] == "mscclpp":
        import mscclpp.ep as ep

        for enum, setting in ((ep.MoEMode, "mode"), (ep.DispatchLayout, "layout")):
            default = "latency" if setting == "mode" else "expert_major"
            value = config["environment"].get(setting, default)
            if not hasattr(enum, value.upper()):
                raise RuntimeError(f"Installed MSCCL++ EP does not expose {setting}={value}; rebuild the extension")


def create_backend(config, rank, x, ids, weights, recv, expert_output, output):
    backend = config["environment"]["communication_backend"]
    if __package__ == "backends":
        from config import SUPPORTED_LAYOUTS
    else:
        from ..config import SUPPORTED_LAYOUTS
    mode = config["environment"].get("mode", "latency")
    layout = config["environment"].get("layout", "expert_major")
    if backend in SUPPORTED_LAYOUTS and layout not in SUPPORTED_LAYOUTS[backend].get(mode, ()):
        raise ValueError(f"{backend} mode={mode} does not support layout={layout}")
    if backend == "mscclpp":
        from .mscclpp import MscclppBackend

        adapter = MscclppBackend
    elif backend == "nccl":
        from .nccl import NcclBackend

        adapter = NcclBackend
    else:
        raise ValueError(f"Unsupported communication backend: {backend!r}")
    return adapter(config, rank, x, ids, weights, recv, expert_output, output)
