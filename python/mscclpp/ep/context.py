# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Mode-specific context for the high-level expert-parallel communicator."""

from __future__ import annotations

from mscclpp.ep._cpp import MoEMode
from mscclpp.ep.types import MoECommunicatorConfig


class Context:
    """Persistent mode-specific configuration, buffers, and metadata."""

    initialized: bool


def create_context(config: MoECommunicatorConfig) -> Context:
    """Construct the context selected by ``config.mode``."""
    if config.mode == MoEMode.LATENCY:
        from mscclpp.ep.latency import LatencyContext

        return LatencyContext(config)
    if config.mode == MoEMode.THROUGHPUT:
        from mscclpp.ep.throughput import ThroughputContext

        return ThroughputContext(config)
    raise ValueError(f"unsupported MoE mode: {config.mode}")
