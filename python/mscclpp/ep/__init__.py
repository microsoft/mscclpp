# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` for migration status and
``python/mscclpp/ep/README.md`` for the high-level API design.

``MoECommunicator`` is the high-level public API. ``mode=MoEMode.LOW_LATENCY``
runs on the ``MoERuntime`` LL backend; ``mode=MoEMode.HIGH_THROUGHPUT`` runs on
the DeepEP-style :class:`Buffer` HT backend (GB200 TMA direct-gather combine +
all-sender dispatch).
"""

from .buffer import Buffer, Config, ExpertParallelRuntime  # noqa: F401
from .communicator import (  # noqa: F401
    CommOverlapConfig,
    DispatchHandle,
    DispatchLayout,
    DispatchOutput,
    MoECommunicator,
    MoECommunicatorConfig,
    MoEMode,
    QuantScales,
)

__all__ = [
    "ExpertParallelRuntime",
    "Buffer",
    "Config",
    "CommOverlapConfig",
    "DispatchHandle",
    "DispatchLayout",
    "DispatchOutput",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoEMode",
    "QuantScales",
]
