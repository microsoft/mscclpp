# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` in the repository for migration status. The
``Buffer`` class mirrors :class:`deep_ep.Buffer` and supports intranode
(NVLink-only) dispatch/combine as well as internode HT and low-latency
paths.
"""

from .buffer import (  # noqa: F401
    Buffer,
    CommOverlapConfig,
    Config,
    DispatchHandle,
    DispatchOutput,
    EventHandle,
    MoECommunicator,
    MoECommunicatorConfig,
    QuantScales,
)

__all__ = [
    "Buffer",
    "CommOverlapConfig",
    "Config",
    "DispatchHandle",
    "DispatchOutput",
    "EventHandle",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "QuantScales",
]
