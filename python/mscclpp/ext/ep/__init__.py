# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` in the repository for migration status.
``MoECommunicator`` is the high-level API, and ``ExpertParallelRuntime`` is
the low-level runtime wrapper for advanced HT/intranode paths.
"""

from .communicator import (  # noqa: F401
    CommOverlapConfig,
    Config,
    DispatchHandle,
    DispatchOutput,
    EventHandle,
    ExpertParallelRuntime,
    MoECommunicator,
    MoECommunicatorConfig,
    QuantScales,
)

__all__ = [
    "CommOverlapConfig",
    "Config",
    "DispatchHandle",
    "DispatchOutput",
    "EventHandle",
    "ExpertParallelRuntime",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "QuantScales",
]
