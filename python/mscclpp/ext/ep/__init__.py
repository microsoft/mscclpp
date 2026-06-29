# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` in the repository for migration status.
``MoECommunicator`` is the high-level public API.
"""

from .communicator import (  # noqa: F401
    CommOverlapConfig,
    DispatchHandle,
    DispatchOutput,
    DispatchLayout,
    MoEMode,
    MoECommunicator,
    MoECommunicatorConfig,
    QuantScales,
)

__all__ = [
    "CommOverlapConfig",
    "DispatchHandle",
    "DispatchOutput",
    "DispatchLayout",
    "MoEMode",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "QuantScales",
]
