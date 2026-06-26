# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` for migration status and
``python/mscclpp/ext/ep/README.md`` for the high-level API design.

* :class:`Buffer` is the low-level DeepEP-style runtime (intranode NVLink,
  internode HT, and low-latency paths).
* :class:`MoECommunicator` is the high-level MoE API. ``mode="ht"``
  (high-throughput, ``DispatchLayout.FLAT``) runs on top of :class:`Buffer`;
  ``mode="ll"`` (low-latency, ``DispatchLayout.EXPERT_MAJOR``) runs on top of
  :class:`MoERuntime`.
"""

from .buffer import Buffer, Config, EventHandle  # noqa: F401
from .communicator import (  # noqa: F401
    CommOverlapConfig,
    DispatchHandle,
    DispatchLayout,
    DispatchOutput,
    MoECommunicator,
    MoECommunicatorConfig,
    MoERuntime,
    QuantScales,
)

__all__ = [
    "Buffer",
    "Config",
    "EventHandle",
    "CommOverlapConfig",
    "DispatchHandle",
    "DispatchLayout",
    "DispatchOutput",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoERuntime",
    "QuantScales",
]
