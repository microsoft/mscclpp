# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MSCCL++ Expert-Parallel (MoE dispatch/combine) extension.

See ``src/ext/ep/README.md`` in the repository for migration status. The
``Buffer`` class mirrors :class:`deep_ep.Buffer` and supports intranode
(NVLink-only) dispatch/combine as well as internode HT and low-latency
paths.
"""

from .buffer import Buffer, Config, EventHandle  # noqa: F401

__all__ = ["Buffer", "Config", "EventHandle"]
