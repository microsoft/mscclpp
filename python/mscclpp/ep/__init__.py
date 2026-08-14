# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ Expert-Parallel


`MoECommunicator` is the public API. `mode=MoEMode.LATENCY` selects
latency-optimized algorithms; `mode=MoEMode.OVERLAP` selects bounded-resource
token-major algorithms.
"""

from mscclpp.ep.communicator import (
    BlockOverlapConfig,
    CommOverlapConfig,
    CombineMode,
    DispatchHandle,
    DispatchDataType,
    DispatchLayout,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicator,
    MoECommunicatorConfig,
    MoEMode,
    OperationOverlapConfig,
    QuantConfig,
)

__all__ = [
    "BlockOverlapConfig",
    "CommOverlapConfig",
    "CombineMode",
    "DispatchHandle",
    "DispatchDataType",
    "DispatchLayout",
    "DispatchLayoutInfo",
    "DispatchOutput",
    "DispatchOutputInfo",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoEMode",
    "OperationOverlapConfig",
    "QuantConfig",
]
