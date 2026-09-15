# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional PyTorch interface to MSCCL++ expert-parallel dispatch and combine."""

try:
    import torch as _torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise ImportError("mscclpp.ep requires PyTorch; install the optional 'mscclpp[ep]' dependencies.") from exc

from ._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode
from .communicator import MoECommunicator
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    PrepareHandle,
    QuantConfig,
)

__all__ = [
    "CombineMode",
    "DispatchDataType",
    "DispatchHandle",
    "DispatchLayout",
    "DispatchLayoutInfo",
    "DispatchOutput",
    "DispatchOutputInfo",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoEMode",
    "PrepareHandle",
    "QuantConfig",
]
