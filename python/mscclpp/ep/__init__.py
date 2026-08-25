# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ Expert-Parallel


``MoECommunicator`` is the public API. ``mode=MoEMode.LOW_LATENCY`` runs on the
LL backend; ``mode=MoEMode.HIGH_THROUGHPUT`` runs on the HT backend (GB200 TMA
direct-gather combine + all-sender dispatch).
"""

from ._cpp import FP8_DEEPGEMM_ABI, FP8_DEEPGEMM_SCALE_BLOCK_SIZE  # noqa: F401
from .communicator import (  # noqa: F401
    BlockOverlapConfig,
    CommOverlapConfig,
    CombineContext,
    CombineMode,
    DispatchHandle,
    DispatchDataType,
    DispatchLayout,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    ExpertMajorDispatchHandle,
    ExpertMajorCombineContext,
    HighThroughputDispatchHandle,
    HighThroughputCombineContext,
    MoECommunicator,
    MoECommunicatorConfig,
    MoEMode,
    OperationOverlapConfig,
    QuantConfig,
    RankMajorDispatchHandle,
    RankMajorCombineContext,
)

__all__ = [
    "FP8_DEEPGEMM_ABI",
    "FP8_DEEPGEMM_SCALE_BLOCK_SIZE",
    "BlockOverlapConfig",
    "CommOverlapConfig",
    "CombineContext",
    "CombineMode",
    "DispatchHandle",
    "DispatchDataType",
    "DispatchLayout",
    "DispatchLayoutInfo",
    "DispatchOutput",
    "DispatchOutputInfo",
    "ExpertMajorDispatchHandle",
    "ExpertMajorCombineContext",
    "HighThroughputDispatchHandle",
    "HighThroughputCombineContext",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoEMode",
    "OperationOverlapConfig",
    "QuantConfig",
    "RankMajorDispatchHandle",
    "RankMajorCombineContext",
]
