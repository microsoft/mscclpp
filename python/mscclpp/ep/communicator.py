# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""High-level MoE dispatch/combine communicator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._cpp import DispatchLayout, MoEMode, OptimizedCombineMode
from .high_throughput import HighThroughputBackend
from .low_latency import LowLatencyBackend
from .types import (
    BlockOverlapConfig,
    CommOverlapConfig,
    CombineContext,
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    ExpertMajorDispatchHandle,
    ExpertMajorCombineContext,
    MoECommunicatorConfig,
    OperationOverlapConfig,
    QuantConfig,
    RowMajorInternodeDispatchHandle,
    RowMajorInternodeCombineContext,
    RowMajorIntranodeDispatchHandle,
    RowMajorIntranodeCombineContext,
)

__all__ = [
    "CommOverlapConfig",
    "BlockOverlapConfig",
    "CombineContext",
    "DispatchHandle",
    "DispatchLayout",
    "DispatchLayoutInfo",
    "DispatchOutput",
    "DispatchOutputInfo",
    "ExpertMajorDispatchHandle",
    "ExpertMajorCombineContext",
    "MoECommunicator",
    "MoECommunicatorConfig",
    "MoEMode",
    "OptimizedCombineMode",
    "OperationOverlapConfig",
    "QuantConfig",
    "RowMajorInternodeDispatchHandle",
    "RowMajorInternodeCombineContext",
    "RowMajorIntranodeDispatchHandle",
    "RowMajorIntranodeCombineContext",
]


class MoECommunicator:
    """High-level MoE communicator for dispatch/combine.

    ``mode=MoEMode.LOW_LATENCY`` selects the LL backend (EXPERT_MAJOR);
    ``mode=MoEMode.HIGH_THROUGHPUT`` selects the HT backend (FLAT).
    """

    def __init__(self, config: Optional[MoECommunicatorConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either MoECommunicatorConfig or keyword arguments, not both")
        if config is None:
            config = MoECommunicatorConfig(**kwargs)

        if config.device is not None:
            torch.cuda.set_device(config.device)

        if not isinstance(config.mode, MoEMode):
            raise TypeError("MoECommunicatorConfig.mode must be a MoEMode")

        _validate_common_config(config)
        self.mode = config.mode
        self.output_layout = _resolve_output_layout(config.output_layout, self.mode)
        if self.mode == MoEMode.LOW_LATENCY:
            self._backend = LowLatencyBackend(config, self.output_layout)
        else:
            self._backend = HighThroughputBackend(config, self.output_layout)
        self._publish_backend_state()

    def _publish_backend_state(self) -> None:
        for name in (
            "comm",
            "rank",
            "world_size",
            "local_rank",
            "device",
            "num_experts",
            "hidden_size",
            "topk",
            "max_tokens_per_rank",
            "num_sms",
            "enable_overlap",
            "num_local_experts",
            "local_expert_start",
        ):
            setattr(self, name, getattr(self._backend, name))

    def is_available(self) -> bool:
        return self._backend.is_available()

    def is_internode_available(self) -> bool:
        return self._backend.is_internode_available()

    def is_internode(self) -> bool:
        return self._backend.is_internode()

    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        quant: Optional[QuantConfig] = None,
        *,
        output_buffer: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        previous_handle: Optional[DispatchHandle] = None,
    ) -> Tuple[DispatchOutput, DispatchHandle]:
        return self._backend.dispatch(
            input,
            topk_ids,
            weights,
            quant,
            output_buffer=output_buffer,
            stream=stream,
            previous_handle=previous_handle,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        return self._backend.combine(expert_output, handle, out=out, stream=stream)

    def dispatch_async(self, *args, **kwargs):
        raise NotImplementedError("dispatch_async is not implemented for MoECommunicator yet")

    def combine_async(self, *args, **kwargs):
        raise NotImplementedError("combine_async is not implemented for MoECommunicator yet")

    def create_overlap_config(
        self, op: str, *, handle: Optional[DispatchHandle] = None, level: str = "op"
    ) -> CommOverlapConfig:
        if op not in ("dispatch", "combine"):
            raise ValueError("op must be 'dispatch' or 'combine'")
        if level != "op":
            raise NotImplementedError("block-level overlap is not implemented yet")
        if op == "combine" and handle is None:
            raise ValueError("combine overlap config requires a DispatchHandle")
        return CommOverlapConfig(operation=OperationOverlapConfig())


def _validate_common_config(config: MoECommunicatorConfig) -> None:
    if config.num_experts <= 0 or config.hidden_size <= 0 or config.topk <= 0 or config.max_tokens_per_rank <= 0:
        raise ValueError("num_experts, hidden_size, topk, and max_tokens_per_rank must be positive")


def _resolve_output_layout(layout: Optional[DispatchLayout], mode: MoEMode) -> DispatchLayout:
    if layout is None:
        return DispatchLayout.EXPERT_MAJOR if mode == MoEMode.LOW_LATENCY else DispatchLayout.FLAT
    if not isinstance(layout, DispatchLayout):
        raise TypeError("MoECommunicatorConfig.output_layout must be a DispatchLayout")
    return layout
