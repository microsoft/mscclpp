# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""High-level MoE dispatch/combine communicator."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from mscclpp.ep._cpp import CombineMode, DispatchDataType, DispatchLayout, MoEMode
from mscclpp.ep.context import create_context
from mscclpp.ep.runtime import Runtime
from mscclpp.ep.types import (
    BlockOverlapConfig,
    OverlapConfig,
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    OperationOverlapConfig,
    QuantConfig,
)

__all__ = [
    "OverlapConfig",
    "BlockOverlapConfig",
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


class MoECommunicator:
    """High-level MoE communicator for dispatch/combine.

    `mode=MoEMode.LATENCY` selects the latency algorithms (EXPERT_MAJOR by
    default); `mode=MoEMode.THROUGHPUT` selects bounded-resource throughput
    algorithms (TOKEN_MAJOR by default, with RANK_MAJOR available explicitly).
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
        self._context = create_context(config)
        self._runtime = Runtime.create(self._context)

    @property
    def comm(self) -> Any:
        return self._context.comm

    @property
    def rank(self) -> int:
        return self._context.rank

    @property
    def world_size(self) -> int:
        return self._context.world_size

    @property
    def local_rank(self) -> int:
        return self._context.local_rank

    @property
    def device(self) -> torch.device:
        return self._context.device

    @property
    def mode(self) -> Any:
        return self._context.mode

    @property
    def output_layout(self) -> Any:
        return self._context.output_layout

    @property
    def num_experts(self) -> int:
        return self._context.num_experts

    @property
    def hidden_size(self) -> int:
        return self._context.hidden_size

    @property
    def topk(self) -> int:
        return self._context.topk

    @property
    def max_tokens_per_rank(self) -> int:
        return self._context.max_tokens_per_rank

    @property
    def num_blocks(self) -> int:
        return self._context.num_blocks

    @property
    def enable_overlap(self) -> bool:
        return self._context.enable_overlap

    @property
    def num_local_experts(self) -> int:
        return self._context.num_local_experts

    @property
    def local_expert_start(self) -> int:
        return self._context.local_expert_start

    def is_available(self) -> bool:
        return self._runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._runtime.is_internode_available()

    def is_internode(self) -> bool:
        return self._runtime.is_internode_available()

    def initialize(self) -> None:
        """Collectively initialize deferred communication resources."""
        self._runtime.initialize()

    def is_initialized(self) -> bool:
        """Return whether deferred communication resources are initialized."""
        return self._runtime.is_initialized()

    def get_dispatch_output_buffer(self) -> torch.Tensor:
        """Return the runtime-owned buffer that may be passed to dispatch."""
        return self._runtime.get_dispatch_output_buffer()

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
        runtime_max_tokens_per_rank: Optional[int] = None,
    ) -> Tuple[DispatchOutput, DispatchHandle]:
        return self._runtime.dispatch(
            input,
            topk_ids,
            weights,
            quant,
            output_buffer=output_buffer,
            stream=stream,
            previous_handle=previous_handle,
            runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        return self._runtime.combine(expert_output, handle, out=out, stream=stream)

    def dispatch_async(self, *args, **kwargs):
        raise NotImplementedError("dispatch_async is not implemented for MoECommunicator yet")

    def combine_async(self, *args, **kwargs):
        raise NotImplementedError("combine_async is not implemented for MoECommunicator yet")

    def create_overlap_config(
        self, op: str, *, handle: Optional[DispatchHandle] = None, level: str = "op"
    ) -> OverlapConfig:
        if op not in ("dispatch", "combine"):
            raise ValueError("op must be 'dispatch' or 'combine'")
        if level != "op":
            raise NotImplementedError("block-level overlap is not implemented yet")
        if op == "combine" and handle is None:
            raise ValueError("combine overlap config requires a DispatchHandle")
        return OverlapConfig(operation=OperationOverlapConfig())


def _validate_common_config(config: MoECommunicatorConfig) -> None:
    if config.num_experts <= 0 or config.hidden_size <= 0 or config.topk <= 0 or config.max_tokens_per_rank <= 0:
        raise ValueError("num_experts, hidden_size, topk, and max_tokens_per_rank must be positive")
    if config.output_layout is not None and not isinstance(config.output_layout, DispatchLayout):
        raise TypeError("MoECommunicatorConfig.output_layout must be a DispatchLayout")
