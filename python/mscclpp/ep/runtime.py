# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Expert-parallel runtime interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from mscclpp.ep.context import Context
from mscclpp.ep.types import DispatchHandle, DispatchOutput, QuantConfig


class Runtime(ABC):
    """Common interface over one mode-configured C++ runtime."""

    def __init__(self, context: Context, cpp_runtime) -> None:
        self.context = context
        self.cpp_runtime = cpp_runtime

    @staticmethod
    def create(context: Context) -> "Runtime":
        from mscclpp.ep.latency import LatencyContext, LatencyRuntime
        from mscclpp.ep.throughput import ThroughputContext, ThroughputRuntime

        if isinstance(context, LatencyContext):
            return LatencyRuntime(context)
        if isinstance(context, ThroughputContext):
            return ThroughputRuntime(context)
        raise TypeError(f"unsupported EP context: {type(context).__name__}")

    def is_available(self) -> bool:
        return self.cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        return self.cpp_runtime.is_internode_available()

    @abstractmethod
    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        *,
        output_buffer: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
        previous_handle: Optional[DispatchHandle],
        runtime_max_tokens_per_rank: Optional[int],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        raise NotImplementedError
