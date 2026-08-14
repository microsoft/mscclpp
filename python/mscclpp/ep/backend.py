# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified high-level expert-parallel backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from mscclpp.ep._cpp import DispatchLayout, MoEMode
from mscclpp.ep.runtime import Runtime
from mscclpp.ep.types import DispatchHandle, DispatchOutput, MoECommunicatorConfig, QuantConfig


class Backend(ABC):
    """Common interface for a mode-specific backend owning one runtime."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def is_available(self) -> bool:
        """Return whether the selected operation family is available."""
        return self.runtime.is_available()

    def is_internode_available(self) -> bool:
        """Return whether the selected operations support this internode topology."""
        return self.runtime.is_internode_available()

    def is_internode(self) -> bool:
        """Return whether the runtime spans more than one node."""
        return self.runtime.is_internode_available()

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
        """Dispatch tokens with the selected operation family."""
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
        """Combine expert output with the selected operation family."""
        raise NotImplementedError


def create_backend(config: MoECommunicatorConfig, output_layout: DispatchLayout) -> Backend:
    """Construct the backend selected by ``config.mode``."""
    if config.mode == MoEMode.LATENCY:
        from mscclpp.ep.latency import LatencyBackend

        return LatencyBackend(config, output_layout)
    if config.mode == MoEMode.OVERLAP:
        from mscclpp.ep.overlap import OverlapBackend

        return OverlapBackend(config, output_layout)
    raise ValueError(f"unsupported MoE mode: {config.mode}")
