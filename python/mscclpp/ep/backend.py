# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified high-level expert-parallel backend."""

from __future__ import annotations

from typing import Optional

import torch

from ._cpp import DispatchLayout, MoEMode
from .latency import _LatencyMethods
from .overlap import _OverlapMethods
from .runtime import Runtime
from .types import DispatchHandle, DispatchOutput, MoECommunicatorConfig, QuantConfig


class Backend(_LatencyMethods, _OverlapMethods):
    """Own one runtime and expose latency or overlap dispatch/combine."""

    def __init__(self, config: MoECommunicatorConfig, output_layout: DispatchLayout) -> None:
        if config.comm is None:
            raise ValueError("MoECommunicator requires an mscclpp.CommGroup via comm=")
        if config.mode == MoEMode.LATENCY:
            self.runtime = Runtime(
                config.comm,
                MoEMode.LATENCY,
                max_tokens_per_rank=config.max_tokens_per_rank,
                hidden=config.hidden_size,
                num_experts=config.num_experts,
                num_topk=config.topk,
                output_layout=output_layout,
            )
            self._init_latency(config, output_layout)
        else:
            max_hidden_bytes = config.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
            self.runtime = Runtime(
                config.comm,
                MoEMode.OVERLAP,
                max_hidden_bytes=max_hidden_bytes,
                num_sms=config.num_sms,
            )
            self._init_overlap(config, output_layout)
            self.expert_output_buffer = None

    def is_available(self) -> bool:
        """Return whether the selected operation family is available."""
        return self.runtime.is_available()

    def is_internode_available(self) -> bool:
        """Return whether the selected operations support this internode topology."""
        return self.runtime.is_internode_available()

    def is_internode(self) -> bool:
        """Return whether the runtime spans more than one node."""
        return self.runtime.is_internode_available()

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
        if self.mode == MoEMode.LATENCY:
            return self._dispatch_latency(
                input,
                topk_ids,
                weights,
                quant,
                output_buffer=output_buffer,
                stream=stream,
                previous_handle=previous_handle,
                runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            )
        return self._dispatch_overlap(
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
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        """Combine expert output with the selected operation family."""
        if self.mode == MoEMode.LATENCY:
            return self._combine_latency(expert_output, handle, out=out, stream=stream)
        return self._combine_overlap(expert_output, handle, out=out, stream=stream)
