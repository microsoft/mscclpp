# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified low-level expert-parallel runtime wrapper."""

from __future__ import annotations

from typing import Any

from mscclpp.ep._cpp import DispatchLayout, MoEMode, create_moe_runtime


class Runtime:
    """Own one C++ runtime configured for latency or overlap algorithms."""

    def __init__(
        self,
        comm: Any,
        mode: MoEMode,
        *,
        max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_experts: int = 0,
        num_topk: int = 0,
        max_hidden_bytes: int = 0,
        num_sms: int = 20,
        output_layout: DispatchLayout = DispatchLayout.EXPERT_MAJOR,
    ) -> None:
        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.cpp_runtime = create_moe_runtime(
            comm.communicator,
            mode,
            max_tokens_per_rank=max_tokens_per_rank,
            hidden=hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            max_hidden_bytes=max_hidden_bytes,
            num_sms=num_sms,
            output_layout=output_layout,
        )

    def is_available(self) -> bool:
        """Return whether the selected algorithms are available."""
        return self.cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        """Return whether the selected algorithms support this internode topology."""
        return self.cpp_runtime.is_internode_available()
