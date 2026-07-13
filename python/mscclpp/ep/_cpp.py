# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared loader for the MSCCL++ expert-parallel Python extension."""

from __future__ import annotations

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON or install with `pip install .[ep]`."
    ) from exc

DispatchLayout = _cpp.DispatchLayout
MoEMode = _cpp.MoEMode
OptimizedCombineMode = _cpp.OptimizedCombineMode
Config = getattr(_cpp, "Config", None)


def get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int, num_topk: int
) -> int:
    return _cpp.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts, num_topk
    )
