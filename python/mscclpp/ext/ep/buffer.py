# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Low-level HT (high-throughput) runtime wrapper for the MSCCL++ EP extension.

This is a thin wrapper around the nanobind extension ``mscclpp_ep_cpp.Buffer``
(the DeepEP-style high-throughput runtime). The extension carries
``torch.Tensor`` across the Python boundary as **DLPack capsules**, so this
wrapper converts tensors to capsules on the way in (``to_dlpack``) and rebuilds
tensors from capsules on the way out (``from_dlpack``). The :class:`Buffer`
surface otherwise mirrors ``deep_ep.Buffer`` so existing test harnesses port
with minimal changes.

The low-latency path is served by :class:`mscclpp.ext.ep.MoERuntime`; this
Buffer exposes only the HT dispatch/combine methods.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.dlpack import from_dlpack, to_dlpack

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON or install via `pip install` after the build."
    ) from exc

Config = _cpp.Config
EventHandle = _cpp.EventHandle


def _cap(t: Optional[torch.Tensor]):
    """torch.Tensor -> DLPack capsule for the C++ boundary (None passes through)."""
    return None if t is None else to_dlpack(t)


def _ten(c):
    """DLPack capsule -> torch.Tensor (None passes through)."""
    return None if c is None else from_dlpack(c)


class Buffer:
    """Core high-throughput expert-parallel (EP) communication buffer.

    Parameters mirror ``deep_ep.Buffer``. ``group`` is the ``torch.distributed``
    process group used only for the out-of-band exchange of device ids, CUDA-IPC
    handles, and the MSCCL++ unique id.
    """

    #: Default number of SMs reserved for comms kernels. Matches DeepEP.
    num_sms: int = 20

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
    ) -> None:
        self.rank: int = group.rank()
        self.group_size: int = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank

        self.runtime = _cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode)

        # Exchange device ids + CUDA-IPC handles + (for RDMA) the MSCCL++ unique id.
        device_ids: List[Optional[int]] = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        ipc_handles: List[Optional[bytes]] = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

        root_unique_id: Optional[bytes] = None
        if self.rank == 0:
            root_unique_id = self.runtime.create_unique_id()
        broadcast_list = [root_unique_id]
        dist.broadcast_object_list(broadcast_list, src=0, group=group)
        root_unique_id = broadcast_list[0]
        assert root_unique_id is not None
        self.runtime.connect(root_unique_id)

        ipc_handles_ba = [bytearray(h) if h is not None else None for h in ipc_handles]
        self.runtime.sync(device_ids, ipc_handles_ba, bytearray(root_unique_id))

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.runtime.is_available()

    def is_internode_available(self) -> bool:
        return self.runtime.is_internode_available()

    def get_local_device_id(self) -> int:
        return self.runtime.get_local_device_id()

    def get_num_rdma_ranks(self) -> int:
        return self.runtime.get_num_rdma_ranks()

    def get_rdma_rank(self) -> int:
        return self.runtime.get_rdma_rank()

    def get_root_rdma_rank(self, global_: bool) -> int:
        return self.runtime.get_root_rdma_rank(global_)

    # ------------------------------------------------------------------
    # Layout / dispatch / combine (DLPack-bridged wrappers).
    # ------------------------------------------------------------------

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        r = self.runtime.get_dispatch_layout(
            _cap(topk_idx), num_experts, previous_event, async_finish, allocate_on_comm_stream
        )
        # (num_tokens_per_rank, num_tokens_per_rdma_rank?, num_tokens_per_expert, is_token_in_rank, event)
        return (_ten(r[0]), _ten(r[1]), _ten(r[2]), _ten(r[3]), r[4])

    def intranode_dispatch(
        self,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        cached_num_recv_tokens,
        cached_rank_prefix_matrix,
        cached_channel_prefix_matrix,
        expert_alignment,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        r = self.runtime.intranode_dispatch(
            _cap(x),
            _cap(x_scales),
            _cap(topk_idx),
            _cap(topk_weights),
            _cap(num_tokens_per_rank),
            _cap(is_token_in_rank),
            _cap(num_tokens_per_expert),
            cached_num_recv_tokens,
            _cap(cached_rank_prefix_matrix),
            _cap(cached_channel_prefix_matrix),
            expert_alignment,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
        )
        # recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
        # num_recv_tokens_per_expert_list(list), rank_prefix_matrix, channel_prefix_matrix,
        # recv_channel_prefix_matrix, recv_src_idx, send_head, event
        return (
            _ten(r[0]), _ten(r[1]), _ten(r[2]), _ten(r[3]), r[4],
            _ten(r[5]), _ten(r[6]), _ten(r[7]), _ten(r[8]), _ten(r[9]), r[10],
        )

    def intranode_combine(
        self,
        x,
        topk_weights,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        r = self.runtime.intranode_combine(
            _cap(x),
            _cap(topk_weights),
            _cap(src_idx),
            _cap(rank_prefix_matrix),
            _cap(channel_prefix_matrix),
            _cap(send_head),
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
        )
        return (_ten(r[0]), _ten(r[1]), r[2])  # combined_x, combined_topk_weights, event

    def internode_dispatch(
        self,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        cached_num_recv_tokens,
        cached_num_rdma_recv_tokens,
        cached_rdma_channel_prefix_matrix,
        cached_recv_rdma_rank_prefix_sum,
        cached_gbl_channel_prefix_matrix,
        cached_recv_gbl_rank_prefix_sum,
        expert_alignment,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        r = self.runtime.internode_dispatch(
            _cap(x),
            _cap(x_scales),
            _cap(topk_idx),
            _cap(topk_weights),
            _cap(num_tokens_per_rank),
            _cap(num_tokens_per_rdma_rank),
            _cap(is_token_in_rank),
            _cap(num_tokens_per_expert),
            cached_num_recv_tokens,
            cached_num_rdma_recv_tokens,
            _cap(cached_rdma_channel_prefix_matrix),
            _cap(cached_recv_rdma_rank_prefix_sum),
            _cap(cached_gbl_channel_prefix_matrix),
            _cap(cached_recv_gbl_rank_prefix_sum),
            expert_alignment,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
        )
        # 15-tuple; element 4 is the list, element 14 is the event; rest are tensors/None.
        return (
            _ten(r[0]), _ten(r[1]), _ten(r[2]), _ten(r[3]), r[4],
            _ten(r[5]), _ten(r[6]), _ten(r[7]), _ten(r[8]), _ten(r[9]),
            _ten(r[10]), _ten(r[11]), _ten(r[12]), _ten(r[13]), r[14],
        )

    def internode_combine(
        self,
        x,
        topk_weights,
        src_meta,
        is_combined_token_in_rank,
        rdma_channel_prefix_matrix,
        rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        combined_rdma_head,
        combined_nvl_head,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        r = self.runtime.internode_combine(
            _cap(x),
            _cap(topk_weights),
            _cap(src_meta),
            _cap(is_combined_token_in_rank),
            _cap(rdma_channel_prefix_matrix),
            _cap(rdma_rank_prefix_sum),
            _cap(gbl_channel_prefix_matrix),
            _cap(combined_rdma_head),
            _cap(combined_nvl_head),
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
        )
        return (_ten(r[0]), _ten(r[1]), r[2])  # combined_x, combined_topk_weights, event

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
    ) -> int:
        return _cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
