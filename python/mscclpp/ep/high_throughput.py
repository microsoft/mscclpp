# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""High-throughput backend for the high-level MoE communicator.

This module contains both the high-level HT backend used by
``MoECommunicator`` and the raw-pointer wrapper around the nanobind extension
``mscclpp_ep_cpp.ExpertParallelRuntime`` (the DeepEP-style high-throughput
runtime). The extension exposes a **torch-free, raw-pointer** boundary identical
in spirit to the low-latency ``MoERuntime``: every device tensor crosses the
boundary as an integer ``tensor.data_ptr()`` plus explicit shape/size arguments,
so the module never links libtorch.

Because the C++ side no longer allocates the data-dependent receive buffers,
dynamic recv sizing uses an explicit **two-phase** protocol on the intranode /
internode dispatch path:

1. ``*_notify_dispatch`` runs the size-exchange kernel and returns
   ``num_recv_tokens`` (and, internode, ``num_rdma_recv_tokens``), writing the
   routing prefix matrices into caller-provided tensors.
2. The wrapper allocates the recv output tensors sized by ``num_recv_tokens``
   (or, for the zero-copy direct path, views this rank's recv pool via
   :meth:`resolve_intranode_recv_x_buffer`).
3. ``*_dispatch`` runs the data-movement kernel into those output pointers.

The cached fast path skips the notify phase (``cached_mode=True``) by reusing a
previous dispatch's prefix matrices and recv count.

The low-latency path is served by ``low_latency.py``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ._cpp import Config, DispatchLayout, MoEMode, _cpp
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    RowMajorInternodeDispatchHandle,
    RowMajorInternodeCombineContext,
    RowMajorIntranodeDispatchHandle,
    RowMajorIntranodeCombineContext,
    MoECommunicatorConfig,
    QuantConfig,
)
from .utils import (
    all_gather_object as _all_gather_object,
    bf16_view as _bf16_view,
    broadcast_object as _broadcast_object,
    current_stream_ptr as _stream_ptr,
    exclusive_cumsum,
    ptr as _ptr,
    resolve_expert_placement,
)


class HighThroughputRuntime:
    """Core high-throughput expert-parallel (EP) communication runtime.

    ``comm`` is the ``mscclpp.CommGroup`` used for rank information and
    out-of-band exchange of device ids, CUDA-IPC handles, and the MSCCL++ unique
    id. All dispatch/combine data movement happens through the MSCCL++ runtime.
    """

    #: Default number of SMs reserved for comms kernels. Matches DeepEP.
    num_sms: int = 20

    def __init__(
        self,
        comm: Any,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
    ) -> None:
        if low_latency_mode:
            raise NotImplementedError(
                "HighThroughputRuntime serves the high-throughput path only; use MoERuntime for low latency."
            )
        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.num_qps_per_rank = num_qps_per_rank

        self.runtime = _cpp.ExpertParallelRuntime(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes)

        # Exchange device ids + CUDA-IPC handles + (for RDMA) the MSCCL++ unique id.
        local_device_id = self.runtime.get_local_device_id()
        device_ids = _all_gather_object(comm, local_device_id, 0xE000)

        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = _all_gather_object(comm, local_ipc_handle, 0xE100)

        root_unique_id: Optional[bytes] = None
        if self.rank == 0:
            root_unique_id = self.runtime.create_unique_id()
        root_unique_id = _broadcast_object(comm, root_unique_id, 0, 0xE200)
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
    # Dispatch layout
    # ------------------------------------------------------------------

    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int):
        """Returns ``(num_tokens_per_rank, num_tokens_per_rdma_rank|None,
        num_tokens_per_expert, is_token_in_rank)``."""
        assert topk_idx.dim() == 2 and topk_idx.is_contiguous()
        num_tokens, num_topk = int(topk_idx.size(0)), int(topk_idx.size(1))

        num_tokens_per_rank = torch.empty((self.group_size,), dtype=torch.int32, device="cuda")
        num_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device="cuda")
        is_token_in_rank = torch.empty((num_tokens, self.group_size), dtype=torch.bool, device="cuda")
        num_tokens_per_rdma_rank = None
        if self.is_internode_available():
            num_tokens_per_rdma_rank = torch.empty(
                (self.runtime.get_num_rdma_ranks(),), dtype=torch.int32, device="cuda"
            )

        self.runtime.get_dispatch_layout(
            _ptr(num_tokens_per_rank),
            _ptr(num_tokens_per_rdma_rank),
            _ptr(num_tokens_per_expert),
            _ptr(is_token_in_rank),
            _ptr(topk_idx),
            num_tokens,
            num_topk,
            num_experts,
            _stream_ptr(),
        )
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank

    # ------------------------------------------------------------------
    # Intranode dispatch (two-phase) + combine
    # ------------------------------------------------------------------

    def intranode_dispatch(
        self,
        x: torch.Tensor,
        x_scales: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: Optional[torch.Tensor],
        cached_num_recv_tokens: int,
        cached_rank_prefix_matrix: Optional[torch.Tensor],
        cached_channel_prefix_matrix: Optional[torch.Tensor],
        expert_alignment: int,
        config,
    ):
        """High-throughput intranode dispatch.

        Returns ``(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
        num_recv_tokens_per_expert_list, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
        send_head)`` to mirror the previous DeepEP-style surface."""
        assert x.dim() == 2 and x.is_contiguous()
        cached_mode = cached_rank_prefix_matrix is not None
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        x_element_size = x.element_size()
        num_channels = self.runtime.get_intranode_dispatch_num_channels(x_element_size, config)

        num_topk = int(topk_idx.size(1)) if topk_idx is not None else 0
        num_scales = 0
        if x_scales is not None:
            num_scales = 1 if x_scales.dim() == 1 else int(x_scales.size(1))

        # ----- Phase A: notify (non-cached) or reuse cached layout -----
        if cached_mode:
            num_recv_tokens = cached_num_recv_tokens
            rank_prefix_matrix = cached_rank_prefix_matrix
            channel_prefix_matrix = cached_channel_prefix_matrix
            num_recv_tokens_per_expert_list: List[int] = []
            num_experts = 0
        else:
            assert num_tokens_per_rank is not None and num_tokens_per_expert is not None
            num_experts = int(num_tokens_per_expert.size(0))
            num_local_experts = num_experts // self.group_size
            rank_prefix_matrix = torch.empty((self.group_size, self.group_size), dtype=torch.int32, device="cuda")
            channel_prefix_matrix = torch.empty((self.group_size, num_channels), dtype=torch.int32, device="cuda")
            num_recv_per_expert_host = torch.empty((num_local_experts,), dtype=torch.int32, device="cpu")
            num_recv_tokens = self.runtime.intranode_notify_dispatch(
                _ptr(rank_prefix_matrix),
                _ptr(channel_prefix_matrix),
                _ptr(num_recv_per_expert_host),
                _ptr(num_tokens_per_rank),
                _ptr(num_tokens_per_expert),
                _ptr(is_token_in_rank),
                num_tokens,
                num_experts,
                x_element_size,
                expert_alignment,
                config,
                _stream_ptr(),
            )
            num_recv_tokens_per_expert_list = num_recv_per_expert_host.tolist()

        # ----- Phase B: allocate recv outputs (or view the recv pool) -----
        recv_x = self._alloc_recv_x(num_recv_tokens, hidden, x_element_size, config)
        recv_src_idx = torch.empty((num_recv_tokens,), dtype=torch.int32, device="cuda")
        send_head = torch.empty((num_tokens, self.group_size), dtype=torch.int32, device="cuda")
        recv_channel_prefix_matrix = torch.empty((self.group_size, num_channels), dtype=torch.int32, device="cuda")
        recv_topk_idx = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.int64, device="cuda") if topk_idx is not None else None
        )
        recv_topk_weights = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        recv_x_scales = (
            torch.empty((num_recv_tokens, num_scales), dtype=torch.float32, device="cuda")
            if x_scales is not None
            else None
        )

        self.runtime.intranode_dispatch(
            _ptr(recv_x),
            _ptr(recv_x_scales),
            _ptr(recv_topk_idx),
            _ptr(recv_topk_weights),
            _ptr(recv_src_idx),
            _ptr(send_head),
            _ptr(recv_channel_prefix_matrix),
            _ptr(x),
            _ptr(x_scales),
            _ptr(topk_idx),
            _ptr(topk_weights),
            _ptr(is_token_in_rank),
            _ptr(rank_prefix_matrix),
            _ptr(channel_prefix_matrix),
            num_tokens,
            hidden,
            num_topk,
            num_scales,
            num_experts,
            x_element_size,
            num_recv_tokens,
            cached_mode,
            config,
            _stream_ptr(),
        )
        return (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
        )

    def _alloc_recv_x(self, num_recv_tokens: int, hidden: int, x_element_size: int, config) -> torch.Tensor:
        """Allocate ``recv_x`` or, when the zero-copy direct path is active, view
        this rank's recv pool (so the sender writes hidden straight to its final
        slot and the TMA combine gathers from the same pool)."""
        pool_ptr = self.runtime.resolve_intranode_recv_x_buffer(num_recv_tokens, hidden, x_element_size, config)
        if pool_ptr != 0:
            return _bf16_view(pool_ptr, num_recv_tokens, hidden, owner=self)
        return torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def intranode_combine(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        src_idx: torch.Tensor,
        rank_prefix_matrix: torch.Tensor,
        channel_prefix_matrix: torch.Tensor,
        send_head: torch.Tensor,
        config,
    ):
        """Returns ``(combined_x, combined_topk_weights|None)``."""
        assert x.dim() == 2 and x.is_contiguous()
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        num_recv_tokens = int(send_head.size(0))
        num_topk = int(topk_weights.size(1)) if topk_weights is not None else 0
        ring_num_channels = int(channel_prefix_matrix.size(1))

        combined_x = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        combined_topk_weights = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        self.runtime.intranode_combine(
            _ptr(combined_x),
            _ptr(combined_topk_weights),
            _ptr(x),
            _ptr(topk_weights),
            _ptr(src_idx),
            _ptr(rank_prefix_matrix),
            _ptr(channel_prefix_matrix),
            _ptr(send_head),
            num_tokens,
            num_recv_tokens,
            hidden,
            num_topk,
            x.element_size(),
            ring_num_channels,
            config,
            _stream_ptr(),
        )
        return combined_x, combined_topk_weights

    # ------------------------------------------------------------------
    # Internode dispatch (two-phase) + combine
    # ------------------------------------------------------------------

    def internode_dispatch(
        self,
        x: torch.Tensor,
        x_scales: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        num_tokens_per_rdma_rank: Optional[torch.Tensor],
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: Optional[torch.Tensor],
        cached_num_recv_tokens: int,
        cached_num_rdma_recv_tokens: int,
        cached_rdma_channel_prefix_matrix: Optional[torch.Tensor],
        cached_recv_rdma_rank_prefix_sum: Optional[torch.Tensor],
        cached_gbl_channel_prefix_matrix: Optional[torch.Tensor],
        cached_recv_gbl_rank_prefix_sum: Optional[torch.Tensor],
        expert_alignment: int,
        config,
    ):
        """High-throughput internode (NVLink + RDMA) dispatch.

        Returns ``(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
        num_recv_tokens_per_expert_list, rdma_channel_prefix_matrix,
        gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head, send_nvl_head)``
        to mirror the previous DeepEP-style surface."""
        assert x.dim() == 2 and x.is_contiguous()
        cached_mode = cached_rdma_channel_prefix_matrix is not None
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        x_element_size = x.element_size()
        num_rdma_ranks = self.runtime.get_num_rdma_ranks()
        num_channels = self.runtime.get_internode_dispatch_num_channels(config)

        num_topk = int(topk_idx.size(1)) if topk_idx is not None else 0
        num_scales = 0
        if x_scales is not None:
            num_scales = 1 if x_scales.dim() == 1 else int(x_scales.size(1))

        # ----- Phase A: notify (non-cached) or reuse cached layout -----
        if cached_mode:
            num_recv_tokens = cached_num_recv_tokens
            num_rdma_recv_tokens = cached_num_rdma_recv_tokens
            rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix
            recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum
            gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix
            recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum
            num_recv_tokens_per_expert_list: List[int] = []
            num_experts = 0
        else:
            assert (
                num_tokens_per_rank is not None
                and num_tokens_per_rdma_rank is not None
                and num_tokens_per_expert is not None
            )
            num_experts = int(num_tokens_per_expert.size(0))
            num_local_experts = num_experts // self.group_size
            rdma_channel_prefix_matrix = torch.empty((num_rdma_ranks, num_channels), dtype=torch.int32, device="cuda")
            recv_rdma_rank_prefix_sum = torch.empty((num_rdma_ranks,), dtype=torch.int32, device="cuda")
            gbl_channel_prefix_matrix = torch.empty((self.group_size, num_channels), dtype=torch.int32, device="cuda")
            recv_gbl_rank_prefix_sum = torch.empty((self.group_size,), dtype=torch.int32, device="cuda")
            # num_recv_tokens_per_expert and num_rdma_recv_tokens are written on the host.
            num_recv_per_expert_host = torch.empty((num_local_experts,), dtype=torch.int32, device="cpu")
            num_rdma_recv_host = torch.empty((1,), dtype=torch.int32, device="cpu")
            num_recv_tokens = self.runtime.internode_notify_dispatch(
                _ptr(rdma_channel_prefix_matrix),
                _ptr(recv_rdma_rank_prefix_sum),
                _ptr(gbl_channel_prefix_matrix),
                _ptr(recv_gbl_rank_prefix_sum),
                _ptr(num_recv_per_expert_host),
                _ptr(num_rdma_recv_host),
                _ptr(num_tokens_per_rank),
                _ptr(num_tokens_per_rdma_rank),
                _ptr(num_tokens_per_expert),
                _ptr(is_token_in_rank),
                num_tokens,
                num_experts,
                hidden,
                num_scales,
                num_topk,
                x_element_size,
                expert_alignment,
                config,
                _stream_ptr(),
            )
            num_rdma_recv_tokens = int(num_rdma_recv_host[0].item())
            num_recv_tokens_per_expert_list = num_recv_per_expert_host.tolist()

        # ----- Phase B: allocate recv outputs (or view the recv pool) -----
        recv_x = self._alloc_internode_recv_x(num_recv_tokens, hidden, x_element_size, config, cached_mode)
        recv_topk_idx = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.int64, device="cuda") if topk_idx is not None else None
        )
        recv_topk_weights = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        recv_x_scales = (
            torch.empty((num_recv_tokens, num_scales), dtype=torch.float32, device="cuda")
            if x_scales is not None
            else None
        )
        # The receiver-side metadata / head buffers are only produced (and only
        # needed by combine) on the non-cached forward path.
        if cached_mode:
            recv_src_meta = None
            recv_rdma_channel_prefix_matrix = None
            recv_gbl_channel_prefix_matrix = None
            send_rdma_head = None
            send_nvl_head = None
        else:
            meta_bytes = self.runtime.get_source_meta_bytes()
            num_max_nvl_peers = self.runtime.get_num_max_nvl_peers()
            recv_src_meta = torch.empty((num_recv_tokens, meta_bytes), dtype=torch.uint8, device="cuda")
            recv_rdma_channel_prefix_matrix = torch.empty(
                (num_rdma_ranks, num_channels), dtype=torch.int32, device="cuda"
            )
            recv_gbl_channel_prefix_matrix = torch.empty(
                (self.group_size, num_channels), dtype=torch.int32, device="cuda"
            )
            send_rdma_head = torch.empty((num_tokens, num_rdma_ranks), dtype=torch.int32, device="cuda")
            send_nvl_head = torch.empty((num_rdma_recv_tokens, num_max_nvl_peers), dtype=torch.int32, device="cuda")

        self.runtime.internode_dispatch(
            _ptr(recv_x),
            _ptr(recv_x_scales),
            _ptr(recv_topk_idx),
            _ptr(recv_topk_weights),
            _ptr(recv_src_meta),
            _ptr(recv_rdma_channel_prefix_matrix),
            _ptr(recv_gbl_channel_prefix_matrix),
            _ptr(send_rdma_head),
            _ptr(send_nvl_head),
            _ptr(x),
            _ptr(x_scales),
            _ptr(topk_idx),
            _ptr(topk_weights),
            _ptr(is_token_in_rank),
            _ptr(rdma_channel_prefix_matrix),
            _ptr(recv_rdma_rank_prefix_sum),
            _ptr(gbl_channel_prefix_matrix),
            _ptr(recv_gbl_rank_prefix_sum),
            num_tokens,
            hidden,
            num_topk,
            num_scales,
            num_experts,
            x_element_size,
            num_recv_tokens,
            num_rdma_recv_tokens,
            cached_mode,
            config,
            _stream_ptr(),
        )
        return (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
        )

    def _alloc_internode_recv_x(
        self, num_recv_tokens: int, hidden: int, x_element_size: int, config, cached_mode: bool
    ) -> torch.Tensor:
        """Allocate ``recv_x`` or, on the non-cached direct path, view this rank's
        recv pool (so the cross-GPU forwarder writes hidden into the pool and the
        direct-gather combine reads it back). The pool view is non-cached only,
        matching the ``ep_use_direct`` gate in the C++ runtime."""
        if not cached_mode:
            pool_ptr = self.runtime.resolve_internode_recv_x_buffer(num_recv_tokens, hidden, x_element_size, config)
            if pool_ptr != 0:
                return _bf16_view(pool_ptr, num_recv_tokens, hidden, owner=self)
        return torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def internode_combine(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        src_meta: torch.Tensor,
        is_combined_token_in_rank: torch.Tensor,
        rdma_channel_prefix_matrix: torch.Tensor,
        rdma_rank_prefix_sum: torch.Tensor,
        gbl_channel_prefix_matrix: torch.Tensor,
        combined_rdma_head: torch.Tensor,
        combined_nvl_head: torch.Tensor,
        config,
    ):
        """Returns ``(combined_x, combined_topk_weights|None)``."""
        assert x.dim() == 2 and x.is_contiguous()
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        num_combined_tokens = int(is_combined_token_in_rank.size(0))
        num_topk = int(topk_weights.size(1)) if topk_weights is not None else 0

        combined_x = torch.empty((num_combined_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        combined_topk_weights = (
            torch.empty((num_combined_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        self.runtime.internode_combine(
            _ptr(combined_x),
            _ptr(combined_topk_weights),
            _ptr(x),
            _ptr(topk_weights),
            _ptr(src_meta),
            _ptr(is_combined_token_in_rank),
            _ptr(rdma_channel_prefix_matrix),
            _ptr(rdma_rank_prefix_sum),
            _ptr(gbl_channel_prefix_matrix),
            _ptr(combined_rdma_head),
            _ptr(combined_nvl_head),
            num_tokens,
            num_combined_tokens,
            hidden,
            num_topk,
            x.element_size(),
            config,
            _stream_ptr(),
        )
        return combined_x, combined_topk_weights


class HighThroughputBackend:
    """Backend implementation for ``MoEMode.HIGH_THROUGHPUT``."""

    def __init__(self, config: MoECommunicatorConfig, output_layout: DispatchLayout) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=HIGH_THROUGHPUT requires an mscclpp.CommGroup via comm=")
        if Config is None or not hasattr(_cpp, "ExpertParallelRuntime"):
            raise ImportError(
                "mscclpp_ep_cpp was built without the high-throughput EP backend. "
                "Rebuild with -DMSCCLPP_BUILD_EXT_EP=ON and ensure Config/ExpertParallelRuntime are exported."
            )

        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.mode = MoEMode.HIGH_THROUGHPUT
        self.output_layout = output_layout

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.num_sms = config.num_sms
        self.enable_overlap = config.enable_overlap

        if self.output_layout != DispatchLayout.FLAT:
            raise NotImplementedError("HT mode currently supports only DispatchLayout.FLAT")

        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )

        if config.quant is not None:
            raise NotImplementedError("HT quantized dispatch (scales) is not implemented yet")

        self.expert_alignment = config.expert_alignment
        self._cfg = Config(
            self.num_sms,
            config.nvl_chunked_send,
            config.nvl_chunked_recv,
            config.rdma_chunked_send,
            config.rdma_chunked_recv,
        )
        hidden_bytes = self.hidden_size * torch.tensor([], dtype=torch.bfloat16).element_size()
        num_nvl_bytes = self._cfg.get_nvl_buffer_size_hint(hidden_bytes, self.world_size)
        num_rdma_bytes = self._cfg.get_rdma_buffer_size_hint(hidden_bytes, self.world_size)
        self._is_internode = num_rdma_bytes > 0
        self._runtime = HighThroughputRuntime(
            comm,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )

    def is_available(self) -> bool:
        return self._runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._runtime.is_internode_available()

    def is_internode(self) -> bool:
        return self._is_internode

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
    ) -> tuple[DispatchOutput, DispatchHandle]:
        del output_buffer
        if stream is not None:
            with torch.cuda.stream(stream):
                return self._dispatch(input, topk_ids, weights, quant, previous_handle)
        return self._dispatch(input, topk_ids, weights, quant, previous_handle)

    def _dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        previous_handle: Optional[DispatchHandle],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        self._validate_dispatch_inputs(input, topk_ids, weights, quant)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        cache = getattr(previous_handle, "_dispatch_cache", None) if previous_handle is not None else None
        if cache is not None:
            num_tokens_per_rank = cache["num_tokens_per_rank"]
            num_tokens_per_rdma_rank = cache["num_tokens_per_rdma_rank"]
            num_tokens_per_expert = cache["num_tokens_per_expert"]
            is_token_in_rank = cache["is_token_in_rank"]
        else:
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
            ) = self._runtime.get_dispatch_layout(topk_ids, self.num_experts)

        if self._is_internode:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                _rdma_channel_prefix_matrix,
                _gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                _recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = self._runtime.internode_dispatch(
                input,
                None,
                topk_ids,
                weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                self.expert_alignment,
                self._cfg,
            )
            combine_context = RowMajorInternodeCombineContext(
                recv_topk_weights=recv_topk_weights,
                src_meta=recv_src_meta,
                is_token_in_rank=is_token_in_rank,
                recv_rdma_channel_prefix_matrix=recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum=recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix=recv_gbl_channel_prefix_matrix,
                send_rdma_head=send_rdma_head,
                send_nvl_head=send_nvl_head,
            )
            dispatch_cache = (
                cache
                if cache is not None
                else {
                    "num_tokens_per_rank": num_tokens_per_rank,
                    "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                    "num_tokens_per_expert": num_tokens_per_expert,
                    "is_token_in_rank": is_token_in_rank,
                }
            )
        elif cache is not None:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                _channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
            ) = self._runtime.intranode_dispatch(
                input,
                None,
                None,
                None,
                None,
                is_token_in_rank,
                None,
                cache["num_recv_tokens"],
                cache["rank_prefix_matrix"],
                cache["channel_prefix_matrix"],
                self.expert_alignment,
                self._cfg,
            )
            combine_context = RowMajorIntranodeCombineContext(
                recv_topk_weights=recv_topk_weights,
                src_idx=recv_src_idx,
                rank_prefix_matrix=rank_prefix_matrix,
                recv_channel_prefix_matrix=recv_channel_prefix_matrix,
                send_head=send_head,
            )
            dispatch_cache = cache
        else:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
            ) = self._runtime.intranode_dispatch(
                input,
                None,
                topk_ids,
                weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                None,
                None,
                self.expert_alignment,
                self._cfg,
            )
            combine_context = RowMajorIntranodeCombineContext(
                recv_topk_weights=recv_topk_weights,
                src_idx=recv_src_idx,
                rank_prefix_matrix=rank_prefix_matrix,
                recv_channel_prefix_matrix=recv_channel_prefix_matrix,
                send_head=send_head,
            )
            dispatch_cache = {
                "num_tokens_per_rank": num_tokens_per_rank,
                "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "is_token_in_rank": is_token_in_rank,
                "rank_prefix_matrix": rank_prefix_matrix,
                "channel_prefix_matrix": channel_prefix_matrix,
                "num_recv_tokens": int(recv_x.size(0)),
            }

        output_info = DispatchOutputInfo(
            layout=DispatchLayoutInfo(
                kind=DispatchLayout.FLAT,
                num_tokens_per_expert=num_recv_tokens_per_expert_list,
                offsets=exclusive_cumsum(num_recv_tokens_per_expert_list),
            ),
            quant=None,
        )
        dispatch_out = DispatchOutput(
            tokens=recv_x,
            quant=output_info.quant,
            layout=output_info.layout,
        )
        handle_cls = (
            RowMajorInternodeDispatchHandle
            if isinstance(combine_context, RowMajorInternodeCombineContext)
            else RowMajorIntranodeDispatchHandle
        )
        handle = handle_cls(output_info=output_info, combine_context=combine_context)
        # The torch-free HT runtime orders its work on the caller's CUDA stream
        # (no separate event handle), so there is nothing to attach here.
        handle._event = None  # type: ignore[attr-defined]
        handle._dispatch_cache = dispatch_cache  # type: ignore[attr-defined]
        return dispatch_out, handle

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        if stream is not None:
            with torch.cuda.stream(stream):
                return self._combine(expert_output, handle, out)
        return self._combine(expert_output, handle, out)

    def _combine(
        self, expert_output: torch.Tensor, handle: DispatchHandle, out: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self._validate_combine_inputs(expert_output, handle)
        if isinstance(handle, RowMajorInternodeDispatchHandle):
            context = handle.combine_context
            combined_x, _combined_w = self._runtime.internode_combine(
                expert_output,
                context.recv_topk_weights,
                context.src_meta,
                context.is_token_in_rank,
                context.recv_rdma_channel_prefix_matrix,
                context.recv_rdma_rank_prefix_sum,
                context.recv_gbl_channel_prefix_matrix,
                context.send_rdma_head,
                context.send_nvl_head,
                self._cfg,
            )
        elif isinstance(handle, RowMajorIntranodeDispatchHandle):
            context = handle.combine_context
            combined_x, _combined_w = self._runtime.intranode_combine(
                expert_output,
                context.recv_topk_weights,
                context.src_idx,
                context.rank_prefix_matrix,
                context.recv_channel_prefix_matrix,
                context.send_head,
                self._cfg,
            )
        else:
            raise ValueError("DispatchHandle does not contain row-major combine context")
        if out is not None:
            out.copy_(combined_x)
            return out
        return combined_x

    def _validate_dispatch_inputs(self, input, topk_ids, weights, quant) -> None:
        if quant is not None:
            raise NotImplementedError("HT dispatch does not support quantized input scales yet")
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("HT dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} != configured {self.hidden_size}")
        if input.size(0) > self.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), self.topk):
            raise ValueError("topk_ids shape must be [input.size(0), topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")

    def _validate_combine_inputs(self, expert_output, handle) -> None:
        if not isinstance(handle, (RowMajorIntranodeDispatchHandle, RowMajorInternodeDispatchHandle)):
            raise TypeError("handle must be a DispatchHandle returned by dispatch")
        if expert_output.dim() != 2 or not expert_output.is_contiguous():
            raise ValueError("expert_output must be a contiguous [total_recv_tokens, hidden] tensor")
        if expert_output.size(1) != self.hidden_size:
            raise ValueError(f"expert_output hidden size {expert_output.size(1)} != configured {self.hidden_size}")
        if self._is_internode != isinstance(handle, RowMajorInternodeDispatchHandle):
            raise ValueError("handle transport does not match this communicator")
