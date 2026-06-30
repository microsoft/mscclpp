# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Low-level HT (high-throughput) runtime wrapper for the MSCCL++ EP extension.

This is a thin wrapper around the nanobind extension
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

The low-latency path is served by :class:`mscclpp.ext.ep.MoERuntime`; this
runtime exposes only the HT dispatch/combine methods.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON or install via `pip install` after the build."
    ) from exc

Config = _cpp.Config


# ----------------------------------------------------------------------------
# Raw-pointer helpers (the boundary is now data_ptr()-based, like MoERuntime).
# ----------------------------------------------------------------------------


def _ptr(t: Optional[torch.Tensor]) -> int:
    """``tensor.data_ptr()`` for a tensor, or 0 (== nullptr) for ``None``."""
    return 0 if t is None else t.data_ptr()


def _stream_ptr() -> int:
    """Raw pointer of the current CUDA stream (matches the C++ ``cudaStream_t``)."""
    return torch.cuda.current_stream().cuda_stream


class _DevicePointerArray:
    """Minimal ``__cuda_array_interface__`` holder wrapping an existing device
    pointer (no allocation, no ownership). Used to view this rank's recv pool as
    a tensor for the zero-copy direct dispatch path, mirroring the old
    ``torch::from_blob`` on ``recv_pool_local_ptr_``."""

    def __init__(self, ptr: int, shape: Tuple[int, ...], typestr: str, owner) -> None:
        # ``owner`` keeps the runtime (and therefore the pool allocation) alive
        # for as long as the resulting tensor is referenced.
        self._owner = owner
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


def _bf16_view(ptr: int, num_tokens: int, hidden: int, owner) -> torch.Tensor:
    """View a raw device pointer as a ``[num_tokens, hidden]`` bfloat16 tensor.

    bfloat16 has no ``__cuda_array_interface__`` typestr, so the memory is
    imported as uint16 and reinterpreted with ``.view(torch.bfloat16)``."""
    u16 = torch.as_tensor(_DevicePointerArray(ptr, (num_tokens, hidden), "<u2", owner), device="cuda")
    return u16.view(torch.bfloat16)


class ExpertParallelRuntime:
    """Core high-throughput expert-parallel (EP) communication runtime.

    ``group`` is the ``torch.distributed`` process group used only for the
    out-of-band exchange of device ids, CUDA-IPC handles, and the MSCCL++ unique
    id. All dispatch/combine data movement happens through the MSCCL++ runtime.
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
        if low_latency_mode:
            raise NotImplementedError(
                "ExpertParallelRuntime serves the high-throughput path only; use MoERuntime for low latency."
            )
        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

        self.rank: int = group.rank()
        self.group_size: int = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.num_qps_per_rank = num_qps_per_rank

        self.runtime = _cpp.ExpertParallelRuntime(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes)

        # Exchange device ids + CUDA-IPC handles + (for RDMA) the MSCCL++ unique id.
        device_ids: List[Optional[int]] = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        ipc_handles: List[Optional[bytes]] = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

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

    def internode_dispatch(self, *args, **kwargs):
        raise NotImplementedError(
            "ExpertParallelRuntime internode dispatch is not yet ported to the torch-free pointer API."
        )

    def internode_combine(self, *args, **kwargs):
        raise NotImplementedError(
            "ExpertParallelRuntime internode combine is not yet ported to the torch-free pointer API."
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
    ) -> int:
        return _cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)


# Backward-compatible alias for the former DeepEP-style name.
Buffer = ExpertParallelRuntime
