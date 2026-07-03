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

The low-latency path is served by :class:`mscclpp.ep.MoERuntime`; this
runtime exposes only the HT dispatch/combine methods.
"""

from __future__ import annotations

import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON or install via `pip install` after the build."
    ) from exc

Config = _cpp.Config


def _send_bytes(comm: Any, payload: bytes, peer: int, tag: int) -> None:
    comm.send(np.frombuffer(payload, dtype=np.uint8), peer, tag)


def _recv_bytes(comm: Any, size: int, peer: int, tag: int) -> bytes:
    payload = np.empty(size, dtype=np.uint8)
    comm.recv(payload, peer, tag)
    return payload.tobytes()


def _all_gather_object(comm: Any, obj: Any, tag_base: int) -> List[Any]:
    payload = pickle.dumps(obj)
    rank = comm.my_rank
    group_size = comm.nranks

    local_size = np.array([len(payload)], dtype=np.int64)
    sizes = np.empty(group_size, dtype=np.int64)
    if rank == 0:
        sizes[0] = local_size[0]
        for peer in range(1, group_size):
            comm.recv(sizes[peer : peer + 1], peer, tag_base)
        for peer in range(1, group_size):
            comm.send(sizes, peer, tag_base + 1)
    else:
        comm.send(local_size, 0, tag_base)
        comm.recv(sizes, 0, tag_base + 1)

    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    total_size = int(offsets[-1])
    gathered = np.empty(total_size, dtype=np.uint8)
    start = int(offsets[rank])
    end = int(offsets[rank + 1])
    if rank == 0:
        gathered[start:end] = np.frombuffer(payload, dtype=np.uint8)
        for peer in range(1, group_size):
            peer_start = int(offsets[peer])
            peer_end = int(offsets[peer + 1])
            comm.recv(gathered[peer_start:peer_end], peer, tag_base + 2)
        for peer in range(1, group_size):
            comm.send(gathered, peer, tag_base + 3)
    else:
        _send_bytes(comm, payload, 0, tag_base + 2)
        comm.recv(gathered, 0, tag_base + 3)

    return [pickle.loads(gathered[int(offsets[i]) : int(offsets[i + 1])].tobytes()) for i in range(group_size)]


def _broadcast_object(comm: Any, obj: Any, root: int, tag_base: int) -> Any:
    rank = comm.my_rank
    group_size = comm.nranks
    if rank == root:
        payload = pickle.dumps(obj)
        payload_size = np.array([len(payload)], dtype=np.int64)
        for peer in range(group_size):
            if peer == root:
                continue
            comm.send(payload_size, peer, tag_base)
        for peer in range(group_size):
            if peer == root:
                continue
            _send_bytes(comm, payload, peer, tag_base + 1)
        return obj

    payload_size = np.empty(1, dtype=np.int64)
    comm.recv(payload_size, root, tag_base)
    return pickle.loads(_recv_bytes(comm, int(payload_size[0]), root, tag_base + 1))


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
                "ExpertParallelRuntime serves the high-throughput path only; use MoERuntime for low latency."
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
