# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Python frontend for the MSCCL++ Expert-Parallel extension.

This is a thin wrapper around the pybind11 extension ``mscclpp_ep_cpp``.
The shape of :class:`Buffer` mirrors :class:`deep_ep.Buffer` so existing
DeepEP users can port with minimal changes.

Current status (see ``src/ext/ep/README.md``):

* Intranode (NVLink-only) dispatch and combine are fully ported.
* ``get_dispatch_layout`` is ported.
* Internode HT and low-latency methods raise from C++ — they still need
  the NVSHMEM/IBGDA -> MSCCL++ PortChannel migration.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON (and ensure PyTorch's CMake prefix is on "
        "CMAKE_PREFIX_PATH) or install via `pip install` after the build."
    ) from exc

Config = _cpp.Config
EventHandle = _cpp.EventHandle


class Buffer:
    """Core expert-parallel (EP) communication buffer.

    Parameters
    ----------
    group:
        The ``torch.distributed`` process group. Used only for out-of-band
        exchange of IPC handles and the MSCCL++ unique id.
    num_nvl_bytes:
        Size of the NVLink-accessible scratch buffer (shared via CUDA IPC).
    num_rdma_bytes:
        Size of the RDMA scratch buffer. Must be 0 until internode/LL
        support is landed.
    low_latency_mode:
        Reserved — must be ``False`` until the LL path is ported.
    num_qps_per_rank:
        Ignored for intranode mode.
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
                "mscclpp.ext.ep.Buffer: low-latency mode is not yet ported. "
                "Set low_latency_mode=False. See src/ext/ep/README.md for the "
                "migration plan."
            )

        self.rank: int = group.rank()
        self.group_size: int = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank

        self.runtime = _cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode)

        # Exchange device IDs + IPC handles + (for RDMA) the MSCCL++ unique id.
        device_ids: List[Optional[int]] = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        ipc_handles: List[Optional[bytes]] = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        root_unique_id: Optional[bytes] = None
        # RDMA path is guarded above; still plumb the unique-id exchange so
        # the code is ready to turn on once internode lands.
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            if num_qps_per_rank <= 0:
                raise ValueError("num_qps_per_rank must be > 0 for RDMA")

            if self.rank == 0:
                unique_id = self.runtime.create_unique_id()
                root_unique_id = unique_id.bytes()
            broadcast_list = [root_unique_id]
            dist.broadcast_object_list(broadcast_list, src=0, group=group)
            root_unique_id = broadcast_list[0]
            assert root_unique_id is not None
            self.runtime.connect(_cpp.UniqueId.from_bytes(root_unique_id))

        self.runtime.sync(device_ids, ipc_handles, root_unique_id)

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
    # Layout / dispatch / combine (thin pass-through wrappers).
    # Signatures mirror deep_ep.Buffer so existing test harnesses can reuse.
    # ------------------------------------------------------------------

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        return self.runtime.get_dispatch_layout(
            topk_idx, num_experts, previous_event, async_finish, allocate_on_comm_stream
        )

    def intranode_dispatch(self, *args, **kwargs):
        return self.runtime.intranode_dispatch(*args, **kwargs)

    def intranode_combine(self, *args, **kwargs):
        return self.runtime.intranode_combine(*args, **kwargs)

    def internode_dispatch(self, *args, **kwargs):
        return self.runtime.internode_dispatch(*args, **kwargs)

    def internode_combine(self, *args, **kwargs):
        return self.runtime.internode_combine(*args, **kwargs)

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        self.runtime.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

    def low_latency_dispatch(self, *args, **kwargs):
        return self.runtime.low_latency_dispatch(*args, **kwargs)

    def low_latency_combine(self, *args, **kwargs):
        return self.runtime.low_latency_combine(*args, **kwargs)

    def get_next_low_latency_combine_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int):
        return self.runtime.get_next_low_latency_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

    def get_local_buffer_tensor(self, dtype: torch.dtype, offset: int = 0, use_rdma_buffer: bool = False) -> torch.Tensor:
        return self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
    ) -> int:
        return _cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
