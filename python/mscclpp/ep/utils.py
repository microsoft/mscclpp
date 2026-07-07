# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal helpers shared by the expert-parallel Python frontend."""

from __future__ import annotations

import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch


def send_bytes(comm: Any, payload: bytes, peer: int, tag: int) -> None:
    comm.send(np.frombuffer(payload, dtype=np.uint8), peer, tag)


def recv_bytes(comm: Any, size: int, peer: int, tag: int) -> bytes:
    payload = np.empty(size, dtype=np.uint8)
    comm.recv(payload, peer, tag)
    return payload.tobytes()


def all_gather_object(comm: Any, obj: Any, tag_base: int) -> List[Any]:
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
        send_bytes(comm, payload, 0, tag_base + 2)
        comm.recv(gathered, 0, tag_base + 3)

    return [pickle.loads(gathered[int(offsets[i]) : int(offsets[i + 1])].tobytes()) for i in range(group_size)]


def broadcast_object(comm: Any, obj: Any, root: int, tag_base: int) -> Any:
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
            send_bytes(comm, payload, peer, tag_base + 1)
        return obj

    payload_size = np.empty(1, dtype=np.int64)
    comm.recv(payload_size, root, tag_base)
    return pickle.loads(recv_bytes(comm, int(payload_size[0]), root, tag_base + 1))


def ptr(tensor: Optional[torch.Tensor]) -> int:
    """``tensor.data_ptr()`` for a tensor, or 0 (== nullptr) for ``None``."""
    return 0 if tensor is None else tensor.data_ptr()


def current_stream_ptr() -> int:
    """Raw pointer of the current CUDA stream (matches the C++ ``cudaStream_t``)."""
    return torch.cuda.current_stream().cuda_stream


def cuda_stream_ptr(stream: Optional[torch.cuda.Stream]) -> int:
    return (stream if stream is not None else torch.cuda.current_stream()).cuda_stream


class DevicePointerArray:
    """Minimal ``__cuda_array_interface__`` holder for a non-owning device pointer."""

    def __init__(self, ptr: int, shape: Tuple[int, ...], typestr: str, owner: Any) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


def bf16_view(ptr: int, num_tokens: int, hidden: int, owner: Any) -> torch.Tensor:
    """View a raw device pointer as a ``[num_tokens, hidden]`` bfloat16 tensor."""
    u16 = torch.as_tensor(DevicePointerArray(ptr, (num_tokens, hidden), "<u2", owner), device="cuda")
    return u16.view(torch.bfloat16)


def requires_dequantization(tensor: torch.Tensor) -> bool:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    return fp8_dtype is not None and tensor.dtype == fp8_dtype


def exclusive_cumsum(counts: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        flat = counts.to(torch.int64).flatten()
        zero = torch.zeros(1, dtype=torch.int64, device=flat.device)
        return torch.cat([zero, torch.cumsum(flat, dim=0)])
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + int(count))
    return torch.tensor(offsets, dtype=torch.int64)


def resolve_expert_placement(
    *,
    num_experts: int,
    world_size: int,
    rank: int,
    num_local_experts: Optional[int],
    local_expert_start: Optional[int],
) -> Tuple[int, int]:
    if num_local_experts is None:
        if num_experts % world_size != 0:
            raise ValueError("num_experts must be divisible by world_size for even contiguous placement")
        num_local_experts = num_experts // world_size
    if num_local_experts * world_size != num_experts:
        raise NotImplementedError("only even contiguous expert placement is currently supported")
    if local_expert_start is None:
        local_expert_start = rank * num_local_experts
    return num_local_experts, local_expert_start
