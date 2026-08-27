# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal helpers shared by the expert-parallel Python frontend."""

from __future__ import annotations

import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from mscclpp.ep._cpp import DispatchDataType, EtpRankOrder
from mscclpp.ep.types import QuantConfig


def resolve_dispatch_data_type(quant: Optional[QuantConfig]) -> DispatchDataType:
    """Resolve dispatch storage type from optional quantization metadata."""
    if quant is None:
        return DispatchDataType.BF16

    quant_format = quant.format
    if quant_format is not None and not isinstance(quant_format, DispatchDataType):
        raise TypeError("quant.format must be a DispatchDataType")
    if quant_format is None:
        raise ValueError("quant.format is required")
    if quant_format != DispatchDataType.FP8_E4M3:
        raise ValueError("unsupported dispatch quantization format")
    if quant.block_scales is not None:
        raise ValueError("communicator quant config must not contain precomputed scales")
    return quant_format


def dispatch_scale_block_size(data_type: DispatchDataType) -> int:
    """Return the hidden-element count represented by one dispatch scale."""
    if data_type == DispatchDataType.FP8_E4M3:
        return 128
    return 0


def dispatch_scale_dtype(data_type: DispatchDataType) -> torch.dtype:
    """Return the scale dtype for a quantized dispatch format."""
    if data_type == DispatchDataType.FP8_E4M3:
        return torch.float32
    raise ValueError("BF16 dispatch does not have block scales")


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


def tensor_from_pointer(
    pointer: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    owner: Any,
) -> Tuple[DevicePointerArray, torch.Tensor]:
    """Create a zero-copy tensor view over runtime-owned CUDA memory."""
    storage_types = {
        torch.bfloat16: "<u2",
        torch.float8_e4m3fn: "|u1",
        torch.int32: "<i4",
        torch.float32: "<f4",
    }
    try:
        typestr = storage_types[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported CUDA pointer view dtype: {dtype}") from exc

    buffer_view = DevicePointerArray(pointer, shape, typestr, owner)
    tensor = torch.as_tensor(buffer_view, device=device)
    if tensor.dtype != dtype:
        tensor = tensor.view(dtype)
    tensor._mscclpp_owner = owner
    tensor._mscclpp_buffer_view = buffer_view
    return buffer_view, tensor


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


def resolve_etp_topology(
    *,
    world_size: int,
    rank: int,
    etp_size: int,
    ep_size: Optional[int] = None,
    order: EtpRankOrder = EtpRankOrder.EP_MAJOR,
) -> Tuple[int, int, int, int]:
    """Resolve ``(ep_size, etp_size, ep_index, tp_index)`` for ``rank``.

    ``etp_size`` ranks share one expert's weights. The default ``EP_MAJOR``
    order numbers ranks as ``rank = ep_index * etp_size + tp_index``.
    """
    if type(etp_size) is not int or etp_size <= 0:
        raise ValueError("etp_size must be a positive int")
    if world_size % etp_size != 0:
        raise ValueError("world_size must be divisible by etp_size")
    derived_ep_size = world_size // etp_size
    if ep_size is not None and ep_size != derived_ep_size:
        raise ValueError(f"ep_size={ep_size} is inconsistent with world_size={world_size} and etp_size={etp_size}")
    if order == EtpRankOrder.EP_MAJOR:
        ep_index, tp_index = divmod(rank, etp_size)
    else:
        tp_index, ep_index = divmod(rank, derived_ep_size)
    return derived_ep_size, etp_size, ep_index, tp_index


def resolve_expert_placement(
    *,
    num_experts: int,
    world_size: int,
    rank: int,
    num_local_experts: Optional[int],
    local_expert_start: Optional[int],
    ep_size: Optional[int] = None,
    ep_index: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve ``(num_local_experts, local_expert_start)`` for this rank.

    Placement is keyed on the expert-parallel group, not the world: with
    ``etp_size > 1`` every rank of an EP group owns the same expert slice.
    """
    group_size = world_size if ep_size is None else ep_size
    group_index = rank if ep_index is None else ep_index
    if num_local_experts is None:
        if num_experts % group_size != 0:
            raise ValueError("num_experts must be divisible by ep_size for even contiguous placement")
        num_local_experts = num_experts // group_size
    if num_local_experts * group_size != num_experts:
        raise NotImplementedError("only even contiguous expert placement is currently supported")
    if local_expert_start is None:
        local_expert_start = group_index * num_local_experts
    return num_local_experts, local_expert_start
