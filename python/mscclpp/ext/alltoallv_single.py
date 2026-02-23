# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
PyTorch-compatible all_to_all_single API using mscclpp optimized kernels.

This module provides:
- MscclppAlltoAllV: A class to manage mscclpp alltoallv state
- all_to_all_single: Drop-in replacement for torch.distributed.all_to_all_single

Uses optimized C++ kernels (alltoallvKernel, alltoallvRingKernel, alltoallvPipelinedKernel)
via the NativeAlgorithm framework with size-adaptive algorithm selection.
"""

from __future__ import annotations
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple
from mscclpp._mscclpp import (
    Communicator,
    TcpBootstrap,
    DataType,
    ReduceOp,
)
from mscclpp.ext.algorithm_collection_builder import AlgorithmCollectionBuilder

__all__ = ["MscclppAlltoAllV", "all_to_all_single"]


def _torch_dtype_to_mscclpp(dtype: torch.dtype) -> DataType:
    """Convert PyTorch dtype to mscclpp DataType."""
    if dtype == torch.float32:
        return DataType.float32
    elif dtype == torch.float16:
        return DataType.float16
    elif dtype == torch.bfloat16:
        return DataType.bfloat16
    elif dtype == torch.int32:
        return DataType.int32
    elif dtype == torch.int64:
        return DataType.int64
    elif dtype == torch.uint8:
        return DataType.uint8
    elif dtype == torch.float64:
        return DataType.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _dtype_size(dtype: torch.dtype) -> int:
    """Get byte size for dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class MscclppAlltoAllV:
    """
    Manages mscclpp state for alltoallv operations.
    
    Uses optimized C++ kernels from alltoallv_fullmesh.cu with size-adaptive selection:
    - Small messages (<1MB): alltoallvKernel (lower latency)
    - Large messages + small world (<=16): alltoallvPipelinedKernel 
    - Large messages + large world (>16): alltoallvRingKernel (avoids congestion)
    
    Example:
        mscclpp_alltoallv = MscclppAlltoAllV(
            rank=rank, world_size=world_size,
            ip_port="10.0.0.1:50000"
        )
        # or use existing communicator:
        # mscclpp_alltoallv = MscclppAlltoAllV(communicator=comm)
        
        # Later:
        output = mscclpp_alltoallv.all_to_all_single(
            input_tensor,
            output_split_sizes=[1024, 2048, ...],  # per-rank sizes
            input_split_sizes=[1024, 2048, ...]
        )
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        ip_port: Optional[str] = None,
        communicator: Optional[Communicator] = None,
        scratch_buffer_size: int = 256 * 1024 * 1024,  # 256MB default
    ):
        """
        Initialize MscclppAlltoAllV.
        
        Args:
            rank: Local rank (required if communicator not provided)
            world_size: Total number of ranks (required if communicator not provided)
            ip_port: IP:port for bootstrap (required if communicator not provided)
            communicator: Existing mscclpp Communicator (alternative to rank/world_size/ip_port)
            scratch_buffer_size: Size of scratch buffer in bytes
        """
        if communicator is not None:
            self._comm = communicator
            self._rank = self._comm.bootstrap().get_rank()
            self._world_size = self._comm.bootstrap().get_n_ranks()
            self._owns_comm = False
        else:
            if rank is None or world_size is None or ip_port is None:
                raise ValueError("Must provide either communicator or (rank, world_size, ip_port)")
            
            self._rank = rank
            self._world_size = world_size
            
            # Create bootstrap
            bootstrap = TcpBootstrap(rank, world_size)
            if rank == 0:
                unique_id = bootstrap.create_unique_id()
                # Broadcast unique_id to other ranks via torch.distributed
                id_tensor = torch.tensor(list(unique_id.encode()), dtype=torch.uint8).cuda()
            else:
                id_tensor = torch.zeros(128, dtype=torch.uint8).cuda()
            
            dist.broadcast(id_tensor, src=0)
            unique_id = bytes(id_tensor.cpu().tolist()).decode().rstrip('\x00')
            
            bootstrap.initialize(unique_id)
            self._comm = Communicator(bootstrap)
            self._owns_comm = True
        
        # Allocate scratch buffer
        self._scratch_buffer = torch.zeros(scratch_buffer_size, dtype=torch.uint8, device='cuda')
        self._scratch_ptr = self._scratch_buffer.data_ptr()
        self._scratch_size = scratch_buffer_size
        
        # Build algorithm collection with default algorithms including alltoallv
        builder = AlgorithmCollectionBuilder()
        self._algo_collection = builder.build_default_algorithms(
            self._scratch_ptr, 
            self._scratch_size, 
            self._rank
        )
        
        # Get the alltoallv algorithm
        alltoallv_algos = self._algo_collection.get_by_collective("alltoallv")
        if not alltoallv_algos:
            raise RuntimeError("No alltoallv algorithm found. Make sure mscclpp is built correctly.")
        self._algo = alltoallv_algos[0]
        
        # Pre-allocate count/displacement buffers on GPU (reused across calls)
        # Using int64 (8 bytes) instead of size_t for safety
        self._d_send_counts = torch.zeros(self._world_size, dtype=torch.int64, device='cuda')
        self._d_send_displs = torch.zeros(self._world_size, dtype=torch.int64, device='cuda')
        self._d_recv_counts = torch.zeros(self._world_size, dtype=torch.int64, device='cuda')
        self._d_recv_displs = torch.zeros(self._world_size, dtype=torch.int64, device='cuda')

    @property
    def rank(self) -> int:
        return self._rank
    
    @property 
    def world_size(self) -> int:
        return self._world_size

    def all_to_all_single(
        self,
        input: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
        output: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        Perform all-to-all exchange with variable-sized chunks.
        
        Compatible with torch.distributed.all_to_all_single signature.
        
        Args:
            input: Input tensor (contiguous, CUDA)
            output_split_sizes: List of sizes to receive from each rank (in elements)
            input_split_sizes: List of sizes to send to each rank (in elements) 
            output: Pre-allocated output tensor (optional)
            stream: CUDA stream (optional, uses current stream if not specified)
            
        Returns:
            Output tensor with received data
        """
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("Input must be a contiguous CUDA tensor")
        
        dtype = input.dtype
        elem_size = _dtype_size(dtype)
        world_size = self._world_size
        
        # Handle split sizes
        if input_split_sizes is None:
            # Equal split
            assert input.numel() % world_size == 0
            chunk_size = input.numel() // world_size
            input_split_sizes = [chunk_size] * world_size
        
        if output_split_sizes is None:
            # All-to-all uniform: send and recv same sizes
            output_split_sizes = input_split_sizes.copy()
        
        # Calculate total output size and allocate if needed
        total_output = sum(output_split_sizes)
        if output is None:
            output = torch.empty(total_output, dtype=dtype, device='cuda')
        elif output.numel() < total_output:
            raise ValueError(f"Output tensor too small: {output.numel()} < {total_output}")
        
        # Calculate displacements
        send_displs = [0]
        for i in range(world_size - 1):
            send_displs.append(send_displs[-1] + input_split_sizes[i])
        
        recv_displs = [0]
        for i in range(world_size - 1):
            recv_displs.append(recv_displs[-1] + output_split_sizes[i])
        
        # Convert to byte sizes/offsets for the kernel
        send_counts_bytes = [s * elem_size for s in input_split_sizes]
        send_displs_bytes = [d * elem_size for d in send_displs]
        recv_counts_bytes = [s * elem_size for s in output_split_sizes]
        recv_displs_bytes = [d * elem_size for d in recv_displs]
        
        # Copy to GPU
        self._d_send_counts.copy_(torch.tensor(send_counts_bytes, dtype=torch.int64))
        self._d_send_displs.copy_(torch.tensor(send_displs_bytes, dtype=torch.int64))
        self._d_recv_counts.copy_(torch.tensor(recv_counts_bytes, dtype=torch.int64))
        self._d_recv_displs.copy_(torch.tensor(recv_displs_bytes, dtype=torch.int64))
        
        # Get stream
        if stream is None:
            stream = torch.cuda.current_stream()
        cuda_stream = stream.cuda_stream
        
        # Build extras dict with GPU pointers
        extras = {
            "sendCounts": self._d_send_counts.data_ptr(),
            "sendDispls": self._d_send_displs.data_ptr(),
            "recvCounts": self._d_recv_counts.data_ptr(),
            "recvDispls": self._d_recv_displs.data_ptr(),
        }
        
        input_size = sum(send_counts_bytes)
        output_size = sum(recv_counts_bytes)
        
        # Execute the optimized kernel
        result = self._algo.execute(
            self._comm,
            input.data_ptr(),
            output.data_ptr(),
            input_size,
            output_size,
            _torch_dtype_to_mscclpp(dtype),
            ReduceOp.NOP,
            cuda_stream,
            None,  # executor (not needed for native algos)
            0,     # nblocks (auto)
            0,     # nthreads_per_block (auto)
            extras,
        )
        
        if result != 0:
            raise RuntimeError(f"alltoallv execution failed with code {result}")
        
        return output

    def __del__(self):
        """Cleanup resources."""
        # Let CUDA handle tensor cleanup automatically
        pass


# Module-level singleton for convenience
_default_instance: Optional[MscclppAlltoAllV] = None


def get_default_instance(**kwargs) -> MscclppAlltoAllV:
    """Get or create a default MscclppAlltoAllV instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = MscclppAlltoAllV(**kwargs)
    return _default_instance


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group=None,
    async_op: bool = False,
) -> Optional[torch.Tensor]:
    """
    Drop-in replacement for torch.distributed.all_to_all_single.
    
    Uses mscclpp optimized kernels internally for better performance,
    especially with imbalanced message sizes (e.g., MoE workloads).
    
    Note: This function requires prior initialization via get_default_instance()
    or will fall back to PyTorch's native implementation.
    
    Args:
        output: Pre-allocated output tensor
        input: Input tensor  
        output_split_sizes: Sizes to receive from each rank
        input_split_sizes: Sizes to send to each rank
        group: Process group (unused, for compatibility)
        async_op: If True, return async handle (not supported, falls back)
        
    Returns:
        None (modifies output in-place) or async handle if async_op=True
    """
    global _default_instance
    
    # Fall back to PyTorch if not initialized or async requested
    if _default_instance is None or async_op:
        return dist.all_to_all_single(
            output, input, 
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op
        )
    
    # Use optimized mscclpp implementation
    result = _default_instance.all_to_all_single(
        input=input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        output=output,
    )
    
    return None  # Matches torch.distributed API (async_op=False returns None)
