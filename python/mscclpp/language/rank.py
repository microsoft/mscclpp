# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import BufferType, Chunk
from mscclpp.language.thread_block_group import *
from mscclpp.language.internal.operations import *
from mscclpp.language.internal.globals import get_program
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Rank:
    """Represents a single rank (GPU) in an MSCCL++ program.

    Rank provides operations that can be performed locally on a single GPU,
    including copy operations, reduce operations, and synchronization barriers.
    It manages local buffer operations and coordinates with other ranks through
    the program context.

    Attributes:
        rank (int): The rank identifier for this GPU.
    """

    rank: int

    def __init__(self, rank: int):
        """Initialize a new Rank.

        Args:
            rank (int): The rank identifier for this GPU.

        Raises:
            RuntimeError: If rank is out of bounds for the current program.

        Example:
            >>> rank = Rank(0)
        """
        self.rank = rank
        if rank >= get_program().num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

    def get_input_buffer(self):
        """Get the input buffer for this rank.

        Returns:
            BaseBuffer: The input buffer associated with this rank.

        Example:
            >>> input_buf = rank.get_input_buffer()
        """
        return get_program().buffers[self.rank][BufferType.input]

    def get_output_buffer(self):
        """Get the output buffer for this rank.

        Returns:
            BaseBuffer: The output buffer associated with this rank.

        Example:
            >>> output_buf = rank.get_output_buffer()
        """
        return get_program().buffers[self.rank][BufferType.output]

    def _copy(
        self,
        dst_chunk: Chunk,
        src_chunk: Chunk,
        tb: int = None,
        tb_group: ThreadBlockGroup = None,
        from_packet: bool = False,
        to_packet: bool = False,
    ):
        """Internal copy operation implementation.

        Performs a local copy operation between chunks on this rank with optional
        packet format conversion. This is used by the public copy methods.

        Args:
            dst_chunk (Chunk): The destination chunk to copy data to.
            src_chunk (Chunk): The source chunk to copy data from.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.
            from_packet (bool, optional): Whether to unpack from packet format. Defaults to False.
            to_packet (bool, optional): Whether to pack to packet format. Defaults to False.

        Raises:
            RuntimeError: If chunk ranks don't match this rank, if chunk sizes differ,
                or if packet operations are used with non-scratch buffers.
        """
        if dst_chunk.rank != self.rank or src_chunk.rank != self.rank:
            raise RuntimeError(
                f"Inconsistent ranks: dst {dst_chunk.rank}, src {src_chunk.rank}, self {self.rank}. They must match."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Inconsistent chunk sizes: dst {dst_chunk.size}, src {src_chunk.size}. They must match."
            )
        if from_packet and src_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Source chunk must be of type scratch.")
        if to_packet and dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            op = CopyOperation(
                src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
                dst_buff=[LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
                from_packet=from_packet,
                to_packet=to_packet,
            )

            get_program().add_operation(self.rank, tb_id, op)

    def copy(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Copy data from source chunk to destination chunk.

        Performs a simple local copy operation between two chunks on this rank
        without any packet format conversion.

        Args:
            dst_chunk (Chunk): The destination chunk to copy data to.
            src_chunk (Chunk): The source chunk to copy data from.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.

        Example:
            >>> rank.copy(dst_chunk, src_chunk, tb=0)
        """
        self._copy(dst_chunk=dst_chunk, src_chunk=src_chunk, tb=tb, tb_group=tb_group)

    def unpack_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Copy data from packet format to regular format.

        Unpacks data from packet format in the source scratch buffer and copies
        it to the destination chunk in regular format.

        Args:
            dst_chunk (Chunk): The destination chunk to copy unpacked data to.
            src_chunk (Chunk): The source scratch chunk containing packed data.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.

        Example:
            >>> rank.unpack_packet(dst_chunk, src_chunk, tb=0)
        """
        self._copy(dst_chunk=dst_chunk, src_chunk=src_chunk, tb=tb, tb_group=tb_group, from_packet=True)

    def copy_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Copy data from regular format to packet format.

        Packs data from the source chunk and copies it to the destination
        scratch buffer in packet format.

        Args:
            dst_chunk (Chunk): The destination scratch chunk to store packed data.
            src_chunk (Chunk): The source chunk containing data to pack.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.

        Example:
            >>> rank.copy_packet(dst_chunk, src_chunk, tb=0)
        """
        self._copy(dst_chunk=dst_chunk, src_chunk=src_chunk, tb=tb, tb_group=tb_group, to_packet=True)

    def reduce(
        self,
        src_chunk: Chunk,
        other_chunks: List[Chunk],
        tb: int = None,
        tb_group: ThreadBlockGroup = None,
        dst_chunk: Chunk = None,
        reduce_op: ReduceOperationType = ReduceOperationType.sum,
        packet: bool = False,
    ):
        """Perform a local reduction operation on this rank.

        Reduces data from multiple chunks locally on this rank, combining
        the source chunk with other chunks using the specified reduction operation.

        Args:
            src_chunk (Chunk): The primary source chunk to reduce.
            other_chunks (List[Chunk]): Additional chunks to include in the reduction.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.
            dst_chunk (Chunk, optional): The destination chunk for the result.
                If None, uses src_chunk. Defaults to None.
            reduce_op (ReduceOperationType, optional): The reduction operation to perform.
                Defaults to ReduceOperationType.sum.
            packet (bool, optional): Whether to operate in packet format. Defaults to False.

        Raises:
            RuntimeError: If chunk ranks don't match this rank, if chunk sizes are inconsistent,
                or if other_chunks is empty.

        Example:
            >>> rank.reduce(src_chunk, other_chunks, tb=0, dst_chunk)
        """
        if dst_chunk is None:
            dst_chunk = src_chunk
        if dst_chunk.rank != self.rank or src_chunk.rank != self.rank:
            raise RuntimeError(
                f"Inconsistent ranks: dst {dst_chunk.rank}, src {src_chunk.rank}, self {self.rank}. They must match."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Inconsistent chunk sizes: dst {dst_chunk.size}, src {src_chunk.size}. They must match."
            )
        if len(other_chunks) == 0:
            raise RuntimeError("Other chunk empty.")
        for chunk in other_chunks:
            if chunk.rank != self.rank:
                raise RuntimeError(f"Other chunk rank {chunk.rank} does not match current rank {self.rank}.")
            if chunk.size != src_chunk.size:
                raise RuntimeError(
                    f"Inconsistent chunk sizes: other {chunk.size}, src {src_chunk.size}. They must match."
                )
            if packet and chunk.buffer != BufferType.scratch:
                raise RuntimeError(f"Other chunk must be of type scratch.")

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            op = ReduceOperation(
                [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)]
                + [LocalChunk(chunk.buffer, chunk.index, chunk.size) for chunk in other_chunks],
                [LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
                reduce_operation=reduce_op,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
                packet=packet,
            )

            get_program().add_operation(self.rank, tb_id, op)

    def barrier(self, tb_list: List[int]):
        """Create a synchronization barrier between thread blocks.

        Synchronizes execution between multiple thread blocks on this rank.
        For a single thread block, creates a sync operation. For multiple
        thread blocks, creates a barrier operation.

        Args:
            tb_list (List[int]): List of thread block IDs to synchronize.

        Raises:
            RuntimeError: If tb_list is empty.

        Example:
            >>> rank0.barrier(tb_list=[0, 1, 2])
        """
        if len(tb_list) == 0:
            raise RuntimeError("Barrier requires at least thread block.")
        elif len(tb_list) == 1:
            op = SyncOperation()
            get_program().add_operation(self.rank, tb_list[0], op)
        else:
            op = BarrierOperation(self.rank, tb_list)
            for tb in tb_list:
                get_program().add_operation(self.rank, tb, op)


class BaseBuffer:
    """Base class for buffer objects in MSCCL++ programs.

    BaseBuffer represents a memory buffer associated with a specific rank,
    providing indexed access to create chunks for communication operations.
    It supports slice-based indexing to create Chunk objects.

    Attributes:
        rank (int): The rank that owns this buffer.
        buffer_type (BufferType): The type of buffer (input, output, scratch).
        offset (int): The starting offset of this buffer.
        size (int): The total size of the buffer.
    """

    def __init__(self, rank: int, buffer_type: BufferType, offset: int, size: int):
        self.rank = rank
        self.buffer_type = buffer_type
        self.offset = offset
        self.size = offset + size

    def __getitem__(self, key):
        if self.offset + key.stop > self.size:
            raise RuntimeError(
                f"Index range from {self.offset + key.start} - {self.offset + key.stop} is out of bounds for buffer {self.buffer_type}. Buffer size: {self.size}"
            )
        return Chunk(self.rank, self.buffer_type, self.offset + key.start, key.stop - key.start)


class Buffer(BaseBuffer):
    """A scratch buffer for temporary data storage during communication operations.

    Buffer extends BaseBuffer to provide dynamically allocated scratch space
    for a specific rank. It automatically manages scratch buffer allocation
    within the GPU's scratch memory space.

    Attributes:
        rank (int): The rank that owns this buffer.
        buffer_type (BufferType): Always BufferType.scratch for Buffer instances.
        offset (int): The starting offset within the rank's scratch space.
        size (int): The total size of the allocated buffer.
    """

    def __init__(self, rank: int, size: int):
        """Initialize a new scratch Buffer.

        Allocates a scratch buffer of the specified size for the given rank,
        automatically managing the offset within the rank's scratch space.

        Args:
            rank (int): The rank to allocate the buffer for.
            size (int): The size of the buffer to allocate.

        Raises:
            RuntimeError: If rank is out of bounds for the current program.

        Example:
            >>> scratch_buf = Buffer(rank=0, size=2)
            >>> chunk = scratch_buf[0:1]
        """
        if rank >= get_program().num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

        self.rank = rank
        self.buffer_type = BufferType.scratch
        self.offset = get_program().gpus[rank].scratch_chunks
        self.size = self.offset + size
        get_program().gpus[rank].scratch_chunks += size


class Semaphore:
    """A semaphore for asynchronus synchronization between thread blocks.

    Semaphore provides acquire and release operations for synchronization
    between thread blocks within a rank. Each semaphore has an initial value
    and supports typical semaphore semantics.

    Attributes:
        id (int): Unique identifier for this semaphore within its rank.
        rank (int): The rank that owns this semaphore.
        initial_value (int): The initial value of the semaphore.
    """

    _semaphore_counts = defaultdict(int)

    @classmethod
    def reset(cls):
        """Reset all semaphore counts."""
        cls._semaphore_counts.clear()

    def __init__(self, rank: int, initial_value: int):
        """Initialize a new Semaphore.

        Creates a semaphore for the specified rank with the given initial value.

        Args:
            rank (int): The rank that will own this semaphore.
            initial_value (int): The initial value of the semaphore.

        Raises:
            RuntimeError: If rank is out of bounds for the current program.

        Example:
            >>> sem = Semaphore(rank=0, initial_value=1)
        """
        num_ranks = get_program().num_ranks
        if rank >= num_ranks:
            raise RuntimeError(f"Source rank {rank} is out of bounds. Number of ranks: {num_ranks}")
        self.id = Semaphore._semaphore_counts[rank]
        Semaphore._semaphore_counts[rank] += 1

        self.rank = rank
        self.initial_value = initial_value

        get_program().add_semaphore(self)

    def acquire(self, tb: int, data_sync: SyncType = SyncType.both):
        """Acquire the semaphore from a thread block.

        Blocks the thread block until the semaphore can be acquired (value > 0),
        then decrements the semaphore value.

        Args:
            tb (int): The thread block ID that will acquire the semaphore.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the acquire operation.
                Defaults to SyncType.both.

        Example:
            >>> sem.acquire(tb=0, data_sync=SyncType.before)
        """
        op = SemaphoreAcquireOperation([self.id], data_sync)
        get_program().add_operation(self.rank, tb, op)

    def release(self, tb: int, data_sync: SyncType = SyncType.both):
        """Release the semaphore from a thread block.

        Increments the semaphore value, potentially unblocking other thread blocks
        waiting to acquire the semaphore.

        Args:
            tb (int): The thread block ID that will release the semaphore.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the release operation.
                Defaults to SyncType.both.

        Example:
            >>> sem.release(tb=0, data_sync=SyncType.after)
        """
        op = SemaphoreReleaseOperation([self.id], data_sync)
        get_program().add_operation(self.rank, tb, op)
