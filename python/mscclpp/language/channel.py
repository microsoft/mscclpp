# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import RemoteBuffer, SyncType, ReduceOperationType, Chunk, RankGroup
from mscclpp.language.thread_block_group import *
from mscclpp.language.internal.globals import get_program
from mscclpp.language.internal.operations import *
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MemoryChannel:
    """A memory channel for direct memory access communication between GPUs.

    MemoryChannel enables direct memory access between GPUs through memory mapping,
    providing high-performance communication for operations like put, get, and reduce.
    Each channel connects a source rank to a destination rank.

    Attributes:
        channel_id (int): Unique identifier for this channel within the source rank.
        dst_rank (int): The destination rank for communication operations.
        src_rank (int): The source rank for communication operations.
        channel_type (ChannelType): The type of channel (memory).
    """

    _channel_counts = defaultdict(int)

    @classmethod
    def reset(cls):
        """Reset all channel counts for this channel type."""
        cls._channel_counts.clear()

    def __init__(self, dst_rank: int, src_rank: int):
        """Initialize a new MemoryChannel.

        Args:
            dst_rank (int): The destination rank for this channel.
            src_rank (int): The source rank for this channel.

        Raises:
            RuntimeError: If src_rank or dst_rank is out of bounds for the current program.

        Example:
            >>> channel = MemoryChannel(dst_rank=1, src_rank=0)
        """
        num_ranks = get_program().num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = MemoryChannel._channel_counts[src_rank]
        MemoryChannel._channel_counts[src_rank] += 1

        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = ChannelType.memory
        get_program().add_channel(self)

    def signal(self, tb: int, data_sync: SyncType = SyncType.both, relaxed: bool = False):
        """Send a signal through the memory channel.

        Signals notify the destination that data is ready or an operation has completed.
        This is used for synchronization between ranks.

        Args:
            tb (int): The thread block ID that will execute this signal operation.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the signal operation.
                Defaults to SyncType.both.
            relaxed (bool, optional): Whether to use relaxed memory ordering.
                Defaults to False.

        Example:
            >>> channel.signal(tb=0, data_sync=SyncType.before)
        """
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_ids, self.channel_type, data_sync, relaxed)
        get_program().add_operation(self.src_rank, tb, op)

    def wait(self, tb: int, data_sync: SyncType = SyncType.both, relaxed: bool = False):
        """Wait for a signal through the memory channel.

        Waits for a signal from the destination rank, typically used for synchronization
        to ensure operations are completed before proceeding.

        Args:
            tb (int): The thread block ID that will execute this wait operation.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the wait operation.
                Defaults to SyncType.both.
            relaxed (bool, optional): Whether to use relaxed memory ordering.
                Defaults to False.

        Example:
            >>> channel.wait(tb=0, data_sync=SyncType.after)
        """
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = WaitOperation(tb_channel_ids, self.channel_type, data_sync, relaxed)
        get_program().add_operation(self.src_rank, tb, op)

    def get(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Retrieve data from remote memory to local memory.

        Performs a get operation to copy data from the destination rank's memory
        to the source rank's local memory through the memory channel.

        Args:
            dst_chunk (Chunk): The destination chunk on the source rank where data will be stored.
            src_chunk (Chunk): The source chunk on the destination rank to retrieve data from.
            tb (int, optional): The thread block ID that will execute this get operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this get operation. Defaults to None.

        Raises:
            RuntimeError: If chunk ranks don't match the channel configuration.

        Example:
            >>> channel.get(dst_chunk, src_chunk, tb=0)
        """
        if dst_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {dst_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if src_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {src_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )

        remote_chunk = RemoteBuffer(dst_chunk.rank, src_chunk.rank, src_chunk.buffer, self.channel_type)

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb_id, remote_chunk, self.channel_type)
            tb_channel_ids = get_program().setup_channel(tb, self)
            op = GetOperation(
                src_buff=[RemoteChunk(src_chunk.buffer, src_chunk.index, src_chunk.size, tb_chunk_id)],
                dst_buff=[LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
                channel_ids=tb_channel_ids,
                channel_type=self.channel_type,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
            )
            get_program().add_operation(self.src_rank, tb_id, op)

    def put(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Send data from local memory to remote memory.

        Performs a put operation to copy data from the source rank's local memory
        to the destination rank's memory through the memory channel.

        Args:
            dst_chunk (Chunk): The destination chunk on the destination rank where data will be stored.
            src_chunk (Chunk): The source chunk on the source rank to send data from.
            tb (int, optional): The thread block ID that will execute this put operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this put operation. Defaults to None.

        Raises:
            RuntimeError: If chunk ranks don't match the channel configuration or
                if chunk sizes don't match.

        Example:
            >>> channel.put(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb_id, remote_chunk, self.channel_type)
            tb_channel_ids = get_program().setup_channel(tb_id, self)
            op = PutOperation(
                src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
                dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
                channel_ids=tb_channel_ids,
                channel_type=self.channel_type,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
            )
            get_program().add_operation(self.src_rank, tb_id, op)

    def read_put_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Transfer data in packet format from local to remote scratch buffer.

        Performs a specialized put operation that transfers data in packet format
        from the source rank's scratch buffer to the destination rank's scratch buffer.
        Both source and destination chunks must be scratch buffers.

        Args:
            dst_chunk (Chunk): The destination scratch chunk on the destination rank.
            src_chunk (Chunk): The source scratch chunk on the source rank.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.

        Raises:
            RuntimeError: If chunk ranks don't match channel configuration, if chunks
                are not scratch buffers, or if chunk sizes don't match.

        Example:
            >>> channel.read_put_packet(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if src_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Source chunk must be of type scratch.")
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise ValueError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb_id, remote_chunk, self.channel_type)
            tb_channel_ids = get_program().setup_channel(tb_id, self)
            op = PutOperation(
                src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
                dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
                channel_ids=tb_channel_ids,
                channel_type=self.channel_type,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
                from_packet=True,
                to_packet=True,
            )
            get_program().add_operation(self.src_rank, tb_id, op)

    def put_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int = None, tb_group: ThreadBlockGroup = None):
        """Transfer data from local buffer to remote scratch buffer in packet format.

        Performs a put operation that transfers data from the source rank's buffer
        to the destination rank's scratch buffer in packet format. The destination
        chunk must be a scratch buffer.

        Args:
            dst_chunk (Chunk): The destination scratch chunk on the destination rank.
            src_chunk (Chunk): The source chunk on the source rank.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.

        Raises:
            RuntimeError: If chunk ranks don't match channel configuration, if destination
                chunk is not a scratch buffer, or if chunk sizes don't match.

        Example:
            >>> channel.put_packet(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb_id, remote_chunk, self.channel_type)
            tb_channel_ids = get_program().setup_channel(tb_id, self)
            op = PutOperation(
                src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
                dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
                channel_ids=tb_channel_ids,
                channel_type=self.channel_type,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
                from_packet=False,
                to_packet=True,
            )

            get_program().add_operation(self.src_rank, tb_id, op)

    def reduce(
        self,
        local_src_chunk: Chunk,
        remote_src_chunks: List[Chunk],
        tb: int = None,
        tb_group: ThreadBlockGroup = None,
        local_dst_chunk: Chunk = None,
        reduce_op: ReduceOperation = ReduceOperationType.sum,
    ):
        """Perform a reduction operation combining local and remote data.

        Reduces data from multiple remote source chunks with a local source chunk,
        storing the result in a local destination chunk. If no destination chunk
        is specified, the result is stored in the local source chunk.

        Args:
            local_src_chunk (Chunk): The local source chunk on the source rank.
            remote_src_chunks (List[Chunk]): List of remote source chunks to reduce with.
            tb (int, optional): The thread block ID that will execute this operation. Defaults to None.
            tb_group (ThreadBlockGroup, optional): The ThreadBlockGroup that will execute this operation. Defaults to None.
            local_dst_chunk (Chunk, optional): The local destination chunk. If None,
                uses local_src_chunk as destination. Defaults to None.
            reduce_op (ReduceOperation, optional): The reduction operation to perform.
                Defaults to ReduceOperationType.sum.

        Raises:
            RuntimeError: If chunk ranks don't match channel configuration or if
                chunk sizes are inconsistent.

        Example:
            >>> channel.reduce(local_chunk, remote_chunks, tb=0)
        """
        if local_dst_chunk is None:
            local_dst_chunk = local_src_chunk
        if local_src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Destination chunk rank {local_src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if local_src_chunk.size != local_dst_chunk.size:
            raise RuntimeError(
                f"Source chunk size {local_src_chunk.size} does not match destination chunk size {local_dst_chunk.size}."
            )
        for chunk in remote_src_chunks:
            if chunk.rank != self.dst_rank:
                raise RuntimeError(
                    f"Source chunk rank {chunk.rank} does not match current channel dst rank {self.dst_rank}."
                )
            if chunk.size != local_src_chunk.size:
                raise RuntimeError(
                    f"Source chunk size {chunk.size} does not match local source chunk size {local_src_chunk.size}."
                )

        if tb is not None:
            tb_list = [tb]
        elif tb_group is not None:
            tb_list = tb_group.tb_list
        else:
            raise RuntimeError(
                "Either 'tb' (thread block ID) or 'tb_group' (ThreadBlockGroup) must be provided, but both are None."
            )

        for tb_id in tb_list:
            remote_chunks = [
                RemoteChunk(
                    chunk.buffer,
                    chunk.index,
                    chunk.size,
                    get_program().setup_remote_chunk(
                        self.src_rank,
                        tb_id,
                        RemoteBuffer(local_src_chunk.rank, chunk.rank, chunk.buffer, self.channel_type),
                        self.channel_type,
                    ),
                )
                for chunk in remote_src_chunks
            ]
            tb_channel_ids = get_program().setup_channel(tb_id, self)

            op = ReduceOperation(
                local_src_buff=[LocalChunk(local_src_chunk.buffer, local_src_chunk.index, local_src_chunk.size)],
                local_dst_buff=[LocalChunk(local_dst_chunk.buffer, local_dst_chunk.index, local_dst_chunk.size)],
                remote_src_buff=remote_chunks,
                remote_dst_buff=[],
                channel_ids=tb_channel_ids,
                channel_type=self.channel_type,
                tbg_info=(
                    ThreadBlockGroupInfo(tb_group.get_internal_id(tb_id), tb_group.numtb())
                    if tb_group is not None
                    else None
                ),
                reduce_operation=reduce_op,
            )

            get_program().add_operation(self.src_rank, tb_id, op)


@dataclass
class PortChannel:
    """A port channel for communication using port-based mechanisms between GPUs.

    PortChannel enables communication between GPUs using interconnection ports, supporting
    operations such as put, signal, wait, and flush. Each channel connects a source rank to a
    destination rank and is suitable for scenarios requiring port-mapping-based data copy
    and synchronization methods.

    Attributes:
        channel_id (int): Unique identifier for this channel within the source rank.
        dst_rank (int): The destination rank for communication operations.
        src_rank (int): The source rank for communication operations.
        channel_type (ChannelType): The type of channel (port).
    """

    _channel_counts = defaultdict(int)

    @classmethod
    def reset(cls):
        """Reset all channel counts for this channel type."""
        cls._channel_counts.clear()

    def __init__(self, dst_rank: int, src_rank: int):
        """Initialize a new PortChannel.

        Args:
            dst_rank (int): The destination rank for this channel.
            src_rank (int): The source rank for this channel.

        Raises:
            RuntimeError: If src_rank or dst_rank is out of bounds for the current program.

        Example:
            >>> channel = PortChannel(dst_rank=1, src_rank=0)
        """
        num_ranks = get_program().num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = PortChannel._channel_counts[src_rank]
        PortChannel._channel_counts[src_rank] += 1

        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = ChannelType.port
        get_program().add_channel(self)

    def signal(self, tb: int, data_sync: SyncType = SyncType.both):
        """Send a signal through the port channel.

        Signals notify the destination that data is ready or an operation has completed.
        This is used for synchronization between ranks through port-based mechanisms.

        Args:
            tb (int): The thread block ID that will execute this signal operation.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the signal operation.
                Defaults to SyncType.both.

        Example:
            >>> channel.signal(tb=0, data_sync=SyncType.before)
        """
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_ids, self.channel_type, data_sync)
        get_program().add_operation(self.src_rank, tb, op)

    def wait(self, tb: int, data_sync: SyncType = SyncType.both):
        """Wait for a signal through the port channel.

        Waits for a signal from the destination rank, typically used for synchronization
        to ensure operations are completed before proceeding through port-based mechanisms.

        Args:
            tb (int): The thread block ID that will execute this wait operation.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the wait operation.
                Defaults to SyncType.both.

        Example:
            >>> channel.wait(tb=0, data_sync=SyncType.after)
        """
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = WaitOperation(tb_channel_ids, self.channel_type, data_sync)
        get_program().add_operation(self.src_rank, tb, op)

    def flush(self, tb: int, data_sync: SyncType = SyncType.both):
        """Flush pending operations through the port channel.

        Forces completion of all pending operations on the port channel, ensuring
        data consistency. This operation is only supported for port channels.

        Args:
            tb (int): The thread block ID that will execute this flush operation.
            data_sync (SyncType, optional): Defines the order where threads inside the thread block
                will be synchronized (equivalent to __syncthreads()) relative to the flush operation.
                Defaults to SyncType.both.

        Example:
            >>> channel.flush(tb=0, data_sync=SyncType.after)
        """
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = FlushOperation(tb_channel_ids, self.channel_type, data_sync)
        get_program().add_operation(self.src_rank, tb, op)

    def put(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        """Send data from local memory to remote memory through the port channel.

        Performs a put operation to copy data from the source rank's local memory
        to the destination rank's memory through the port channel.

        Args:
            dst_chunk (Chunk): The destination chunk on the destination rank where data will be stored.
            src_chunk (Chunk): The source chunk on the source rank to send data from.
            tb (int): The thread block ID that will execute this put operation.

        Raises:
            RuntimeError: If chunk ranks don't match the channel configuration or
                if chunk sizes don't match.

        Example:
            >>> channel.put(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = PutOperation(
            src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_with_signal(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        """Send data from local memory to remote memory with automatic signaling.

        Performs a put operation that transfers data and automatically sends a signal
        to notify the destination that the data transfer is complete. This combines
        data transfer and synchronization in a single operation.

        Args:
            dst_chunk (Chunk): The destination chunk on the destination rank where data will be stored.
            src_chunk (Chunk): The source chunk on the source rank to send data from.
            tb (int): The thread block ID that will execute this put operation.

        Raises:
            RuntimeError: If chunk ranks don't match the channel configuration or
                if chunk sizes don't match.

        Example:
            >>> channel.put_with_signal(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = PutOperation(
            src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            with_signal=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_with_signal_and_flush(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        """Send data from local memory to remote memory with signal and flush.

        Performs a put operation that transfers data, automatically sends a signal,
        and flushes the channel. This provides the guarantee of data transfer completion.

        Args:
            dst_chunk (Chunk): The destination chunk on the destination rank where data will be stored.
            src_chunk (Chunk): The source chunk on the source rank to send data from.
            tb (int): The thread block ID that will execute this put operation.

        Raises:
            RuntimeError: If chunk ranks don't match the channel configuration or
                if chunk sizes don't match.

        Example:
            >>> channel.put_with_signal_and_flush(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = PutOperation(
            src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            with_signal_and_flush=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        """Transfer data from local buffer to remote scratch buffer in packet format.

        Performs a specialized put operation that transfers data from the source rank's buffer
        to the destination rank's scratch buffer in packet format through the port channel.
        The destination chunk must be a scratch buffer.

        Args:
            dst_chunk (Chunk): The destination scratch chunk on the destination rank.
            src_chunk (Chunk): The source chunk on the source rank (any buffer type).
            tb (int): The thread block ID that will execute this operation.

        Raises:
            RuntimeError: If chunk ranks don't match channel configuration, if destination
                chunk is not a scratch buffer, or if chunk sizes don't match.

        Example:
            >>> channel.put_packets(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = PutOperation(
            src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            from_packet=False,
            to_packet=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def read_put_packets(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        """Transfer data in packet format from local to remote scratch buffer.

        Performs a specialized put operation that transfers data in packet format
        from the source rank's scratch buffer to the destination rank's scratch buffer
        through the port channel. Both source and destination chunks must be scratch buffers.

        Args:
            dst_chunk (Chunk): The destination scratch chunk on the destination rank.
            src_chunk (Chunk): The source scratch chunk on the source rank.
            tb (int): The thread block ID that will execute this operation.

        Raises:
            RuntimeError: If chunk ranks don't match channel configuration, if chunks
                are not scratch buffers, or if chunk sizes don't match.

        Example:
            >>> channel.read_put_packet(dst_chunk, src_chunk, tb=0)
        """
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if src_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Source chunk must be of type scratch.")
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )
        if dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Destination chunk size {dst_chunk.size} does not match source chunk size {src_chunk.size}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = PutOperation(
            src_buff=[LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            dst_buff=[RemoteChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size, tb_chunk_id)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            from_packet=True,
            to_packet=True,
        )

        get_program().add_operation(self.src_rank, tb, op)


@dataclass
class SwitchChannel:
    """A switch channel for collective communication operations among multiple GPUs.

    SwitchChannel enables collective operations like reduce and broadcast among a group
    of ranks through a switch-based communication mechanism. It supports operations
    on shared buffers across multiple ranks in the group.

    Attributes:
        channel_ids (dict): Dictionary mapping ranks to their channel IDs.
        channel_type (ChannelType): The type of channel (switch).
        buffer_type (BufferType): The type of buffer used for operations.
        rank_group (RankGroup): The group of ranks participating in this channel.
    """

    _channel_counts = defaultdict(int)

    @classmethod
    def reset(cls):
        """Reset all channel counts for this channel type."""
        cls._channel_counts.clear()

    def __init__(self, rank_list: List[int], buffer_type: BufferType):
        """Initialize a new SwitchChannel.

        Args:
            rank_list (List[int]): List of ranks that will participate in this switch channel.
            buffer_type (BufferType): The type of buffer to use for switch operations.

        Raises:
            RuntimeError: If any rank in rank_list is out of bounds for the current program.

        Example:
            >>> channel = SwitchChannel(rank_list=[0, 1, 2, 3], buffer_type=BufferType.input)
        """
        num_ranks = get_program().num_ranks
        if not all(rank < num_ranks for rank in rank_list):
            raise RuntimeError(f"One or more ranks in {rank_list} are out of bounds. Number of ranks: {num_ranks}")

        self.channel_ids = {}
        for rank in rank_list:
            if rank >= num_ranks:
                raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {num_ranks}")
            self.channel_ids[rank] = SwitchChannel._channel_counts[rank]
            SwitchChannel._channel_counts[rank] += 1

        self.channel_type = ChannelType.switch
        self.buffer_type = buffer_type
        self.rank_group = RankGroup(len(rank_list), rank_list)

        get_program().add_channel(self)

    def at_rank(self, rank):
        """Get a rank-specific view of this switch channel.

        Returns a SwitchChannelRankView that provides rank-specific operations
        for reduce and broadcast on this switch channel.

        Args:
            rank (int): The rank to create a view for.

        Returns:
            SwitchChannelRankView: A rank-specific view of this channel.

        Raises:
            RuntimeError: If rank is not part of this channel's rank group.

        Example:
            >>> channel.at_rank(0)
        """
        if rank not in self.rank_group.ranks:
            raise RuntimeError(f"Rank {rank} is not part of this SwitchChannel's rank group.")
        return SwitchChannel.SwitchChannelRankView(self, rank)

    def reduce(self, rank, buffer_offset, size, dst_chunk: Chunk, tb, reduce_op=ReduceOperationType.sum):
        """Perform a reduction operation across all ranks in the switch channel.

        Reduces data from the specified buffer region across all ranks in the
        rank group, storing the result in the destination chunk.

        Args:
            rank (int): The rank that will execute this reduction operation.
            buffer_offset (int): The offset in the buffer where reduction data starts.
            size (int): The size of data to reduce.
            dst_chunk (Chunk): The destination chunk where the result will be stored.
            tb (int): The thread block ID that will execute this operation.
            reduce_op (ReduceOperationType, optional): The reduction operation to perform.
                Defaults to ReduceOperationType.sum.

        Raises:
            RuntimeError: If dst_chunk rank is not in the rank group, if chunk size
                doesn't match the required size, or if buffer size is insufficient.

        Example:
            >>> channel.reduce(rank=0, buffer_offset=0, size=1, dst_chunk=chunk, tb=0)
        """
        self.src_rank = rank
        if dst_chunk.rank not in self.rank_group.ranks:
            raise RuntimeError(
                f"Destination chunk rank {dst_chunk.rank} is not part of the rank group {self.rank_group.ranks}."
            )
        if dst_chunk.size != size:
            raise RuntimeError(f"Destination chunk size {dst_chunk.size} does not match the required size {size}.")

        for rank in self.rank_group.ranks:
            if self.buffer_type == BufferType.scratch:
                buffer_size = get_program().gpus[rank].scratch_chunks
            else:
                buffer_size = get_program().buffers[rank][self.buffer_type].size

            if buffer_size < buffer_offset + size:
                raise RuntimeError(
                    f"Buffer size {buffer_size} is smaller than required size {buffer_offset + size} for rank {rank}."
                )

        tb_channel_ids = get_program().setup_channel(tb, self)
        op = GroupLoadReduce(
            self.buffer_type,
            buffer_offset,
            size,
            dst_chunk,
            tb_channel_ids,
            self.channel_type,
            reduce_op,
        )
        get_program().add_operation(self.src_rank, tb, op)

    def broadcast(self, rank, src_chunk: Chunk, buffer_offset, size, tb):
        """Broadcast data from source chunk to all ranks in the switch channel.

        Broadcasts data from the source chunk to the specified buffer region
        across all ranks in the rank group.

        Args:
            rank (int): The rank that will execute this broadcast operation.
            src_chunk (Chunk): The source chunk containing data to broadcast.
            buffer_offset (int): The offset in the destination buffer where data will be stored.
            size (int): The size of data to broadcast.
            tb (int): The thread block ID that will execute this operation.

        Raises:
            RuntimeError: If src_chunk rank is not in the rank group, if chunk size
                doesn't match the required size, or if buffer size is insufficient.

        Example:
            >>> channel.broadcast(rank=0, src_chunk=chunk, buffer_offset=0, size=1, tb=0)
        """
        self.src_rank = rank
        if src_chunk.rank not in self.rank_group.ranks:
            raise RuntimeError(
                f"Destination chunk rank {src_chunk.rank} is not part of the rank group {self.rank_group.ranks}."
            )
        if src_chunk.size != size:
            raise RuntimeError(f"Destination chunk size {src_chunk.size} does not match the required size {size}.")

        for rank in self.rank_group.ranks:
            if self.buffer_type == BufferType.scratch:
                buffer_size = get_program().gpus[rank].scratch_chunks
            else:
                buffer_size = get_program().buffers[rank][self.buffer_type].size

            if buffer_size < buffer_offset + size:
                raise RuntimeError(
                    f"Buffer size {buffer_size} is smaller than required size {buffer_offset + size} for rank {rank}."
                )

        tb_channel_ids = get_program().setup_channel(tb, self)
        op = GroupStore(src_chunk, self.buffer_type, buffer_offset, size, tb_channel_ids, self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

    class SwitchChannelRankView:
        """A rank-specific view of a SwitchChannel for performing operations.

        This class provides a convenient interface for performing switch channel
        operations from the perspective of a specific rank, automatically passing
        the rank parameter to the underlying channel methods.

        Attributes:
            _channel (SwitchChannel): The underlying switch channel.
            _rank (int): The rank this view represents.
        """

        def __init__(self, channel, rank):
            """Initialize a new SwitchChannelRankView.

            Args:
                channel (SwitchChannel): The switch channel to create a view for.
                rank (int): The rank this view will represent.
            """
            self._channel: SwitchChannel = channel
            self._rank: int = rank

        def reduce(self, buffer_offset, size, dst_chunk: Chunk, tb, reduce_op=ReduceOperationType.sum):
            """Perform a reduction operation from this rank's perspective.

            Convenience method that calls the underlying channel's reduce method
            with this view's rank automatically provided.

            Args:
                buffer_offset (int): The offset in the buffer where reduction data starts.
                size (int): The size of data to reduce.
                dst_chunk (Chunk): The destination chunk where the result will be stored.
                tb (int): The thread block ID that will execute this operation.
                reduce_op (ReduceOperationType, optional): The reduction operation to perform.
                    Defaults to ReduceOperationType.sum.

            Returns:
                The result of the underlying channel's reduce operation.

            Example:
                >>> rank_view.reduce(buffer_offset=0, size=1, dst_chunk=chunk, tb=0)
            """
            return self._channel.reduce(self._rank, buffer_offset, size, dst_chunk, tb, reduce_op)

        def broadcast(self, src_chunk: Chunk, buffer_offset, size, tb):
            """Perform a broadcast operation from this rank's perspective.

            Convenience method that calls the underlying channel's broadcast method
            with this view's rank automatically provided.

            Args:
                src_chunk (Chunk): The source chunk containing data to broadcast.
                buffer_offset (int): The offset in the destination buffer where data will be stored.
                size (int): The size of data to broadcast.
                tb (int): The thread block ID that will execute this operation.

            Returns:
                The result of the underlying channel's broadcast operation.

            Example:
                >>> rank_view.broadcast(src_chunk=chunk, buffer_offset=0, size=1, tb=0)
            """
            return self._channel.broadcast(self._rank, src_chunk, buffer_offset, size, tb)
