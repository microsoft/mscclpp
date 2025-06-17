from mscclpp.language.internal.types import RemoteBuffer, SyncType, ReduceOperationType, Chunk, RankGroup
from mscclpp.language.internal.globals import get_program
from mscclpp.language.internal.operations import *
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Channel:
    __channel_counts = defaultdict(int)

    def __init__(self, dst_rank: int, src_rank: int):
        num_ranks = get_program().num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = Channel.__channel_counts[src_rank]
        Channel.__channel_counts[src_rank] += 1

        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = ChannelType.memory
        get_program().add_channel(self)

    def signal(self, tb: int, data_sync: SyncType = SyncType.none, relaxed=False):
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_ids, self.channel_type, data_sync, relaxed)
        get_program().add_operation(self.src_rank, tb, op)

    def wait(self, tb: int, data_sync: SyncType = SyncType.none, relaxed=False):
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = WaitOperation(tb_channel_ids, self.channel_type, data_sync, relaxed)
        get_program().add_operation(self.src_rank, tb, op)

    def get(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        if dst_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {dst_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if src_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {src_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )

        remote_chunk = RemoteBuffer(dst_chunk.rank, src_chunk.rank, src_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk, self.channel_type)
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = GetOperation(
            src_buff=[RemoteChunk(tb_chunk_id, src_chunk.index, src_chunk.size)],
            dst_buff=[LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put(
        self,
        dst_chunk: Chunk,
        src_chunk: Chunk,
        tb: int,
    ):
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_packet(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int, from_packet: bool = False):
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if from_packet and src_chunk.buffer != BufferType.scratch:
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            from_packet=from_packet,
            to_packet=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def reduce(
        self,
        local_src_chunk: Chunk,
        remote_src_chunks: List[Chunk],
        tb: int,
        local_dst_chunk: Chunk = None,
        reduce_op: ReduceOperation = ReduceOperationType.sum,
    ):
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

        remote_chunks = [
            RemoteChunk(
                get_program().setup_remote_chunk(
                    self.src_rank,
                    tb,
                    RemoteBuffer(local_src_chunk.rank, chunk.rank, chunk.buffer, self.channel_type),
                    self.channel_type,
                ),
                chunk.index,
                chunk.size,
            )
            for chunk in remote_src_chunks
        ]
        tb_channel_ids = get_program().setup_channel(tb, self)

        op = ReduceOperation(
            local_src_buff=[LocalChunk(local_src_chunk.buffer, local_src_chunk.index, local_src_chunk.size)],
            local_dst_buff=[LocalChunk(local_dst_chunk.buffer, local_dst_chunk.index, local_dst_chunk.size)],
            remote_src_buff=remote_chunks,
            remote_dst_buff=[],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            reduce_operation=reduce_op,
        )

        get_program().add_operation(self.src_rank, tb, op)


@dataclass
class PortChannel:
    __channel_counts = defaultdict(int)

    def __init__(self, dst_rank: int, src_rank: int):
        num_ranks = get_program().num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = PortChannel.__channel_counts[src_rank]
        PortChannel.__channel_counts[src_rank] += 1

        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = ChannelType.port
        get_program().add_channel(self)

    def signal(self, tb: int, data_sync: SyncType = SyncType.none, relaxed=False):
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_ids, self.channel_type, relaxed)
        get_program().add_operation(self.src_rank, tb, data_sync, op)

    def wait(self, tb: int, data_sync: SyncType = SyncType.none, relaxed=False):
        tb_channel_ids = get_program().setup_channel(tb, self)
        op = WaitOperation(tb_channel_ids, self.channel_type, data_sync, relaxed)
        get_program().add_operation(self.src_rank, tb, op)

    def flush(self, tb: int, data_sync: SyncType = SyncType.none):
        if self.channel_type != ChannelType.port:
            raise RuntimeError(f"Flush operation is only supported for ChannelType.port.")

        tb_channel_ids = get_program().setup_channel(tb, self)
        op = FlushOperation(tb_channel_ids, self.channel_type, data_sync)
        get_program().add_operation(self.src_rank, tb, op)

    def put(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_with_signal(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            with_signal=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put_with_signal_and_flush(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            with_signal_and_flush=True,
        )

        get_program().add_operation(self.src_rank, tb, op)

    # Put operation transfer in packet format on the local buffer to packet format on the remote buffer.
    def put_packet(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
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
            dst_buff=[RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            channel_ids=tb_channel_ids,
            channel_type=self.channel_type,
            from_packet=True,
            to_packet=True,
        )

        get_program().add_operation(self.src_rank, tb, op)


@dataclass
class SwitchChannel:
    __channel_counts = defaultdict(int)

    def __init__(self, rank_list: List[int], buffer_type: BufferType):
        num_ranks = get_program().num_ranks
        if not all(rank < num_ranks for rank in rank_list):
            raise RuntimeError(f"One or more ranks in {rank_list} are out of bounds. Number of ranks: {num_ranks}")

        self.channel_ids = {}
        for rank in rank_list:
            if rank >= num_ranks:
                raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {num_ranks}")
            self.channel_ids[rank] = SwitchChannel.__channel_counts[rank]
            SwitchChannel.__channel_counts[rank] += 1

        self.channel_type = ChannelType.switch
        self.buffer_type = buffer_type
        self.rank_group = RankGroup(len(rank_list), rank_list)

        get_program().add_channel(self)

    def at_rank(self, rank):
        if rank not in self.rank_group.ranks:
            raise RuntimeError(f"Rank {rank} is not part of this SwitchChannel's rank group.")
        return SwitchChannel.SwitchChannelRankView(self, rank)

    def group_load_reduce(self, rank, buffer_offset, size, dst_chunk: Chunk, tb, reduce_op=ReduceOperationType.sum):
        self.src_rank = rank
        if dst_chunk.rank not in self.rank_group.ranks:
            raise RuntimeError(
                f"Destination chunk rank {dst_chunk.rank} is not part of the rank group {self.rank_group.ranks}."
            )
        if dst_chunk.size != size:
            raise RuntimeError(f"Destination chunk size {dst_chunk.size} does not match the required size {size}.")

        if self.buffer_type != BufferType.scratch:
            for rank in self.rank_group.ranks:
                if get_program().buffers[rank][self.buffer_type].size < buffer_offset + size:
                    raise RuntimeError(
                        f"Buffer size {get_program().buffers[rank][BufferType.input].size} is smaller than "
                        f"required size {buffer_offset + size} for rank {rank}."
                    )
        else:
            for rank in self.rank_group.ranks:
                if get_program().gpus[rank].scratch_chunks < buffer_offset + size:
                    get_program().gpus[rank].scratch_chunks = buffer_offset + size

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

    def group_store(self, rank, src_chunk: Chunk, buffer_offset, size, tb):
        self.src_rank = rank
        if src_chunk.rank not in self.rank_group.ranks:
            raise RuntimeError(
                f"Destination chunk rank {src_chunk.rank} is not part of the rank group {self.rank_group.ranks}."
            )
        if src_chunk.size != size:
            raise RuntimeError(f"Destination chunk size {src_chunk.size} does not match the required size {size}.")

        if self.buffer_type != BufferType.scratch:
            for rank in self.rank_group.ranks:
                if get_program().buffers[rank][self.buffer_type].size < buffer_offset + size:
                    raise RuntimeError(
                        f"Buffer size {get_program().buffers[rank][BufferType.input].size} is smaller than "
                        f"required size {buffer_offset + size} for rank {rank}."
                    )
        else:
            for rank in self.rank_group.ranks:
                if get_program().gpus[rank].scratch_chunks < buffer_offset + size:
                    get_program().gpus[rank].scratch_chunks = buffer_offset + size

        tb_channel_ids = get_program().setup_channel(tb, self)
        op = GroupStore(src_chunk, self.buffer_type, buffer_offset, size, tb_channel_ids, self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

    class SwitchChannelRankView:
        def __init__(self, channel, rank):
            self._channel: SwitchChannel = channel
            self._rank: int = rank

        def group_load_reduce(self, buffer_offset, size, dst_chunk: Chunk, tb, reduce_op=ReduceOperationType.sum):
            return self._channel.group_load_reduce(self._rank, buffer_offset, size, dst_chunk, tb, reduce_op)

        def group_store(self, src_chunk: Chunk, buffer_offset, size, tb):
            return self._channel.group_store(self._rank, src_chunk, buffer_offset, size, tb)
