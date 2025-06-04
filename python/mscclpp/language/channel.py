from mscclpp.language.internal.types import RemoteBuffer, SyncType, ReduceOperationType, Chunk, RankGroup
from mscclpp.language.internal.globals import get_program
from mscclpp.language.internal.operations import *
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Channel:
    __channel_counts = defaultdict(lambda: defaultdict(int))

    def __init__(self, dst_rank: int, src_rank: int, channel_type: ChannelType):
        num_ranks = get_program().num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = Channel.__channel_counts[src_rank][channel_type]
        Channel.__channel_counts[src_rank][channel_type] += 1

        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = channel_type
        get_program().add_channel(self)

    def signal(self, tb: int, sync: SyncType = SyncType.none):
        if sync == SyncType.before or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_id, self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

        if sync == SyncType.after or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

    def wait(self, tb: int, sync: SyncType = SyncType.none):
        if sync == SyncType.before or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

        tb_channel_id = get_program().setup_channel(tb, self)
        op = WaitOperation(tb_channel_id, self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

        if sync == SyncType.after or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

    def relaxed_signal(self, tb: int, sync: SyncType = SyncType.none):
        if sync == SyncType.before or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_id, self.channel_type, True)
        get_program().add_operation(self.src_rank, tb, op)

        if sync == SyncType.after or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

    def relaxed_wait(self, tb: int, sync: SyncType = SyncType.none):
        if sync == SyncType.before or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation(tb_channel_id, self.channel_type, True)
        get_program().add_operation(self.src_rank, tb, op)

        if sync == SyncType.after or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

    def flush(self, tb: int, sync: SyncType = SyncType.none):
        if self.channel_type != ChannelType.port:
            raise RuntimeError(f"Flush operation is only supported for ChannelType.port.")

        if sync == SyncType.before or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

        tb_channel_id = get_program().setup_channel(tb, self)
        op = FlushOperation(tb_channel_id, self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

        if sync == SyncType.after or sync == SyncType.both:
            sync_op = SyncOperation()
            get_program().add_operation(self.src_rank, tb, sync_op)

    def get(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int):
        if dst_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {dst_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if src_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {src_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )

        remote_chunk = RemoteBuffer(src_chunk.rank, src_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk)
        tb_channel_id = get_program().setup_channel(tb, self)

        op = GetOperation(
            [RemoteChunk(tb_chunk_id, src_chunk.index, src_chunk.size)],
            [LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
            tb_channel_id,
            self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)

    def put(
        self,
        dst_chunk: Chunk,
        src_chunk: Chunk,
        tb: int,
        with_signal: bool = False,
        with_signal_and_flush: bool = False,
    ):
        if (with_signal or with_signal_and_flush) and self.channel_type != ChannelType.port:
            raise RuntimeError(f"Only ChannelType.port support put with signal operation.")
        if src_chunk.rank != self.src_rank:
            raise RuntimeError(
                f"Source chunk rank {src_chunk.rank} does not match current channel source rank {self.src_rank}."
            )
        if dst_chunk.rank != self.dst_rank:
            raise RuntimeError(
                f"Dst chunk rank {dst_chunk.rank} does not match current channel dst rank {self.dst_rank}."
            )

        remote_chunk = RemoteBuffer(dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk)
        tb_channel_id = get_program().setup_channel(tb, self)

        op = PutOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            [RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            tb_channel_id,
            self.channel_type,
            with_signal=with_signal,
            with_signal_and_flush=with_signal_and_flush,
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

        remote_chunk = RemoteBuffer(dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(self.src_rank, tb, remote_chunk)
        tb_channel_id = get_program().setup_channel(tb, self)

        op = PutOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            [RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            tb_channel_id,
            self.channel_type,
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
        for chunk in remote_src_chunks:
            if chunk.rank != self.dst_rank:
                raise RuntimeError(
                    f"Source chunk rank {chunk.rank} does not match current channel dst rank {self.dst_rank}."
                )

        remote_chunks = [
            RemoteChunk(
                get_program().setup_remote_chunk(
                    self.src_rank, tb, RemoteBuffer(chunk.rank, chunk.buffer, self.channel_type)
                ),
                chunk.index,
                chunk.size,
            )
            for chunk in remote_src_chunks
        ]
        tb_channel_id = get_program().setup_channel(tb, self)

        op = ReduceOperation(
            [LocalChunk(local_src_chunk.buffer, local_src_chunk.index, local_src_chunk.size)],
            [LocalChunk(local_dst_chunk.buffer, local_dst_chunk.index, local_dst_chunk.size)],
            remote_chunks,
            [],
            tb_channel_id,
            self.channel_type,
            reduce_op,
        )

        get_program().add_operation(self.src_rank, tb, op)


@dataclass
class SwitchChannel:
    __channel_counts = defaultdict(lambda: defaultdict(int))

    def __init__(self, rank_list: List[int], buffer_type: BufferType):
        num_ranks = get_program().num_ranks
        if not all(rank < num_ranks for rank in rank_list):
            raise RuntimeError(f"One or more ranks in {rank_list} are out of bounds. Number of ranks: {num_ranks}")

        self.channel_ids = {}
        for rank in rank_list:
            if rank >= num_ranks:
                raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {num_ranks}")
            self.channel_ids[rank] = SwitchChannel.__channel_counts[rank][buffer_type]
            SwitchChannel.__channel_counts[rank][buffer_type] += 1

        self.channel_type = ChannelType.switch
        self.buffer_type = buffer_type
        self.rank_group = RankGroup(len(rank_list), rank_list)

        get_program().add_channel(self)

    def group_load_reduce(self, buffer_offset, size, dst_chunk: Chunk, tb, reduce_op=ReduceOperationType.sum):
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

        tb_channel_id = get_program().setup_channel(tb, self)
        for i in range(len(self.rank_group.ranks)):
            op = GroupLoadReduce(
                self.buffer_type,
                buffer_offset,
                size,
                NVLSChunk(dst_chunk.rank, dst_chunk.buffer, dst_chunk.index, dst_chunk.size),
                [tb_channel_id[i]],
                self.channel_type,
                reduce_op,
            )
            get_program().add_operation(self.rank_group.ranks[i], tb, op)

    def group_store(self, src_chunk: Chunk, buffer_offset, size, tb, reduce_op=ReduceOperationType.sum):
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

        tb_channel_id = get_program().setup_channel(tb, self)
        for i in range(len(self.rank_group.ranks)):
            op = GroupStore(
                NVLSChunk(src_chunk.rank, src_chunk.buffer, src_chunk.index, src_chunk.size),
                self.buffer_type,
                buffer_offset,
                size,
                [tb_channel_id[i]],
                self.channel_type,
                reduce_op,
            )
            get_program().add_operation(self.rank_group.ranks[i], tb, op)
