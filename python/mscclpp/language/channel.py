from mscclpp.language.internal.channel import BaseChannel
from mscclpp.language.internal.types import RemoteBuffer
from mscclpp.language.internal.globals import get_program
from dataclasses import dataclass
from collections import defaultdict
from mscclpp.language.internal.operations import *


@dataclass
class Channel(BaseChannel):
    __channel_counts = defaultdict(lambda: defaultdict(int))

    def __init__(self, dst_rank, src_rank, channel_type):
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

    def signal(self, tb, sync):
        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation([tb_channel_id], self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

    def wait(self, tb, sync):
        tb_channel_id = get_program().setup_channel(tb, self)
        op = WaitOperation([tb_channel_id], self.channel_type)
        get_program().add_operation(self.src_rank, tb, op)

    def relaxed_signal(self, tb, sync):
        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation([tb_channel_id], self.channel_type, True)
        get_program().add_operation(self.src_rank, tb, op)

    def relaxed_wait(self, tb, sync):
        tb_channel_id = get_program().setup_channel(tb, self)
        op = SignalOperation([tb_channel_id], self.channel_type, True)
        get_program().add_operation(self.src_rank, tb, op)

    def put(self, dst_chunk, src_chunk, tb):
        remote_chunk = RemoteBuffer(dst_chunk.rank, dst_chunk.buffer, self.channel_type)
        tb_chunk_id = get_program().setup_remote_chunk(tb, remote_chunk)
        tb_channel_id = get_program().setup_channel(tb, self)

        op = PutOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            [RemoteChunk(tb_chunk_id, dst_chunk.index, dst_chunk.size)],
            [tb_channel_id],
            self.channel_type,
        )

        get_program().add_operation(self.src_rank, tb, op)
