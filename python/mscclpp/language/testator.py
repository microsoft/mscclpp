# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from mscclpp.language.collectives import Collective
from mscclpp.language.buffer import *
from mscclpp.language.types import DataFormat, ChannelType, ChunkRef, ReplicationPolicy, Threadblock
from mscclpp.language.ir import *
from mscclpp.language.dag import DagOptimizer, DagLower, InstructionDAG
from mscclpp.language.rank import Rank

# For msccl++ program, we have one assumption that for channel can be identified by (send_buffer, recv_buffer, type, send_tb/recv_tb)
# which means the send_tb and recv_tb should be the same for a pair of signal and wait, also same for put/get operation.
# If one sender what to send data to peer want to use different tb in receiver side. We need to send to same tb in receiver side first,
# then performance a across tb sync. This is a limitation of current implementation.
class Testador:
    def __init__(
        self,
        collective: Collective,
        num_ranks: int,
    ):
        self.collective = collective
        self.num_ranks = num_ranks
        self.buffers = collective.init_buffers()

    # Tracks a send operation on the buffers
    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

    # Tracks a reduce operation on the buffers
    def apply_reduce(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            reduce_chunk = db[dst_index + i]
            sent_chunk = sb[src_index + i]
            db[dst_index + i] = reduce_chunk.reduce(dst, sent_chunk)

    def check_buffer_exists(self, rank, name):
        if name not in self.buffers[rank]:
            self.buffers[rank][name] = BufferSlice(Buffer.scratch, name)

    def _get_buffer_index(self, src_rank, remote_rank, buffer, index):
        if index == -1 and buffer == None:
            return src_rank.buffer, src_rank.index
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            return buffer, self.buffers[remote_rank][buffer].instance_size()
        return buffer, index

    def _put(
        self,
        src,
        dst,
        buffer=None,
        index=-1,
        sendtb=-1,
        src_format=DataFormat.raw,
        chan_type=ChannelType.memory,
        use_packet=False,
    ):
        self.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(src, dst, buffer, index)
        self.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)

    def put(self, src, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.memory):
        return self._put(src, dst, buffer, index, sendtb, DataFormat.raw, chan_type)

    def put_packet(
        self,
        src,
        dst,
        buffer=None,
        index=-1,
        sendtb=-1,
        src_format=DataFormat.raw,
        chan_type=ChannelType.memory,
        temp_buffer=None,
        temp_buffer_index=-1,
    ):
        chunk_ref = src
        if chan_type == ChannelType.port and src_format == DataFormat.raw:
            chunk_ref = self._copy(
                src.rank, temp_buffer, temp_buffer_index, sendtb, trans_from_packet=False, trans_to_packet=True
            )
        return self._put(chunk_ref, dst, buffer, index, sendtb, src_format, chan_type, True)

    def get(self, dst, src, buffer=None, index=-1, recvtb=-1, chan_type=ChannelType.memory):
        self.check_buffer_exists(src, buffer)
        buffer, index = self._get_buffer_index(dst, src, buffer, index)
        self.apply_send(src, buffer, index, dst.rank, dst.buffer, dst.index, dst.size)

    def _copy(self, src, dst, buffer=None, index=-1, sendtb=-1, trans_from_packet=False, trans_to_packet=False):
        self.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(src, dst, buffer, index)
        self.apply_send(src.rank, src.buffer, src.index, dst, buffer, index, src.size)

    def copy(self, dst, buffer=None, index=-1, sendtb=-1):
        return self._copy(dst, buffer, index, sendtb)

    def copy_packet(self, dst, buffer=None, index=-1, sendtb=-1):
        return self._copy(dst, buffer, index, sendtb, trans_from_packet=True, trans_to_packet=False)

    def _reduce(self, src, other_chunkref, recvtb=-1, channel_type=ChannelType.memory, use_packet=False):
        dst = src.rank
        src = other_chunkref.rank
        self.apply_reduce(
            src, other_chunkref.buffer, other_chunkref.index, dst, src.buffer, src.index, src.size
        )

    def reduce(self, src, other_chunkref, recvtb=-1, channel_type=ChannelType.memory):
        return self._reduce(src, other_chunkref, recvtb, channel_type)
    
    def reduce_packet(self, src, other_chunkref, recvtb=-1):
        return self._reduce(src, other_chunkref, recvtb, use_packet=True)

    def group_load_reduce(self, src, other_chunkrefs: list, recvtb=-1, chan_type=ChannelType.nvls):
        for other_chunkref in other_chunkrefs:
            src_chunkref = other_chunkref
            self.apply_reduce(
                src_chunkref.rank,
                src_chunkref.buffer,
                src_chunkref.index,
                src.rank,
                src.buffer,
                src.index,
                src.size,
            )

    def group_store(self, src, dsts: list, index=-1, buffer=None, sendtb=-1, chan_type=ChannelType.nvls):
        for dst in dsts:
            self.check_buffer_exists(dst, buffer)
        
        for dst in dsts:
            buffer, index = self._get_buffer_index(src, dst, buffer, index)
            self.apply_send(src.rank, src.buffer, src.index, dst, buffer, index, src.size)