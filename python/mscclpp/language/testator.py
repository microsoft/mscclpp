# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from mscclpp.language.collectives import Collective
from mscclpp.language.buffer import *
from mscclpp.language.types import DataFormat, ChannelType, ChunkRef, ReplicationPolicy, Threadblock
from mscclpp.language.ir import *


class Testator:
    def __init__(self):
        self.collective = None
        self.buffers = None

    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

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

    def _put(self, src, dst, buffer=None, index=-1):
        self.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(src, dst, buffer, index)
        self.apply_send(src.rank, src.buffer, src.index, dst, buffer, index, src.size)

    def _put_packet(
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
            self._copy(src.rank, temp_buffer, temp_buffer_index)
        self._put(chunk_ref, dst, buffer, index)

    def _get(self, dst, src, buffer=None, index=-1):
        self.check_buffer_exists(src, buffer)
        buffer, index = self._get_buffer_index(dst, src, buffer, index)
        self.apply_send(src, buffer, index, dst.rank, dst.buffer, dst.index, dst.size)

    def _copy(self, src, dst, buffer=None, index=-1):
        self.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(src, dst, buffer, index)
        self.apply_send(src.rank, src.buffer, src.index, dst, buffer, index, src.size)

    def _reduce(self, src, other_chunkref):
        dst = src.rank
        src_rank = other_chunkref.rank
        self.apply_reduce(src_rank, other_chunkref.buffer, other_chunkref.index, dst, src.buffer, src.index, src.size)

    def _group_load_reduce(self, src, other_chunkrefs: list):
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

    def _group_store(self, src, dsts: list, index=-1, buffer=None):
        for dst in dsts:
            self.check_buffer_exists(dst, buffer)

        for dst in dsts:
            buffer, index = self._get_buffer_index(src, dst, buffer, index)
            self.apply_send(src.rank, src.buffer, src.index, dst, buffer, index, src.size)

    def _execute(self, operations):
        for op in operations:
            if op.inst == Instruction.put:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._put(src, op.dst.rank, op.dst.buffer, op.dst.index)
            elif op.inst == Instruction.put_packet:
                src_format = op.extra.get("src_format")
                temp_buffer = op.extra.get("temp_buffer")
                temp_buffer_index = op.extra.get("temp_buffer_index")
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._put_packet(
                    src,
                    op.dst.rank,
                    op.dst.buffer,
                    op.dst.index,
                    sendtb=op.tb,
                    src_format=src_format,
                    chan_type=op.channel_type,
                    temp_buffer=temp_buffer,
                    temp_buffer_index=temp_buffer_index,
                )
            elif op.inst == Instruction.get:
                dst = ChunkRef(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size)
                self._get(dst, op.src.rank, op.src.buffer, op.src.index)
            elif op.inst == Instruction.copy:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._copy(src, op.dst.rank, op.dst.buffer, op.dst.index)
            elif op.inst == Instruction.copy_packet:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._copy(src, op.dst.rank, op.dst.buffer, op.dst.index)
            elif op.inst == Instruction.reduce:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._reduce(src, ChunkRef(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size))
            elif op.inst == Instruction.reduce_packet:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._reduce(src, ChunkRef(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size))
            elif op.inst == Instruction.group_load_reduce:
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._group_load_reduce(src, other_chunkrefs=op.srcs)
            elif op.inst == Instruction.group_store:
                dsts = op.extra.get("dsts")
                index = op.extra.get("index")
                buffer = op.extra.get("buffer")
                src = ChunkRef(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                self._group_store(src, dsts=dsts, index=index, buffer=buffer)

    def check(self, collective: Collective, operations: List):
        self.collective = collective
        self.buffers = collective.init_buffers()
        self._execute(operations)
        return collective.check(self)
