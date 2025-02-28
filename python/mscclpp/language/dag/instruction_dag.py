# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import defaultdict
from mscclpp.language.buffer import Buffer
from mscclpp.language.types import (
    Channel,
    DataFormat,
    ChannelType,
    ChunkRef,
    Instruction,
    Op,
)


class InstructionDAG:
    def __init__(self, num_ranks: int, buffers: list):
        self.num_ranks = num_ranks
        self.buffers = buffers
        # State for the actual instruction DAG
        self.operations = {}  # slot -> operations
        self.last_writer = {}  # slot -> last writing op
        self.last_readers = defaultdict(list)  # slot -> list of last reading ops
        # State for the MSCCLPP-IR
        self.tbs = []
        for _ in range(num_ranks):
            self.tbs.append({})
        self.tb_mapping = {}
        self.num_channels = [1] * num_ranks
        self.tb_steps = [{} for _ in range(num_ranks)]

    def convert_set_list(self):
        ops = []
        visited = set()
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                op.next = list(op.next)
                for o in op.next:
                    ops.append(o)
            elif op.inst != Instruction.copy:
                ops.append(op)

            while len(ops) > 0:
                op = ops[0]
                if op not in visited:
                    visited.add(op)
                    op.next = list(op.next)
                    ops = ops[1:] + op.next
                else:
                    ops = ops[1:]
        return visited

    def complete_channels(self):
        send_op = [
            Instruction.put,
            Instruction.signal,
            Instruction.put_packet,
            Instruction.get,
            Instruction.read_put_packet,
        ]
        recv_op = [Instruction.wait, Instruction.read_reduce_copy]
        group_send_op = [Instruction.group_store]
        group_recv_op = [Instruction.group_load_reduce]
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                chans = set()
                for op in tb.ops:
                    if op.inst == Instruction.barrier:
                        continue
                    if op.src != None:
                        src_buffer = (
                            Buffer.scratch
                            if op.src.buffer is not Buffer.input and op.src.buffer is not Buffer.output
                            else op.src.buffer
                        )
                    if op.dst != None:
                        dst_buffer = (
                            Buffer.scratch
                            if op.dst.buffer is not Buffer.input and op.dst.buffer is not Buffer.output
                            else op.dst.buffer
                        )
                    if op.channel_type == ChannelType.nvls:
                        if op.inst in group_send_op:
                            ranks = [dst[0].rank for dst in op.dsts]
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, ranks)
                            chans.add(chan)
                        elif op.inst in group_recv_op:
                            ranks = [src[0].rank for src in op.srcs]
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, ranks)
                            chans.add(chan)
                    else:
                        if op.inst in send_op:
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, op.dst.rank)
                            chans.add(chan)
                        elif op.inst in recv_op:
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, op.src.rank)
                            chans.add(chan)
                tb.channels = list(chans)

    # InstructionDAG - builds the roots of the DAG
    def add_start(self, rank, buffer, index, ref):
        slot = (rank, buffer, index)
        op = Op(Instruction.start, rank, ref, ref, next=set(), prev=set())
        self.operations[slot] = op
        self.last_writer[slot] = op

    # InstructionDAG - adds a copy node
    def add_copy(self, rank, send_ref, recv_ref, tb, trans_from_packet=False, trans_to_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if trans_from_packet:
            op = Op(Instruction.copy_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        elif trans_to_packet:
            op = Op(
                Instruction.transform_to_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step
            )
        else:
            op = Op(Instruction.copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        # Sending part of copy [Read]
        self._read(rank, srcbuffer, srcindex, size, op)
        # Receiving part of copy [Write]
        self._write(rank, dstbuffer, dstindex, size, op)
        return op

    # InstructionDAG - adds a redduce node
    def add_reduce(self, rank, send_ref, recv_ref, tb, use_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if use_packet:
            op = Op(Instruction.reduce_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        else:
            op = Op(Instruction.reduce, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        # Sending part of reduce
        self._read(rank, srcbuffer, srcindex, size, op)
        # Reduce part of copy
        self._write(rank, dstbuffer, dstindex, size, op, read=True)
        return op

    # InstructionDAG - adds a put node
    def add_put(self, rank, send_ref, recv_ref, tb, src_format, ch_type, use_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if use_packet:
            if src_format == DataFormat.raw:
                op = Op(
                    Instruction.put_packet,
                    rank,
                    send_ref,
                    recv_ref,
                    next=set(),
                    prev=set(),
                    tb=tb,
                    channel_type=ch_type,
                    step=tb_step,
                )
            elif src_format == DataFormat.packet:
                op = Op(
                    Instruction.read_put_packet,
                    rank,
                    send_ref,
                    recv_ref,
                    next=set(),
                    prev=set(),
                    tb=tb,
                    channel_type=ch_type,
                    step=tb_step,
                )
        else:
            op = Op(
                Instruction.put,
                rank,
                send_ref,
                recv_ref,
                next=set(),
                prev=set(),
                tb=tb,
                channel_type=ch_type,
                step=tb_step,
            )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_get(self, rank, recv_ref, send_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.get, rank, recv_ref, send_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step
        )
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op)
        op.srcs.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.dsts.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    # InstructionDAG - adds a signal node.
    def add_signal(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.signal,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        # treat signal as a write. signal acts as a barrier for the next instruction which prevents the
        # below instructions to be scheduled above the signal instruction.
        self._write(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_flush(self, rank, send_ref, recv_ref, tb):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.flush,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ChannelType.port,
            step=tb_step,
        )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_wait(self, rank, dst_ref, src_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.wait, rank, src_ref, dst_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step
        )
        buffer = dst_ref.buffer
        index = dst_ref.index
        size = dst_ref.size
        self._write(rank, buffer, index, size, op)
        op.srcs.append((ChunkRef(src_ref.rank, src_ref.buffer, src_ref.index, src_ref.size), tb_step))
        op.dsts.append((ChunkRef(dst_ref.rank, dst_ref.buffer, dst_ref.index, dst_ref.size), tb_step))
        return op

    def add_read_reduce(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.read_reduce_copy,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        self._write(rank, buffer, index, size, op, read=True)
        return op

    def add_barrier(self, rank, tb_list, barrier_id):
        buffers = self.buffers[rank]
        for tb in tb_list:
            tb_step = self._get_tb_step(rank, tb)
            extra = {"tb_list": tb_list, "barrier_id": barrier_id}
            op = Op(Instruction.barrier, rank, None, None, next=set(), prev=set(), tb=tb, step=tb_step, extra=extra)
            for buffer_type, buffer in buffers.items():
                self._write(rank, buffer_type, 0, len(buffer), op)

    def add_group_load_reduce(self, rank, send_refs, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.group_load_reduce,
            rank,
            recv_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        # treat recv_ref as src for group_load_reduce
        op.srcs.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        for send_ref in send_refs:
            op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op, read=True)

    def add_group_store(self, rank, send_ref, recv_refs, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.group_store,
            rank,
            send_ref,
            send_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        # treat send_ref as dst for group_store
        op.dsts.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        for recv_ref in recv_refs:
            op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        return op

    def _get_tb_step(self, rank: int, tb: int):
        if tb in self.tb_steps[rank]:
            self.tb_steps[rank][tb] += 1
            return self.tb_steps[rank][tb]
        else:
            self.tb_steps[rank][tb] = 0
            return 0

    # InstructionDAG helper - identifies the dependencies for a write-type operation (recv, copy, rrc, reduce)
    def _write(self, rank, buffer, index, size, op, read=False):
        prev_ops = set()
        for i in range(index, index + size):
            slot = (rank, buffer, i)
            if read:
                assert slot in self.last_writer, f"Destination slot has never been written before a reduce {op}"

            # First write to this slot
            if slot not in self.operations:
                self.operations[slot] = op

            # If there are active readers - these are the previous operations
            # Else the previous operation is the last write (if there is one)
            readers = self.last_readers[slot]
            if len(readers) > 0:
                prev_ops.update(readers)
            elif slot in self.last_writer:
                prev_ops.add(self.last_writer[slot])

            # Set the last_writer to this op, and clear all readers
            self.last_writer[slot] = op
            self.last_readers[slot] = []

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG helper - identifies the dependencies for read-type operations (send, copy, reduce)
    def _read(self, rank, buffer, index, size, op):
        prev_ops = set()
        for i in range(index, index + size):
            slot = (rank, buffer, i)
            assert slot in self.last_writer, f"Slot has never been written before a read-type {op}"
            # The previous operation for a reader is the last write to the slot
            writer = self.last_writer[slot]
            prev_ops.add(writer)
            self.last_readers[slot].append(op)

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)
