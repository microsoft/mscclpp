# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.utils import (
    buf_dst_src_match,
    circular_dep_after_merge,
    merge_op,
    remove_op,
    same_chan_type,
    same_count,
    same_buf_dst,
    same_buf_src,
    same_src_dst_buffer_type,
    same_tb,
    all_prevs_visited_after_merge,
)
from mscclpp.language.dag.instruction_dag import InstructionDAG
from mscclpp.language.types import ChunkRef, ChannelType, Instruction, Op, Threadblock


class _InstructionOptimizer:
    def try_merge_same_instructions(
        self,
        op: Op,
        next_op: Op,
        tb: Threadblock,
        queue: list,
        expected_next_inst: Instruction,
        same_buf_func: callable,
    ) -> bool:
        """
        Attempts to merge two instruction if conditions are met.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param expected_next_inst: The instruction type expected for the next operation.
        :param same_buf_func: The function to check if the buffer is the same (same_buf_dst or same_buf_src).
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == expected_next_inst
            and same_tb(op, next_op)
            and same_buf_func(op, next_op)
            and same_count(op, next_op)
            and same_chan_type(op, next_op)
            and not circular_dep_after_merge(op, next_op)
            and all_prevs_visited_after_merge(op, next_op)
        ):
            # Append the source chunks from next_op
            op.srcs.append(
                (
                    ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size),
                    next_op.step,
                )
            )
            # For 'signal' and 'wait' instructions, append destination chunks too
            if expected_next_inst in [Instruction.signal, Instruction.wait, Instruction.flush]:
                op.dsts.append(
                    (
                        ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size),
                        next_op.step,
                    )
                )
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False

    def try_compact_instructions(
        self, op: Op, tb: Threadblock, queue: list, inst_type: Instruction, same_src_dst_func: callable
    ) -> bool:
        """
        Try to campact the instructions with the same instruction type. This optimization will
        compact multiple instructions of the same type into a single instruction.
        :param op: The current operation.
        :param seq_op: The sequential operation to merge with.
        :param tb: The task block containing the operations.
        :param queue: The queue of operations.
        :param inst_type: The type of the instruction being processed (get, put, put_packet).
        :return: True if operations are merged, False otherwise.
        """
        if len(queue) > 1:
            seq_op = queue[1]
            if (
                seq_op.inst == inst_type
                and same_src_dst_func(op, seq_op)
                and same_chan_type(op, seq_op)
                and same_count(op, seq_op)
                and not circular_dep_after_merge(op, seq_op)
                and all_prevs_visited_after_merge(op, seq_op)
            ):
                # Append the source and destination chunks from seq_op
                op.dsts.append(
                    (
                        ChunkRef(seq_op.dst.rank, seq_op.dst.buffer, seq_op.dst.index, seq_op.dst.size),
                        seq_op.step,
                    )
                )
                op.srcs.append(
                    (
                        ChunkRef(seq_op.src.rank, seq_op.src.buffer, seq_op.src.index, seq_op.src.size),
                        seq_op.step,
                    )
                )
                merge_op(op, seq_op)
                tb.ops.remove(seq_op)
                queue.remove(seq_op)
                return True
        return False

    def try_fuse_with_put(self, op: Op, next_op: Op, tb: Threadblock, queue: list) -> bool:
        """
        Attempts to fuse 'put' operations with other operations like read_reduce_copy, reduce, etc.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param inst_type: The type of the instruction being processed.
        :param chan_type: Channel type if applicable.
        :return: True if operations are merged, False otherwise.
        """
        if (
            (next_op.inst == Instruction.put or next_op.inst == Instruction.put_packet)
            and same_tb(op, next_op)
            and same_count(op, next_op)
            and buf_dst_src_match(op, next_op)
            and next_op.channel_type == ChannelType.memory
            and (op.channel_type == ChannelType.none or op.channel_type == ChannelType.memory)
            and not circular_dep_after_merge(op, next_op)
            and all_prevs_visited_after_merge(op, next_op)
        ):
            if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                return False
            # Adjust instruction type and channel if needed
            if op.inst == Instruction.read_reduce_copy:
                op.inst = Instruction.read_reduce_copy_send
            elif op.inst == Instruction.reduce:
                op.inst = Instruction.reduce_send
                op.channel_type = ChannelType.memory
            elif op.inst == Instruction.reduce_packet:
                op.inst = Instruction.reduce_send_packet
                op.channel_type = ChannelType.memory
            # Append the destination chunk from next_op
            op.dsts.append(
                (
                    ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size),
                    next_op.step,
                )
            )
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False

    def try_fuse_instructions_using_port_channel(
        self, op: Op, next_op: Op, tb: Threadblock, queue: list, expected_next_inst: Instruction
    ) -> bool:
        """
        Attempts to fuse operations which using port channel.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param expected_next_inst: The instruction type expected for the next operation.
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == expected_next_inst
            and same_tb(op, next_op)
            and same_count(op, next_op)
            and same_buf_dst(op, next_op)
            and same_buf_src(op, next_op)
            and same_chan_type(op, next_op)
            and op.channel_type == ChannelType.port
            and not circular_dep_after_merge(op, next_op)
            and all_prevs_visited_after_merge(op, next_op)
        ):
            if op.inst == Instruction.put and next_op.inst == Instruction.signal:
                op.inst = Instruction.put_with_signal
            elif op.inst == Instruction.put_with_signal and next_op.inst == Instruction.flush:
                op.inst = Instruction.put_with_signal_and_flush
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False

    def try_fuse_with_group_store(self, op: Op, next_op: Op, tb: Threadblock, queue: list) -> bool:
        """
        Attempts to fuse 'gruop_load_reduce' operations with 'group_store' operations.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == Instruction.group_store
            and same_count(op, next_op)
            and buf_dst_src_match(op, next_op)
            and same_chan_type(op, next_op)
            and not circular_dep_after_merge(op, next_op)
            and all_prevs_visited_after_merge(op, next_op)
        ):
            # Append the destination chunk from next_op
            op.inst = Instruction.group_load_reduce_store
            op.src = next_op.src
            for dst in next_op.dsts:
                op.dsts.append(dst)
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False

    def try_remove_op(self, pending_remove_op: Op, condition: bool) -> bool:
        if condition:
            remove_op(pending_remove_op)
            return True
        return False


class DagOptimizer:
    def __init__(self, instruction_dag: InstructionDAG):
        self.optimizer = _InstructionOptimizer()
        self.dag = instruction_dag

    def remove_redundant_signal_wait(self):
        # For packet ops, we can remove signal/wait
        for rank, rank_tbs in enumerate(self.dag.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst == Instruction.put_packet:
                        for next_op in op.next:
                            fused = self.optimizer.try_remove_op(next_op, next_op.inst == Instruction.signal)
                            if fused:
                                break
                    elif op.inst == Instruction.reduce_packet or op.inst == Instruction.copy_packet:
                        for prev_op in op.prev:
                            fused = self.optimizer.try_remove_op(prev_op, prev_op.inst == Instruction.wait)
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    def fuse_instructions(self):
        self._fuse_instructions_using_port_channel()
        self._fuse_same_instructions()
        self._optimize_rrcs_rs()
        self._optimize_group_ops()
        self._compact_instructions()

    # put(src, sbuf, si, dst, dbuf, di) signal(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignal(src, sbuf, si, dst, dbuf, di)
    # put(src, sbuf, si, dst, dbuf, di) signal(src, sbuf, si, dst, dbuf, di) flush(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    def _fuse_instructions_using_port_channel(self):
        inst_followup_map = {
            Instruction.put: Instruction.signal,
            Instruction.put_with_signal: Instruction.flush,
        }
        for rank, rank_tbs in enumerate(self.dag.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in inst_followup_map:
                        for next_op in op.next:
                            fused = self.optimizer.try_fuse_instructions_using_port_channel(
                                op, next_op, tb, queue, inst_followup_map[op.inst]
                            )
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) rrc(_,_,_,dst,dbuf,di) -> rrc(list[src,sbuf,si], dst, dbuf, di)
    # signal(_,_,_,dst,dbuf,di) signal(_,_,_,dst,dbuf,di) -> signal(_,_,_,list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    # reduce(_,_,_,dst,dbuf,di) reduce(_,_,_,dst,dbuf,di) -> reduce(list[src,sbuf,si], dst, dbuf, di)
    # reduce_packet(_,_,_,dst,dbuf,di) reduce_packet(_,_,_,dst,dbuf,di) -> reduce_packet(list[src,sbuf,si], dst, dbuf, di)
    def _fuse_same_instructions(self):
        # Mapping instruction to their respective condition checks and same buffer function
        instruction_handlers = {
            Instruction.read_reduce_copy: same_buf_dst,
            Instruction.reduce: same_buf_dst,
            Instruction.reduce_packet: same_buf_dst,
            Instruction.signal: same_buf_src,
            Instruction.wait: same_buf_dst,
        }

        for _, rank_tbs in enumerate(self.dag.tbs):
            for _, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    inst_type = op.inst
                    if inst_type in instruction_handlers:
                        for next_op in op.next:
                            same_buf_func = instruction_handlers[inst_type]
                            if self.optimizer.try_merge_same_instructions(
                                op, next_op, tb, queue, inst_type, same_buf_func
                            ):
                                fused = True
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rrcs(_,_,_,_,_,_)
    # reduce(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rs(_,_,_,_,_,_)
    def _optimize_rrcs_rs(self):
        inst_types = [
            Instruction.read_reduce_copy,
            Instruction.reduce,
            Instruction.reduce_packet,
            Instruction.read_reduce_copy_send,
            Instruction.reduce_send,
            Instruction.reduce_send_packet,
        ]
        for _, rank_tbs in enumerate(self.dag.tbs):
            for _, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in inst_types:
                        for next_op in op.next:
                            fused = self.optimizer.try_fuse_with_put(op, next_op, tb, queue)
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # glre(srcs, sbuf, si, _, _, _), gstore (_, _, _, dsts, dbuf, di) -> glres(srcs, sbuf, si, dsts, dbuf, di)
    def _optimize_group_ops(self):
        inst_types = [
            Instruction.group_load_reduce,
        ]
        for _, rank_tbs in enumerate(self.dag.tbs):
            for _, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in inst_types:
                        for next_op in op.next:
                            fused = self.optimizer.try_fuse_with_group_store(op, next_op, tb, queue)
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # merge ops which are independent of other operations and no other operations in between
    # get(src, sbuf. si, dst, dbuf, di) get(src, sbuf, si, dst, dbuf, di) -> get(list[src,sbuf,si], list[dst,dbuf,di])
    # put(src, sbuf, si, dst, dbuf, di) put(src, sbuf, si, dst, dbuf, di) -> put(list[src,sbuf,si], list[dst,dbuf,di])
    # putWithSignal/putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    # putWithSignal/putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignal/putWithSignalAndFlush(list[src,sbuf,si], list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    def _compact_instructions(self):
        campactable_inst = [
            Instruction.get,
            Instruction.put,
            Instruction.put_packet,
            Instruction.put_with_signal,
            Instruction.put_with_signal_and_flush,
            Instruction.signal,
            Instruction.flush,
            Instruction.wait,
        ]
        for _, rank_tbs in enumerate(self.dag.tbs):
            for _, tb in rank_tbs.items():
                if tb.id == -1:
                    continue
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in campactable_inst:
                        fused = self.optimizer.try_compact_instructions(
                            op, tb, queue, op.inst, same_src_dst_buffer_type
                        )

                    if fused:
                        continue
                    queue = queue[1:]
