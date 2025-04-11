from mscclpp.language.types import *
from mscclpp.language.program import chunk_exec, rank
from queue import Queue
from typing import Dict, Tuple


class SortDAG:
    """
    A DAG structure to enforce correct execution order of collective communication operations.
    Supports topological sorting based on rank/threadblock execution and signal/wait synchronization.
    """

    start_nodes: list["Node"]
    last_node: Dict[Tuple[int, int], int]
    signalling: Dict[Tuple[int, int, int], Queue]
    waiting: Dict[Tuple[int, int, int], Queue]

    def __init__(self):
        self.start_nodes = []
        self.last_node = {}
        self.signalling = {}
        self.waiting = {}

    def insert_operation(self, op: "Op"):
        """
        Inserts an operation into the DAG, adding edges based on dependencies.
        """
        node = self.Node(op)
        rank = op.rank
        tb = op.tb

        if op.inst == Instruction.barrier:
            for tb in op.extra.get("tb_list", None):
                if (rank, tb) not in self.last_node:
                    self.last_node[(rank, tb)] = node
                    self.start_nodes.append(node)
                else:
                    prev_node = self.last_node[(rank, tb)]
                    prev_node.next_nodes.append(node)
                    node.input += 1
                    self.last_node[(rank, tb)] = node

        else:
            if (rank, tb) not in self.last_node:
                self.last_node[(rank, tb)] = node
                self.start_nodes.append(node)
            else:
                prev_node = self.last_node[(rank, tb)]
                prev_node.next_nodes.append(node)
                node.input += 1
                self.last_node[(rank, tb)] = node

        if op.inst == Instruction.signal:
            if (op.src.rank, op.dst.rank, tb) not in self.waiting or self.waiting[
                (op.src.rank, op.dst.rank, tb)
            ].empty():
                if (op.src.rank, op.dst.rank, tb) not in self.signalling:
                    self.signalling[(op.src.rank, op.dst.rank, tb)] = Queue()
                self.signalling[(op.src.rank, op.dst.rank, tb)].put(node)
            else:
                waiting_node = self.waiting[(op.src.rank, op.dst.rank, tb)].get()
                node.next_nodes.append(waiting_node)
                waiting_node.input += 1

        if op.inst == Instruction.wait:
            if (op.src.rank, op.dst.rank, tb) not in self.signalling or self.signalling[
                (op.src.rank, op.dst.rank, tb)
            ].empty():
                if (op.src.rank, op.dst.rank, tb) not in self.waiting:
                    self.waiting[(op.src.rank, op.dst.rank, tb)] = Queue()
                self.waiting[(op.src.rank, op.dst.rank, tb)].put(node)
            else:
                signalling_node = self.signalling[(op.src.rank, op.dst.rank, tb)].get()
                signalling_node.next_nodes.append(node)
                node.input += 1

    def execute_operations(self):
        """
        Executes operations in topological order, honoring dependency constraints.
        """
        queue = Queue()
        for node in self.start_nodes:
            queue.put(node)

        while not queue.empty():
            node = queue.get()
            op = node.operation

            if op.inst == Instruction.put:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.put_exec(op.dst.rank, op.dst.buffer, op.dst.index, sendtb=op.tb, chan_type=op.channel_type)
            elif op.inst == Instruction.put_packet:
                src_format = op.extra.get("src_format")
                temp_buffer = op.extra.get("temp_buffer")
                temp_buffer_index = op.extra.get("temp_buffer_index")
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.put_packet_exec(
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
                c = chunk_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size)
                c.get_exec(op.src.rank, op.src.buffer, op.src.index, op.tb, op.channel_type)
            elif op.inst == Instruction.flush:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.flush_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.tb, op.channel_type)
            elif op.inst == Instruction.wait:
                c = chunk_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size)
                c.wait_exec(op.src.rank, op.src.buffer, op.src.index, op.tb, op.channel_type)
            elif op.inst == Instruction.signal:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.signal_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.tb, op.channel_type)
            elif op.inst == Instruction.copy:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.copy_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.tb)
            elif op.inst == Instruction.copy_packet:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.copy_packet_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.tb)
            elif op.inst == Instruction.reduce:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.reduce_exec(
                    chunk_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size),
                    recvtb=op.tb,
                    channel_type=op.channel_type,
                )
            elif op.inst == Instruction.reduce_packet:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.reduce_packet_exec(chunk_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size), recvtb=op.tb)
            elif op.inst == Instruction.group_load_reduce:
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.group_load_reduce_exec(
                    chunk_exec(op.dst.rank, op.dst.buffer, op.dst.index, op.dst.size),
                    recvtb=op.tb,
                    channel_type=op.channel_type,
                )
            elif op.inst == Instruction.group_store:
                dsts = op.extra.get("dsts")
                index = op.extra.get("index")
                buffer = op.extra.get("buffer")
                c = chunk_exec(op.src.rank, op.src.buffer, op.src.index, op.src.size)
                c.group_store_exec(dsts=dsts, index=index, buffer=buffer, sendtb=op.tb, channel_type=op.channel_type)
            elif op.inst == Instruction.barrier:
                r = rank(op.rank)
                r.barrier_exec(op.extra.get("tb_list"))

            for next_node in node.next_nodes:
                next_node.reach += 1
                if next_node.reach == next_node.input:
                    queue.put(next_node)

    class Node:
        operation: "Op"
        next_nodes: list
        input: int
        reach: int

        def __init__(self, operation: "Op"):
            self.operation = operation
            self.next_nodes = []
            self.input = 0
            self.reach = 0
