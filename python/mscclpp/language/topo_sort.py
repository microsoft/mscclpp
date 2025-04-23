from mscclpp.language.types import *
from queue import Queue
from typing import Dict, Tuple


class SortDAG:
    """
    A DAG structure to enforce correct execution order of collective communication operations.
    Supports topological sorting based on rank/threadblock execution and signal/wait synchronization.
    """

    start_nodes: List["Node"]
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

    def operation_order(self):
        """
        Returns the order of operations in the DAG.
        """
        order = []
        queue = Queue()
        for node in self.start_nodes:
            queue.put(node)

        while not queue.empty():
            node = queue.get()
            op = node.operation
            order.append(op)
            for next_node in node.next_nodes:
                next_node.reach += 1
                if next_node.reach == next_node.input:
                    queue.put(next_node)

        return order

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