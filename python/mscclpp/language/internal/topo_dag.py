# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.operations import *
from mscclpp.language.internal.dsl_types import *
from mscclpp.language.internal.channel_register import ChannelRegister
from queue import Queue
from typing import Dict, Tuple


class OperationDependencyGraph:
    """
    A DAG structure to enforce correct execution order of collective communication operations.
    Supports topological sorting based on rank/threadblock execution and signal/wait synchronization.
    """

    def __init__(self):
        self.root_nodes: Set[OperationDependencyGraph.Node] = set()
        self.previous_node: Dict[Tuple[int, int], int] = {}
        self.signalling: Dict[Tuple[int, int, int], Queue] = {}
        self.waiting: Dict[Tuple[int, int, int], Queue] = {}

        self.barrier_nodes: Dict[Tuple[int, int], List[OperationDependencyGraph.Node]] = {}
        self.tb_barriers: Dict[Tuple[int, int, int], int] = {}

    def add_operation(self, operation):
        """
        Inserts an operation into the DAG, adding edges based on dependencies.
        """
        rank = operation.rank
        threadblock = operation.threadblock

        if isinstance(operation, BarrierOperation):
            if (rank,  threadblock, operation.barrier_id) not in self.tb_barriers:
                self.tb_barriers[(rank, threadblock, operation.barrier_id)] = 0
            if (rank, operation.barrier_id) not in self.barrier_nodes:
                self.barrier_nodes[(rank, operation.barrier_id)] = []

            barrier_count = self.tb_barriers[(rank, threadblock, operation.barrier_id)]
            if barrier_count > len(self.barrier_nodes[(rank, operation.barrier_id)]):
                raise RuntimeError(f"Barrier node not create correctly for rank {rank}, threadblock {threadblock}, barrier_id {operation.barrier_id}.")
            elif barrier_count == len(self.barrier_nodes[(rank, operation.barrier_id)]):
                node = self.Node(operation)
                self.barrier_nodes[(rank, operation.barrier_id)].append(node)
            else:
                node = self.barrier_nodes[(rank, operation.barrier_id)][barrier_count]
            
            self.tb_barriers[(rank, threadblock, operation.barrier_id)] += 1    
        else:
            node = self.Node(operation)

        if (rank, threadblock) not in self.previous_node:
            self.previous_node[(rank, threadblock)] = node
            if node.input == 0:
                self.root_nodes.add(node)
        else:
            prev_node = self.previous_node[(rank, threadblock)]
            prev_node.next_nodes.append(node)
            node.input += 1
            self.previous_node[(rank, threadblock)] = node
            if node in self.root_nodes:
                self.root_nodes.remove(node)

        if isinstance(operation, SignalOperation):
            for tb_channel_id in operation.channel_ids:
                channel = ChannelRegister.get_channel(rank, threadblock, tb_channel_id)
                op_info = (channel.src_rank, channel.dst_rank, channel.channel_id)
                if op_info not in self.waiting or self.waiting[op_info].empty():
                    if op_info not in self.signalling:
                        self.signalling[op_info] = Queue()
                    self.signalling[op_info].put(node)
                else:
                    waiting_node = self.waiting[op_info].get()
                    node.next_nodes.append(waiting_node)
                    waiting_node.input += 1

        if isinstance(operation, WaitOperation):
            for tb_channel_id in operation.channel_ids:
                channel = ChannelRegister.get_channel(rank, threadblock, tb_channel_id)
                op_info = (channel.dst_rank, channel.src_rank, channel.channel_id)
                if op_info not in self.signalling or self.signalling[op_info].empty():
                    if op_info not in self.waiting:
                        self.waiting[op_info] = Queue()
                    self.waiting[op_info].put(node)
                else:
                    signalling_node = self.signalling[op_info].get()
                    signalling_node.next_nodes.append(node)
                    node.input += 1

    def check(self):
        """
        Validates the DAG structure, ensuring all nodes are reachable and dependencies are correctly set.
        """
        if len(self.signalling) > 0:
            for key, queue in self.signalling.items():
                if not queue.empty():
                    raise RuntimeError(f"Signalling from {key[0]} to {key[1]} on channel {key[2]} hasn't equivalent wait operation.")
        if len(self.waiting) > 0:
            for key, queue in self.waiting.items():
                if not queue.empty():
                    raise RuntimeError(f"Waiting for {key[0]} to {key[1]} on channel {key[2]} hasn't equivalent signal operation.")
            

    def get_execution_order(self):
        """
        Returns the order of operations in the DAG.
        """
        self.check()
        
        order = []
        queue = Queue()
        for node in self.root_nodes:
            queue.put(node)

        while not queue.empty():
            node = queue.get()
            op = node.operation
            order.append(op)
            print(f"Operation {op.name} on rank {op.rank} threadblock {op.threadblock}")
            for next_node in node.next_nodes:
                print(f"  -> Next operation {next_node.operation.name} on rank {next_node.operation.rank} threadblock {next_node.operation.threadblock}")
                next_node.reach += 1
                if next_node.reach == next_node.input:
                    queue.put(next_node)

        return order

    class Node:
        def __init__(self, operation):
            self.operation = operation
            self.next_nodes = []
            self.input = 0
            self.reach = 0