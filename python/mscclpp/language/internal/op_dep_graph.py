# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.globals import *
from mscclpp.language.internal.operations import *
from mscclpp.language.internal.types import *
from mscclpp.language.internal.register import ChannelRegister, SemaphoreRegister
from queue import Queue
from typing import Set, Dict, Tuple
import warnings


class OperationDependencyGraph:
    """
    A DAG structure to enforce correct execution order of collective communication operations.
    Supports topological sorting based on rank/threadblock execution and signal/wait synchronization.
    """

    def __init__(self):
        self.root_nodes: Set[OperationDependencyGraph.Node] = set()
        self.last_node: Dict[Tuple[int, int], int] = {}
        self.signalling: Dict[Tuple[int, int, int], Queue] = {}
        self.waiting: Dict[Tuple[int, int, int], Queue] = {}

        self.barrier_nodes: Dict[Tuple[int, int], List[OperationDependencyGraph.Node]] = {}
        self.tb_barriers: Dict[Tuple[int, int, int], int] = {}
        self.node_list = []
    
    def add_operation(self, operation, agg_node = None):
        """
        Inserts an operation into the DAG, adding edges based on dependencies.
        """
        rank = operation.rank
        threadblock = operation.threadblock
        node = self.Node(operation)
        if agg_node is not None:
            agg_node.add_node(node)

        if isinstance(operation, BarrierOperation):
            if (rank,  threadblock, operation.barrier_id) not in self.tb_barriers:
                self.tb_barriers[(rank, threadblock, operation.barrier_id)] = 0
            if (rank, operation.barrier_id) not in self.barrier_nodes:
                self.barrier_nodes[(rank, operation.barrier_id)] = []

            barrier_count = self.tb_barriers[(rank, threadblock, operation.barrier_id)]
            if barrier_count > len(self.barrier_nodes[(rank, operation.barrier_id)]):
                raise RuntimeError(f"Barrier node not create correctly for rank {rank}, threadblock {threadblock}, barrier_id {operation.barrier_id}.")
            elif barrier_count == len(self.barrier_nodes[(rank, operation.barrier_id)]):
                agg_node = self.AggregateNode()
                self.barrier_nodes[(rank, operation.barrier_id)].append(agg_node)
            else:
                agg_node = self.barrier_nodes[(rank, operation.barrier_id)][barrier_count]

            self.tb_barriers[(rank, threadblock, operation.barrier_id)] += 1
            agg_node.add_node(node)
            node = agg_node

        self.node_list.append(node)
        if (rank, threadblock) not in self.last_node:
            self.last_node[(rank, threadblock)] = node
            if node.get_input() == 0:
                self.root_nodes.add(node)
        else:
            prev_node = self.last_node[(rank, threadblock)]
            if prev_node is not node:
                prev_node.next_nodes.append(node)
                node.previous_nodes.append(prev_node)
                node.add_input()
                self.last_node[(rank, threadblock)] = node
                if node in self.root_nodes:
                    self.root_nodes.remove(node)

        if isinstance(operation, SignalOperation) or (isinstance(operation, PutOperation) and (operation.with_signal or operation.with_signal_and_flush)):
            for tb_channel_id in operation.channel_ids:
                channel = ChannelRegister.get_channel(rank, threadblock, tb_channel_id)
                op_info = (channel.src_rank, channel.dst_rank, channel.channel_peer_id)
                if op_info not in self.waiting or self.waiting[op_info].empty():
                    if op_info not in self.signalling:
                        self.signalling[op_info] = Queue()
                    self.signalling[op_info].put(node)
                else:
                    waiting_node = self.waiting[op_info].get()
                    node.next_nodes.append(waiting_node)
                    waiting_node.previous_nodes.append(node)
                    waiting_node.add_input()

        if isinstance(operation, WaitOperation):
            for tb_channel_id in operation.channel_ids:
                channel = ChannelRegister.get_channel(rank, threadblock, tb_channel_id)
                op_info = (channel.dst_rank, channel.src_rank, channel.channel_peer_id)
                if op_info not in self.signalling or self.signalling[op_info].empty():
                    if op_info not in self.waiting:
                        self.waiting[op_info] = Queue()
                    self.waiting[op_info].put(node)
                else:
                    signalling_node = self.signalling[op_info].get()
                    signalling_node.next_nodes.append(node)
                    node.previous_nodes.append(signalling_node)
                    node.add_input()

        return node

    def add_tbg_operation(self, operations):
        agg_node = self.AggregateNode()
        for operation in operations:
            self.add_operation(operation, agg_node)

    def add_semaphore_dependency(self):
        queue = Queue()
        processed_node = set()
        sem_rel = {}
        sem_acq = {}
        sem_val = {}

        self.reset()

        def compute_sem_op(sem_op, node):
            operation = node.operation
            for id in operation.semaphore_ids:
                if (operation.rank, id) not in sem_op:
                    sem_op[(operation.rank, id)] = []
                    sem_val[(operation.rank, id)] = SemaphoreRegister.get_semaphore(operation.rank, id).initial_value
                sem_op[(operation.rank, id)].append((node, operation.pipeline_context))
            
            return True

        def process_node(node):
            if node in processed_node:
                return
            processed_node.add(node)
            
            for next_node in node.next_nodes:
                next_node.add_reach()
                if next_node.get_reach() == next_node.get_input():
                    if isinstance(next_node, self.Node) and next_node.agg_node is not None:
                        for sub_node in next_node.agg_node.nodes:
                            queue.put(sub_node)
                    else:
                       queue.put(next_node) 

        for node in self.root_nodes:
            queue.put(node)

        while True:
            sem_ops_found = False
            new_sem_rel_node = []
            while not queue.empty():
                node = queue.get()
                if isinstance(node, self.Node) and isinstance(node.operation, SemaphoreReleaseOperation):
                    sem_ops_found = compute_sem_op(sem_rel, node)
                    new_sem_rel_node.append(node)
                elif isinstance(node, self.Node) and isinstance(node.operation, SemaphoreAcquireOperation):
                    sem_ops_found = compute_sem_op(sem_acq, node)
                else:
                    process_node(node)

            if not sem_ops_found:
                break
            else:
                removed_keys = []
                for key in sem_acq.keys():
                    if key not in sem_rel:
                        sem_rel[key] = []
                    if len(sem_acq[key]) > 1 or sem_val[key] < len(sem_rel[key]) - len(sem_acq[key]):
                        get_program().disable_inter_tb_sync()
                        warnings.warn(f"Undefined Behaviour Semaphore Id.", UserWarning)
                        return
                    
                    for sem_rel_node in new_sem_rel_node:
                        process_node(sem_rel_node)
                    
                    if sem_val[key] == len(sem_rel[key]) - len(sem_acq[key]):
                        sem_acq_node, sem_acq_ctx = sem_acq[key][0]
                        sem_val[key] = 0
                        if sem_acq_node in self.root_nodes:
                            self.root_nodes.remove(sem_acq_node)
                        process_node(sem_acq_node)
                        for sem_rel_node, sem_rel_ctx in sem_rel[key]:
                            if sem_rel_ctx is not sem_acq_ctx:
                                raise RuntimeError(f"Semaphore cross pipeline context violation.")
                            sem_rel_node.next_nodes.append(sem_acq_node)
                            sem_acq_node.operation.add_tb_sync(sem_rel_node.operation.threadblock)
                            sem_acq_node.previous_nodes.append(sem_rel_node)
                            sem_acq_node.add_input()
                    
                    removed_keys.append(key)
                
                for key in removed_keys:
                    sem_rel.pop(key)
                    sem_acq.pop(key)

        if len(sem_acq.keys()) > 0:
            raise RuntimeError(f"Semaphore acquire hanging.")
    
    def reset(self):
        for node in self.node_list:
            node.reset()

    def print(self):
        self.reset()
        self.check()
        
        queue = Queue()
        for node in self.root_nodes:
            queue.put(node)

        while not queue.empty():
            node = queue.get()
            print(f"node {node.print()}")
            for next_node in node.next_nodes:
                next_node.add_reach()
                print(f"next_node {next_node.print()}")
                if next_node.get_reach() == next_node.get_input():
                    if isinstance(next_node, self.Node) and next_node.agg_node is not None:
                        for sub_node in next_node.agg_node.nodes:
                            queue.put(sub_node)
                    else:
                       queue.put(next_node) 
            print()

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
        self.reset()
        self.check()
        
        order = []
        queue = Queue()
        for node in self.root_nodes:
            queue.put(node)

        while not queue.empty():
            node = queue.get()
            order.extend(node.get_operations())
            for next_node in node.next_nodes:
                next_node.add_reach()
                if next_node.get_reach() == next_node.get_input():
                    if isinstance(next_node, self.Node) and next_node.agg_node is not None:
                        for sub_node in next_node.agg_node.nodes:
                            queue.put(sub_node)
                    else:
                       queue.put(next_node) 

        return order

    class BaseNode():
        def __init__(self):
            self.previous_nodes = []
            self.next_nodes = []
            self.input = 0
            self.reach = 0

        def add_input(self):
            self.input += 1
        
        def add_reach(self):
            self.reach += 1

        def get_input(self):
            return self.input

        def get_reach(self):
            return self.reach

        def reset(self):
            self.reach = 0

    class Node(BaseNode):
        def __init__(self, operation):
            self.operation = operation
            self.agg_node = None
            super().__init__()

        def get_operations(self):
            return [self.operation]

        def add_input(self):
            if self.agg_node is not None:
                self.agg_node.input += 1
            else:
                self.input += 1

        def add_reach(self):
            if self.agg_node is not None:
                self.agg_node.reach += 1
            else:
                self.reach += 1

        def get_input(self):
            if self.agg_node is not None:
                 return self.agg_node.input
            else:
                return self.input
                
        def get_reach(self):
            if self.agg_node is not None:
                 return self.agg_node.reach
            else:
                return self.reach

        def reset(self):
            if self.agg_node is not None:
                 self.agg_node.reset()
            else:
                self.reach = 0

        def print(self):
            return f"rank {self.operation.rank} tb {self.operation.threadblock} {self.operation.name}"
            

    class AggregateNode(BaseNode):
        def __init__(self):
            self.nodes = []
            super().__init__()
        
        def add_node(self, node):
            self.nodes.append(node)
            node.agg_node = self

        def get_operations(self):
            operations = []
            for node in self.nodes:
                operations.extend(node.get_operations())
            return operations

        def print(self):
            return f"rank {self.operations[0].rank} tb {self.operations[0].threadblock} {self.operations[0].name}"