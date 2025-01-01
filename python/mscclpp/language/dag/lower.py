# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from typing import List
from mscclpp.language.buffer import Buffer
from mscclpp.language.dag.instruction_dag import InstructionDAG
from mscclpp.language.types import ChunkRef, Gpu, Instruction, Op, ReplicationPolicy, Threadblock


class DagLower:
    def __init__(self, dag: InstructionDAG):
        self.dag = dag
        self.instanced_tbs = []

    def lower(self, instances: int, replication_policy: ReplicationPolicy):
        self._infer_dependencies()
        self._lower_buffers(instances)
        self._replicate(instances, replication_policy)
        return self._lower_tbs()

    def _replicate(self, instances: int, replication_policy: ReplicationPolicy):
        # update op step
        for rank, rank_tbs in enumerate(self.dag.tbs):
            for _, tb in rank_tbs.items():
                for id, op in enumerate(tb.ops):
                    op.step = id

        if instances == 1:
            self.instanced_tbs = self.dag.tbs
            return

        self.instanced_tbs = []
        for _ in range(self.dag.num_ranks):
            self.instanced_tbs.append({})

        def get_new_index(rank, buffer, index, size, i):
            if replication_policy == ReplicationPolicy.interleaved:
                return index * instances + i * size
            return len(self.dag.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            if ref is None:
                return None
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        def update_extra(op, ori_op):
            if op.inst == Instruction.barrier:
                tb_list = ori_op.extra["tb_list"]
                new_tb_list = [tb * instances + i for tb in tb_list]
                op.extra["tb_list"] = new_tb_list
                op.extra["barrier_id"] = ori_op.extra["barrier_id"] * instances + i

        for i in range(instances):
            # Generate all the threadblocks and ops
            for rank, rank_tbs in enumerate(self.dag.tbs):
                # rank_channels = self.num_channels[rank]
                for tbid, tb in rank_tbs.items():
                    itbid = tbid * instances + i
                    itb = Threadblock(id=itbid)
                    itb.ops = [None] * len(tb.ops)
                    for s, op in enumerate(tb.ops):
                        isrc = get_instance_ref(op.src)
                        idst = get_instance_ref(op.dst)
                        idepends = []
                        # Note: We don't need the fill out the rest of the metadata since replication is the last optimization
                        iop = Op(
                            op.inst,
                            op.rank,
                            isrc,
                            idst,
                            idepends,
                            op.step,
                            itbid,
                            channel_type=op.channel_type,
                            extra=copy.deepcopy(op.extra),
                        )
                        update_extra(iop, op)
                        itb.ops[s] = iop
                        for src, step in op.srcs:
                            isrc = get_instance_ref(src)
                            iop.srcs.append((isrc, step))
                        for dst, step in op.dsts:
                            idst = get_instance_ref(dst)
                            iop.dsts.append((idst, step))
                    for chan in tb.channels:
                        itb.channels.append(chan)
                    self.instanced_tbs[op.rank][itbid] = itb

        # Redo dependency analysis
        for rank, rank_tbs in enumerate(self.dag.tbs):
            for tbid, tb in rank_tbs.items():
                for i in range(instances):
                    itbid = tbid * instances + i
                    itb = self.instanced_tbs[rank][itbid]
                    for op, iop in zip(tb.ops, itb.ops):
                        iop.depends = [None] * len(op.depends)
                        for s, dep in enumerate(op.depends):
                            dep_tbid = dep.tb
                            dep_itbid = dep_tbid * instances + i
                            dep_step = dep.step
                            iop.depends[s] = self.instanced_tbs[op.rank][dep_itbid].ops[dep_step]

        # Convert local scratch buffers to index into one global scratch buffer

    def _lower_chunk(self, chunk):
        if chunk is not None and chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            buffer = self.dag.buffers[chunk.rank][chunk.buffer].get_buffer()
            index = self.dag.buffers[chunk.rank][chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(chunk.rank, buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def _lower_buffers(self, instances):
        for rank_buffers in self.dag.buffers:
            offset = 0
            for key, buf in rank_buffers.items():
                if key is not Buffer.input and key is not Buffer.output:
                    buf.set_offset(offset)
                    offset += buf.instance_size() * instances

    def _lower_tbs(self) -> List[Gpu]:
        gpus = []
        for rank, rank_tbs in enumerate(self.instanced_tbs):
            lowered_tbs = {}
            for tbid, tb in rank_tbs.items():
                for op in tb.ops:
                    op.src = self._lower_chunk(op.src)
                    op.dst = self._lower_chunk(op.dst)
                    srcs = sorted(op.srcs, key=lambda x: x[1])
                    dsts = sorted(op.dsts, key=lambda x: x[1])
                    op.srcs = [self._lower_chunk(src[0]) for src in srcs]
                    op.dsts = [self._lower_chunk(dst[0]) for dst in dsts]
                lowered_tbs[tbid] = tb
            gpus.append(Gpu(rank, list(lowered_tbs.values())))
        return gpus

    def _infer_dependencies(self):
        visited = set()
        for _, op in self.dag.operations.items():
            if op in visited:
                continue
            frontier = [op]
            while len(frontier) > 0:
                op = frontier[0]
                if op in visited:
                    frontier = frontier[1:]
                    continue
                # Dependencies for every op is the same as the ops that are stored in prev
                # Filter out dependencies that are satisified by tbs executing ops sequentially
                # If multiple dependent ops from the same tb keep the one that happens last
                depends = {}
                for dep_op in list(op.prev):
                    if dep_op.inst != Instruction.start:
                        tb = dep_op.tb
                        if tb not in depends or dep_op.step > depends[tb].step:
                            depends[tb] = dep_op
                op.depends = list(depends.values())
                visited.add(op)
                frontier = frontier[1:] + op.next
