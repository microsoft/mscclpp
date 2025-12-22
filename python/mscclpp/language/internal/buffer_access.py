# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sortedcontainers import SortedDict
from typing import List
from mscclpp.language.internal.types import *
from mscclpp.language.internal.operations import *
from enum import Enum


class BuffersAccess:
    def __init__(self, num_ranks, intra_rank_sync):
        self.intra_rank_sync = intra_rank_sync
        self.rank_intervals = [
            {
                BufferType.input: SortedDict(),
                BufferType.output: SortedDict(),
                BufferType.scratch: SortedDict(),
            }
            for _ in range(num_ranks)
        ]
        self.track_sync = {}
        self.track_barrier = {}

    def process_operations(self, operations):
        result_operations = []
        for i in range(len(operations)):
            operation = operations[i]
            if operation.name == Instruction.nop or operation.name == Instruction.barrier:
                self.track_sync[operation.rank, operation.threadblock] = i
                if operation.name == Instruction.barrier:
                    self.update_barrier(operation, i)
            if operation.name == Instruction.sem_acquire:
                self.update_semaphore(operation, i)
            else:
                data_access = operation.local_data_access(i)
                data_access_conflict_same_ctx = DataAccessConflict(operation.rank)
                data_access_conflict_diff_ctx = DataAccessConflict(operation.rank)
                for data_access_element in data_access:
                    computed_data_access_conflict_same_ctx, computed_data_access_conflict_diff_ctx = (
                        self.compute_data_access(data_access_element)
                    )
                    data_access_conflict_same_ctx = (
                        data_access_conflict_same_ctx + computed_data_access_conflict_same_ctx
                    )
                    data_access_conflict_diff_ctx = (
                        data_access_conflict_diff_ctx + computed_data_access_conflict_diff_ctx
                    )
                fix_operations = self.resolve_conflicts(
                    operation.rank,
                    operation.threadblock,
                    operation.pipeline_context,
                    i,
                    data_access_conflict_same_ctx,
                    data_access_conflict_diff_ctx,
                )
                result_operations.extend(fix_operations)

            result_operations.append(operation)

        result_operations = self.add_pipeline_context_sync_operations(result_operations)

        return result_operations

    def update_barrier(self, operation, order_id):
        for tb in operation.barrier_info.tb_list:
            if operation.threadblock != tb:
                self.track_barrier[operation.rank, operation.threadblock, tb] = order_id
                self.track_sync[operation.rank, operation.threadblock] = order_id

    def update_semaphore(self, operation, order_id):
        for tb in operation.tb_sync:
            if operation.threadblock != tb:
                self.track_barrier[operation.rank, operation.threadblock, tb] = order_id

    def compute_data_access(self, data_access: DataAccess) -> bool:
        intervals = self.rank_intervals[data_access.rank]
        keys = intervals[data_access.buffer_type].keys()
        idx = self.lower_bound(0, len(keys) - 1, keys, data_access)
        conflict_same_ctx = DataAccessConflict(data_access.rank)
        conflict_diff_ctx = DataAccessConflict(data_access.rank)

        while len(keys) > 0 and data_access.overlaps(keys[idx]):
            conflict_data_access = keys[idx]
            conflict_operation_type = intervals[data_access.buffer_type][conflict_data_access]
            if (
                data_access.pipeline_context is None
                or data_access.pipeline_context == conflict_data_access.pipeline_context
            ):
                conflict_same_ctx = conflict_same_ctx + data_access.check_conflict(conflict_data_access)
            else:
                conflict_diff_ctx = conflict_diff_ctx + data_access.check_conflict(conflict_data_access)

            intervals[data_access.buffer_type].pop(conflict_data_access)
            if conflict_data_access.end > data_access.end:
                intervals[data_access.buffer_type][
                    DataAccess(
                        conflict_data_access.rank,
                        conflict_data_access.threadblock,
                        conflict_data_access.operation_global_id,
                        conflict_data_access.operation_order_id,
                        data_access.end,
                        conflict_data_access.end,
                        conflict_data_access.buffer_type,
                        conflict_operation_type,
                        conflict_data_access.tb_group,
                        conflict_data_access.pipeline_context,
                    )
                ] = conflict_operation_type
            if conflict_data_access.start < data_access.start:
                intervals[data_access.buffer_type][
                    DataAccess(
                        conflict_data_access.rank,
                        conflict_data_access.threadblock,
                        conflict_data_access.operation_global_id,
                        conflict_data_access.operation_order_id,
                        conflict_data_access.start,
                        data_access.start,
                        conflict_data_access.buffer_type,
                        conflict_operation_type,
                        conflict_data_access.tb_group,
                        conflict_data_access.pipeline_context,
                    )
                ] = conflict_operation_type

            keys = intervals[data_access.buffer_type].keys()
            idx = self.lower_bound(0, len(keys) - 1, keys, data_access)

        intervals[data_access.buffer_type][data_access] = data_access.data_access_type
        return (conflict_same_ctx, conflict_diff_ctx)

    def resolve_conflicts(
        self,
        rank,
        threadblock,
        pipeline_context,
        order_id,
        data_access_conflict_same_ctx: DataAccessConflict,
        data_access_conflict_diff_ctx: DataAccessConflict,
    ):
        fix_operations = []
        if data_access_conflict_same_ctx.conflict_type == DataAccessConflictType.intra_threadblock:
            for tb in data_access_conflict_same_ctx.threadblocks:
                if (rank, threadblock) not in self.track_sync or tb[1] > self.track_sync[(rank, threadblock)]:
                    fix_operations.append(SyncOperation(rank, threadblock))
                    self.track_sync[(rank, threadblock)] = order_id
                    break
        if (
            data_access_conflict_same_ctx.conflict_type == DataAccessConflictType.inter_threadblock
            and self.intra_rank_sync
        ):
            conflict_tb = set([threadblock])
            for tb in data_access_conflict_same_ctx.threadblocks:
                if threadblock != tb[0] and (
                    (rank, threadblock, tb[0]) not in self.track_barrier
                    or self.track_barrier[(rank, threadblock, tb[0])] < tb[1]
                ):
                    if not tb[2]:
                        raise RuntimeError("Operations order not defined.")
                    conflict_tb.add(tb[0])
            if len(conflict_tb) > 1:
                for tb in conflict_tb:
                    op = BarrierOperation(rank, tb, conflict_tb)
                    self.update_barrier(op, order_id)
                    fix_operations.append(op)

        if pipeline_context is not None:
            if (rank, threadblock) not in pipeline_context.pre_operations:
                pipeline_context.pre_operations[(rank, threadblock)] = []

            if data_access_conflict_diff_ctx.conflict_type == DataAccessConflictType.intra_threadblock:
                for tb in data_access_conflict_diff_ctx.threadblocks:
                    if (rank, threadblock) not in self.track_sync or tb[1] > self.track_sync[(rank, threadblock)]:
                        self.track_sync[(rank, threadblock)] = order_id
                        pipeline_context.pre_operations[(rank, threadblock)].append(SyncOperation(rank, threadblock))
                        break
            if (
                data_access_conflict_diff_ctx.conflict_type == DataAccessConflictType.inter_threadblock
                and self.intra_rank_sync
            ):
                conflict_tb = set([threadblock])
                for tb in data_access_conflict_diff_ctx.threadblocks:
                    if threadblock != tb[0] and (
                        (rank, threadblock, tb[0]) not in self.track_barrier
                        or self.track_barrier[(rank, threadblock, tb[0])] < tb[1]
                    ):
                        if not tb[2]:
                            raise RuntimeError("Operations order not defined.")
                        conflict_tb.add(tb[0])
                if len(conflict_tb) > 1:
                    for tb in conflict_tb:
                        op = BarrierOperation(rank, tb, conflict_tb)
                        self.update_barrier(op, order_id)
                        pipeline_context.pre_operations[(rank, threadblock)].append(op)

        return fix_operations

    def add_pipeline_context_sync_operations(self, operations):
        result_operations = []
        pipeline_operations = dict()
        for i in range(len(operations)):
            operation = operations[i]
            if operation.pipeline_context is not None:
                pipeline_context = operation.pipeline_context
                result_operations.extend(
                    pipeline_context.pre_operations.get((operation.rank, operation.threadblock), [])
                )
                pipeline_context.pre_operations.pop((operation.rank, operation.threadblock), None)

                if (operation.rank, operation.threadblock, operation.pipeline_context) not in pipeline_operations:
                    pipeline_operations[(operation.rank, operation.threadblock, operation.pipeline_context)] = (
                        PipelineOperation(
                            operation.rank,
                            operation.threadblock,
                            operation.pipeline_context.unit,
                            operation.pipeline_context.num_chunks,
                        )
                    )
                    result_operations.append(
                        pipeline_operations[(operation.rank, operation.threadblock, operation.pipeline_context)]
                    )
                pipeline_operations[(operation.rank, operation.threadblock, operation.pipeline_context)].add_operation(
                    operation
                )
            else:
                result_operations.append(operation)

        return result_operations

    def lower_bound(self, init_pos, final_pos, data_access_list, data_access):
        if init_pos >= final_pos:
            return init_pos

        mid_pos = (init_pos + final_pos) // 2
        if data_access.lower_overlaps(data_access_list[mid_pos]):
            final_pos = mid_pos
        else:
            init_pos = mid_pos + 1
        return self.lower_bound(init_pos, final_pos, data_access_list, data_access)
