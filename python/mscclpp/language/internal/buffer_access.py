# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sortedcontainers import SortedDict
from typing import List
from mscclpp.language.internal.types import BufferType, DataAccessType
from mscclpp.language.internal.operations import *
from enum import Enum


class BuffersAccess:
    def __init__(self):
        self.intervals = {
            BufferType.input: SortedDict(),
            BufferType.output: SortedDict(),
            BufferType.scratch: SortedDict(),
        }

    def process_operations(self, operations):
        result_operations = []
        for operation in operations:
            if operation.name == Instruction.nop or operation.name == Instruction.barrier:
                self.clear_data_access()
            else:
                if operation.name == Instruction.pipeline:
                    pipeline_buffer_access = BuffersAccess()
                    pipeline_result_operations = pipeline_buffer_access.process_operations(operation.operations)
                    operation.operations = pipeline_result_operations
                data_access = operation.local_data_access()
                sync_added = False
                for data_access_element in data_access:
                    if self.compute_data_access(data_access_element) and not sync_added:
                        result_operations.append(SyncOperation())
                        sync_added = True

            result_operations.append(operation)

        return result_operations

    def compute_data_access(self, data_access: DataAccess) -> bool:
        keys = self.intervals[data_access.buffer_type].keys()
        idx = self.lower_bound(0, len(keys) - 1, keys, data_access)
        conflict = False

        while len(keys) > 0 and data_access.overlaps(keys[idx]):
            conflict_data_access = keys[idx]
            conflict_operation_type = self.intervals[data_access.buffer_type][conflict_data_access]
            if data_access.check_conflict(conflict_data_access):
                self.clear_data_access()
                conflict = True
                break

            self.intervals[data_access.buffer_type].pop(conflict_data_access)
            if conflict_data_access.end > data_access.end:
                self.intervals[data_access.buffer_type][
                    DataAccess(
                        conflict_data_access.operation_id,
                        data_access.end + 1,
                        conflict_data_access.end,
                        conflict_data_access.buffer_type,
                        conflict_operation_type,
                    )
                ] = conflict_operation_type
            if conflict_data_access.start < data_access.start:
                self.intervals[data_access.buffer_type][
                    DataAccess(
                        conflict_data_access.operation_id,
                        conflict_data_access.start,
                        data_access.start - 1,
                        conflict_data_access.buffer_type,
                        conflict_operation_type,
                    )
                ] = conflict_operation_type

            keys = self.intervals[data_access.buffer_type].keys()
            idx = self.lower_bound(0, len(keys) - 1, keys, data_access)

        self.intervals[data_access.buffer_type][data_access] = data_access.data_access_type
        return conflict

    def clear_data_access(self):
        self.intervals[BufferType.input].clear()
        self.intervals[BufferType.output].clear()
        self.intervals[BufferType.scratch].clear()

    def lower_bound(self, init_pos, final_pos, data_access_list, data_access):
        if init_pos >= final_pos:
            return init_pos

        mid_pos = (init_pos + final_pos) // 2
        if data_access.start <= data_access_list[mid_pos].end:
            final_pos = mid_pos
        else:
            init_pos = mid_pos + 1
        return self.lower_bound(init_pos, final_pos, data_access_list, data_access)
