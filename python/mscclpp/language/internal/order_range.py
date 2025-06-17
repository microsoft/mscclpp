from sortedcontainers import SortedDict
from typing import List
from mscclpp.language.internal.dsl_types import BufferType, DataAccessType
from mscclpp.language.internal.operations import *
from enum import Enum

class IntervalMap:
    def __init__(self):
        self.intervals = {
            BufferType.input: SortedDict(),
            BufferType.output: SortedDict(),
            BufferType.scratch: SortedDict(),
        }

    def process_operations(self, operations):
        result_operations = []
        for operation in operations:
            data_access = operation.local_data_access()

            conflict = False
            for data_access_element in data_access:
                if operation.name == Instruction.nop or operation.name == Instruction.barrier:
                    self.clear_data_access()
                elif self.add_op(data_access_element):
                    if conflict is False:
                        conflict = True
                        result_operations.append(SyncOperation())
                    else:
                        raise RuntimeError(f"Data access conflict detected inside same operation for {operation}")
            result_operations.append(operation)
        
        return result_operations


    def add_op(self, data_access: DataAccess) -> bool:
        range_element = IntervalMap.RangeElemet(data_access.start, data_access.end)
        keys = self.intervals[data_access.buffer_type].keys()
        idx = self.lower_bound(0, len(keys) - 1, keys, range_element)
        result = False

        while idx < len(keys) and range_element.overlaps(keys[idx]):
            conflict_element = keys[idx]
            conflict_operation_type = self.intervals[data_access.buffer_type][conflict_element]
            self.intervals[data_access.buffer_type].pop(conflict_element)
            if conflict_element.end > range_element.end:
                self.intervals[data_access.buffer_type][
                    IntervalMap.RangeElemet(range_element.end + 1, conflict_element.end)
                ] = conflict_operation_type
            if conflict_element.start < range_element.start:
                self.intervals[data_access.buffer_type][IntervalMap.RangeElemet(conflict_element.start, range_element.start - 1)] = conflict_operation_type

            if data_access.data_access_type != DataAccessType.read and data_access.buffer_type != DataAccessType.read:
                self.clear_data_access()
                result = True
                break

        self.intervals[data_access.buffer_type][range_element] = data_access.data_access_type
        return result

    def clear_data_access(self):
        self.intervals[BufferType.input].clear()
        self.intervals[BufferType.output].clear()
        self.intervals[BufferType.scratch].clear()

    def lower_bound(self, init_pos, final_pos, range_elemet_list, range_element):
        if init_pos >= final_pos:
            return init_pos

        mid_pos = (init_pos + final_pos) // 2
        if range_element.start <= range_elemet_list[mid_pos].end:
            final_pos = mid_pos
        else:
            init_pos = mid_pos + 1
        return self.lower_bound(init_pos, final_pos, range_elemet_list, range_element)

    class RangeElemet:
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __lt__(self, other):
            # Sort primarily by start, then by end
            if self.start != other.start:
                return self.start < other.start
            return self.end < other.end

        def __eq__(self, other):
            return self.start == other.start and self.end == other.end

        def __hash__(self):
            return hash((self.start, self.end))

        def overlaps(self, other) -> bool:
            return self.start <= other.end and other.start <= self.end


def main():
    im = IntervalMap()

    range_elements = [IntervalMap.RangeElemet(3, 4), IntervalMap.RangeElemet(5, 7), IntervalMap.RangeElemet(10, 12)]
    print(im.lower_bound(0, len(range_elements) - 1, range_elements, IntervalMap.RangeElemet(1, 2)))


if __name__ == "__main__":
    main()
