from mscclpp.language.internal.operations import *
from mscclpp.language.internal.types import SyncType, FusionStatus


def fuse_instructions(operations):
    operation_index = 0
    fused_operations = []
    while operation_index < len(operations):
        next_operation_index = operation_index + 1
        current_operation = operations[operation_index]
        fused_operation = current_operation

        if current_operation.name == Instruction.pipeline:
            pipeline_fused_operations = fuse_instructions(current_operation.operations)
            current_operation.operations = pipeline_fused_operations

        while next_operation_index < len(operations):
            next_operation = operations[next_operation_index]
            fused_operation = current_operation + next_operation
            if fused_operation != None:
                break
            current_operation = fused_operation
            next_operation_index += 1

        fused_operations.append(current_operation)
        operation_index = next_operation_index

    return fused_operations


def adding_data_sync(operations):
    result_operations = []
    data_sync_operations = {
        Instruction.sem_acquire,
        Instruction.sem_release,
        Instruction.signal,
        Instruction.wait,
        Instruction.relaxed_signal,
        Instruction.relaxed_wait,
        Instruction.flush,
    }

    for operation in operations:
        if operation.name == Instruction.pipeline:
            pipeline_result_operations = adding_data_sync(operation.operations)
            operation.operations = pipeline_result_operations

        if operation.name in data_sync_operations and (
            operation.data_sync == SyncType.before or operation.data_sync == SyncType.both
        ):
            result_operations.append(SyncOperation())
        result_operations.append(operation)
        if operation.name in data_sync_operations and (
            operation.data_sync == SyncType.after or operation.data_sync == SyncType.both
        ):
            result_operations.append(SyncOperation())

    return result_operations
