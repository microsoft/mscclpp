# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.operations import *
from mscclpp.language.internal.types import SyncType


def fuse_operations(operations):
    operation_index = 0
    fused_operations = []
    while operation_index < len(operations):
        next_operation_index = operation_index + 1
        current_operation = operations[operation_index]
        fused_operation = current_operation

        if current_operation.name == Instruction.pipeline:
            pipeline_fused_operations = fuse_operations(current_operation.operations)
            current_operation.operations = pipeline_fused_operations

        while next_operation_index < len(operations):
            next_operation = operations[next_operation_index]
            fused_operation = current_operation + next_operation
            if fused_operation is None:
                break
            current_operation = fused_operation
            next_operation_index += 1

        fused_operations.append(current_operation)
        operation_index = next_operation_index

    return fused_operations
