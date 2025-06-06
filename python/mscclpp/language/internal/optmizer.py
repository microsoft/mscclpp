from mscclpp.language.internal.operations import *

def fuse_instructions(operations):
    operation_index = 0
    fused_operations = []
    while operation_index < len(operations):
        next_operation_index = operation_index + 1
        current_operation = operations[operation_index]
        previous_operation = None
        fused_operation = current_operation
        
        while next_operation_index < len(operations):
            next_operation = operations[next_operation_index]
            fused_operation = current_operation + next_operation
            if fused_operation is None: 
                if previous_operation is not None and previous_operation.name == Instruction.nop: 
                    next_operation_index -=1
                break
            current_operation = fused_operation
            previous_operation = next_operation
            next_operation_index += 1

        fused_operations.append(current_operation)
        operation_index = next_operation_index

    return fused_operations