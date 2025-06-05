from mscclpp.language.internal.operations import *

def _fuse_signal_signal_operations(current_operation: SignalOperation, next_operation: SignalOperation) -> SignalOperation:
    fused_operation = None
    if current_operation.channel_type == next_operation.channel_type:
        fused_operation = SignalOperation(current_operation.channel_ids + next_operation.channel_ids, current_operation.channel_type)

    return fused_operation

def _fuse_wait_wait_operations(current_operation: WaitOperation, next_operation: WaitOperation) -> WaitOperation:
    fused_operation = None
    if current_operation.channel_type == next_operation.channel_type:
        fused_operation = WaitOperation(current_operation.channel_ids + next_operation.channel_ids, current_operation.channel_type)

    return fused_operation
    
def _fuse_relaxed_signal_relaxed_signal_operations(current_operation: SignalOperation, next_operation: SignalOperation) -> SignalOperation:
    fused_operation = None
    if current_operation.channel_type == next_operation.channel_type:
        fused_operation = SignalOperation(current_operation.channel_ids + next_operation.channel_ids, current_operation.channel_type, relaxed=True)

    return fused_operation

def _fuse_relaxed_wait_relaxed_wait_operations(current_operation: WaitOperation, next_operation: WaitOperation) -> WaitOperation:
    fused_operation = None
    if current_operation.channel_type == next_operation.channel_type:
        fused_operation = WaitOperation(current_operation.channel_ids + next_operation.channel_ids, current_operation.channel_type, relaxed=True)

    return fused_operation


map_fuse = {
    (Instruction.signal, Instruction.signal): _fuse_signal_signal_operations,
    (Instruction.wait, Instruction.wait): _fuse_wait_wait_operations,
    (Instruction.relaxed_signal, Instruction.relaxed_signal): _fuse_relaxed_signal_relaxed_signal_operations,
    (Instruction.relaxed_wait, Instruction.relaxed_wait): _fuse_relaxed_wait_relaxed_wait_operations,
}

def fuse_instructions(operations):
    operation_index = 0
    fused_operations = []
    while operation_index < len(operations):
        next_operation_index = operation_index + 1
        current_operation = operations[operation_index]
        
        while next_operation_index < len(operations):
            next_operation = operations[next_operation_index]
            if (current_operation.name, next_operation.name) in map_fuse:
                fused_operation = map_fuse[(current_operation.name, next_operation.name)](current_operation, next_operation)
                if fused_operation is not None:
                    current_operation = fused_operation
                    next_operation_index += 1
                else:
                    break
            else:
                break

        fused_operations.append(current_operation)
        operation_index = next_operation_index

    return fused_operations