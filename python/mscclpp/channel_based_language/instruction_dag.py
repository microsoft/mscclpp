from mscclpp.channel_based_language.types import Operation, ChannelType
from mscclpp.channel_based_language.json_generation.types import InfoLocation, RemoteBuffer
from typing import List

class InstructionDAG:
    def __init__(self, num_ranks):
        self.operations: List[List[Operation]] = []
        self.tb_per_rank = []

        for rank in range(num_ranks):
            self.operations.append([])
            self.tb_per_rank.append(0)

    def add_operation(self, operation):
        if operation.tb >= len(self.operations[operation.rank]):
            for _ in range(len(self.operations[operation.rank]), operation.tb + 1):
                self.operations[operation.rank].append([])
        self.operations[operation.rank][operation.tb].append(operation)
        self.tb_per_rank[operation.rank] = max(self.tb_per_rank[operation.rank], operation.tb + 1)

    def retrieve_num_tb_per_rank(self, rank):
        return self.tb_per_rank[rank]

    def retrieve_remote_buffers(self, rank):
        remote_buffers = set()
        for tb in range(len(self.operations[rank])):
            for operation in self.operations[rank][tb]:
                for chunk in operation.remote_chunks:
                    if operation.channel_type == ChannelType.memory:
                        remote_buffer = RemoteBuffer(chunk.rank, chunk.buffer, InfoLocation.gpu)
                        remote_buffers.add(remote_buffer)
                    elif operation.channel_type == ChannelType.port:
                        remote_buffer = RemoteBuffer(chunk.rank, chunk.buffer, InfoLocation.cpu)
                        remote_buffers.add(remote_buffer)
        return remote_buffers

    def retrieve_operations(self):
        return self.operations