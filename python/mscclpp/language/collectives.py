# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import BufferType
from mscclpp.language.rank import BaseBuffer


class Collective:
    def __init__(self, num_ranks, chunk_factor, inplace, num_ranks_per_node=-1):
        self.num_ranks = num_ranks
        self.chunk_factor = chunk_factor
        self.inplace = inplace
        self.name = "custom"
        if num_ranks_per_node == -1:
            self.num_ranks_per_node = num_ranks
        else:
            self.num_ranks_per_node = num_ranks_per_node

    def init_buffers(self):
        pass

    def check(self, prog):
        pass

    def get_buffer_index(self, rank, buffer, index):
        return buffer, index

    def get_output_chunk_count(self, buffer_length, instances):
        if self.inplace:
            return 0
        else:
            return buffer_length * instances


class TestCollective(Collective):
    def __init__(self, num_ranks, input_size, output_size):
        Collective.__init__(self, num_ranks, 1, False)
        self.name = "test"
        self.input_size = input_size
        self.output_size = output_size

    # Initializes input buffer for a test collective
    def init_buffers(self):
        rank_buffers = []
        for rank in range(self.num_ranks):
            buffers = {
                BufferType.input: BaseBuffer(rank, BufferType.input, 0, self.input_size),
                BufferType.output: BaseBuffer(rank, BufferType.output, 0, self.output_size),
            }
            rank_buffers.append(buffers)
        return rank_buffers


class AllGather(Collective):
    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "allgather"

    # Initializes input buffer for an allgather
    def init_buffers(self):
        rank_buffers = []
        for rank in range(self.num_ranks):
            input_buffer_size = self.chunk_factor
            output_buffer_size = self.num_ranks * self.chunk_factor
            buffers = {
                BufferType.input: BaseBuffer(rank, BufferType.input, 0, input_buffer_size),
                BufferType.output: BaseBuffer(rank, BufferType.output, 0, output_buffer_size),
            }
            rank_buffers.append(buffers)
        return rank_buffers


class AllReduce(Collective):
    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "allreduce"

    # Initializes input buffer for an allgather
    def init_buffers(self):
        rank_buffers = []
        for rank in range(self.num_ranks):
            input_buffer_size = self.num_ranks * self.chunk_factor
            output_buffer_size = self.num_ranks * self.chunk_factor
            buffers = {
                BufferType.input: BaseBuffer(rank, BufferType.input, 0, input_buffer_size),
                BufferType.output: BaseBuffer(rank, BufferType.output, 0, output_buffer_size),
            }
            rank_buffers.append(buffers)
        return rank_buffers
