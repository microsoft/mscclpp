# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import BufferType
from mscclpp.language.rank import BaseBuffer


class Collective:
    """Base class for defining collective communication patterns.

    Collective serves as the foundation for implementing various collective
    communication algorithms like AllGather, AllReduce, and ReduceScatter.
    It defines the common interface and behavior that all collective operations
    must implement.

    Attributes:
        num_ranks (int): The number of ranks participating in the collective.
        chunk_factor (int): The chunk factor for data subdivision.
        inplace (bool): Whether the collective operates in-place.
        name (str): The name of the collective operation.
    """

    def __init__(self, num_ranks, chunk_factor, inplace):
        self.num_ranks = num_ranks
        self.chunk_factor = chunk_factor
        self.inplace = inplace
        self.name = "custom"

    def init_buffers(self):
        pass

    def check(self, prog):
        pass


class TestCollective(Collective):
    """A test collective for validation and testing purposes.

    TestCollective provides a simple collective implementation used for
    testing the DSL functionality with custom input and output buffer sizes.

    Attributes:
        input_size (int): The size of the input buffer.
        output_size (int): The size of the output buffer.
    """

    def __init__(self, num_ranks, input_size, output_size):
        """Initialize a new TestCollective.

        Args:
            num_ranks (int): The number of ranks participating in the test collective.
            input_size (int): The size of the input buffer for each rank.
            output_size (int): The size of the output buffer for each rank.

        Example:
            >>> test_collective = TestCollective(num_ranks=4, input_size=4, output_size=4)
        """
        Collective.__init__(self, num_ranks, 1, False)
        self.name = "test"
        self.input_size = input_size
        self.output_size = output_size

    def init_buffers(self):
        """Initialize input and output buffers for the test collective.

        Creates input and output buffers with the specified sizes for each rank.

        Returns:
            list: A list of buffer dictionaries, one for each rank.
        """
        rank_buffers = []
        for rank in range(self.num_ranks):
            buffers = {
                BufferType.input: BaseBuffer(rank, BufferType.input, 0, self.input_size),
                BufferType.output: BaseBuffer(rank, BufferType.output, 0, self.output_size),
            }
            rank_buffers.append(buffers)
        return rank_buffers


class AllGather(Collective):
    """An AllGather collective communication pattern.

    The AllGather operation is a collective communication pattern where each rank
    begins with a unique block of data and, by the end of the operation, every
    process holds the concatenation of all data blocks from all ranks.

    This operation creates input buffers sized by chunk_factor and output buffers
    sized to hold data from all ranks (num_ranks * chunk_factor).
    """

    def __init__(self, num_ranks, chunk_factor, inplace):
        """Initialize a new AllGather collective.

        Args:
            num_ranks (int): The number of ranks participating in the AllGather.
            chunk_factor (int): The size factor for data chunks.
            inplace (bool): Whether the operation should be performed in-place.

        Example:
            >>> allgather = AllGather(num_ranks=4, chunk_factor=1, inplace=False)
        """
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "allgather"

    def init_buffers(self):
        """Initialize buffers for the AllGather operation.

        Creates input buffers sized by chunk_factor and output buffers
        sized to hold data from all ranks (num_ranks * chunk_factor).

        Returns:
            list: A list of buffer dictionaries, one for each rank.
        """
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
    """An AllReduce collective communication operation.

    The AllReduce operation combines data from all ranks using
    a specified reduction operation (e.g., sum, max, min) and
    then distributes the final reduced result back to all ranks.

    This operation creates input and output buffers both sized
    to hold the complete dataset (num_ranks * chunk_factor)
    """

    def __init__(self, num_ranks, chunk_factor, inplace):
        """Initialize a new AllReduce collective.

        Args:
            num_ranks (int): The number of ranks participating in the AllReduce.
            chunk_factor (int): The size factor for data chunks.
            inplace (bool): Whether the operation should be performed in-place.

        Example:
            >>> allreduce = AllReduce(num_ranks=4, chunk_factor=1, inplace=True)
        """
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "allreduce"

    def init_buffers(self):
        """Initialize buffers for the AllReduce operation.

        Creates input and output buffers both sized to hold the complete
        dataset (num_ranks * chunk_factor) since AllReduce operates on
        the full data from all ranks.

        Returns:
            list: A list of buffer dictionaries, one for each rank.
        """
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


class ReduceScatter(Collective):
    """A ReduceScatter collective communication operation.

    ReduceScatter performs a reduction operation across all ranks and then
    scatters the result, with each rank receiving a unique portion of the
    reduced data. This is the inverse of AllGather.

    This operations creates input buffers sized to hold the complete dataset
    (num_ranks * chunk_factor) and output buffers sized to hold
    each rank's portion (chunk_factor).
    """

    def __init__(self, num_ranks, chunk_factor, inplace):
        """Initialize a new ReduceScatter collective.

        Args:
            num_ranks (int): The number of ranks participating in the ReduceScatter.
            chunk_factor (int): The size factor for data chunks.
            inplace (bool): Whether the operation should be performed in-place.

        Example:
            >>> reduce_scatter = ReduceScatter(num_ranks=4, chunk_factor=1, inplace=False)
        """
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "reducescatter"

    def init_buffers(self):
        """Initialize buffers for the ReduceScatter operation.

        Creates input buffers sized to hold the complete dataset
        (num_ranks * chunk_factor) and output buffers sized to hold
        each rank's portion (chunk_factor).

        Returns:
            list: A list of buffer dictionaries, one for each rank.
        """
        rank_buffers = []
        for rank in range(self.num_ranks):
            input_buffer_size = self.num_ranks * self.chunk_factor
            output_buffer_size = self.chunk_factor
            buffers = {
                BufferType.input: BaseBuffer(rank, BufferType.input, 0, input_buffer_size),
                BufferType.output: BaseBuffer(rank, BufferType.output, 0, output_buffer_size),
            }
            rank_buffers.append(buffers)
        return rank_buffers


class AllToAll(Collective):

    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = "alltoall"

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
