from mscclpp.language.internal.types import BufferType, Chunk
from mscclpp.language.internal.operations import CopyOperation, LocalChunk
from mscclpp.language.internal.globals import get_program
from dataclasses import dataclass


@dataclass
class Rank:
    rank: int

    def __init__(self, rank: int):
        self.rank = rank
        if rank >= get_program().num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

    def get_input_buffer(self):
        buffer_size = get_program().buffers_size[self.rank][BufferType.input]
        return BaseBuffer(self.rank, BufferType.input, 0, buffer_size)

    def get_output_buffer(self):
        buffer_size = get_program().buffers_size[self.rank][BufferType.output]
        return BaseBuffer(self.rank, BufferType.output, 0, buffer_size)

    def copy(self, dst_chunk, src_chunk, tb):
        if dst_chunk.rank != self.rank:
            raise RuntimeError(f"Cannot copy to chunk from different rank: {dst_chunk.rank} != {self.rank}")

        op = CopyOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            [LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
        )

        get_program().add_operation(self.rank, tb, op)


class BaseBuffer:
    def __init__(self, rank, buffer_type, offset, size):
        self.rank = rank
        self.buffer_type = buffer_type
        self.offset = offset
        self.size = offset + size

    def __getitem__(self, key):
        if self.offset + key.stop > self.size:
            raise RuntimeError(
                f"Index range from {self.offset + key.start} - {self.offset + key.stop} is out of bounds for buffer {self.buffer_type}. Buffer size: {self.size}"
            )
        return Chunk(self.rank, self.buffer_type, self.offset + key.start, key.stop - key.start)


class Buffer(BaseBuffer):
    def __init__(self, rank, size):
        if rank >= get_program().num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

        self.rank = rank
        self.buffer_type = BufferType.scratch
        self.offset = get_program().buffers_size[self.rank][BufferType.scratch]
        self.size = self.offset + size
        get_program().buffers_size[self.rank][BufferType.scratch] += size
