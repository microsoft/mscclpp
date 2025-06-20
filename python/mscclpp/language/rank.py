from mscclpp.language.internal.dsl_types import BufferType, Chunk
from mscclpp.language.internal.operations import *
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
        return get_program().buffers[self.rank][BufferType.input]

    def get_output_buffer(self):
        return get_program().buffers[self.rank][BufferType.output]

    def copy(self, dst_chunk: Chunk, src_chunk: Chunk, tb: int, from_packet: bool = False, to_packet: bool = False):
        if dst_chunk.rank != self.rank or src_chunk.rank != self.rank:
            raise RuntimeError(
                f"Inconsistent ranks: dst {dst_chunk.rank}, src {src_chunk.rank}, self {self.rank}. They must match."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Inconsistent chunk sizes: dst {dst_chunk.size}, src {src_chunk.size}. They must match."
            )
        if from_packet and src_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Source chunk must be of type scratch.")
        if to_packet and dst_chunk.buffer != BufferType.scratch:
            raise RuntimeError(f"Destination chunk must be of type scratch.")

        op = CopyOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)],
            [LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
            from_packet,
            to_packet,
        )

        get_program().add_operation(self.rank, tb, op)

    def reduce(
        self,
        src_chunk: Chunk,
        other_chunks: List[Chunk],
        tb: int,
        dst_chunk: Chunk = None,
        reduce_op: ReduceOperationType = ReduceOperationType.sum,
        packet: bool = False,
    ):
        if dst_chunk is None:
            dst_chunk = src_chunk
        if dst_chunk.rank != self.rank or src_chunk.rank != self.rank:
            raise RuntimeError(
                f"Inconsistent ranks: dst {dst_chunk.rank}, src {src_chunk.rank}, self {self.rank}. They must match."
            )
        if dst_chunk.size != src_chunk.size:
            raise RuntimeError(
                f"Inconsistent chunk sizes: dst {dst_chunk.size}, src {src_chunk.size}. They must match."
            )
        for chunk in other_chunks:
            if chunk.rank != self.rank:
                raise RuntimeError(f"Other chunk rank {chunk.rank} does not match current rank {self.rank}.")
            if chunk.size != src_chunk.size:
                raise RuntimeError(
                    f"Inconsistent chunk sizes: other {chunk.size}, src {src_chunk.size}. They must match."
                )
            if packet and chunk.buffer != BufferType.scratch:
                raise RuntimeError(f"Other chunk must be of type scratch.")

        op = ReduceOperation(
            [LocalChunk(src_chunk.buffer, src_chunk.index, src_chunk.size)]
            + [LocalChunk(chunk.buffer, chunk.index, chunk.size) for chunk in other_chunks],
            [LocalChunk(dst_chunk.buffer, dst_chunk.index, dst_chunk.size)],
            reduce_operation=reduce_op,
            packet=packet,
        )
        get_program().add_operation(self.rank, tb, op)

    def barrier(self, tb_list: List[int]):
        if len(tb_list) == 0:
            raise RuntimeError("Barrier requires at least thread block.")
        elif len(tb_list) == 1:
            op = SyncOperation()
            get_program().add_operation(self.rank, tb_list[0], op)
        else:
            op = BarrierOperation(self.rank, tb_list)
            for tb in tb_list:
                get_program().add_operation(self.rank, tb, op)


class BaseBuffer:
    def __init__(self, rank: int, buffer_type: BufferType, offset: int, size: int):
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
    def __init__(self, rank: int, size: int):
        if rank >= get_program().num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

        self.rank = rank
        self.buffer_type = BufferType.scratch
        self.offset = get_program().gpus[rank].scratch_chunks
        self.size = self.offset + size
        get_program().gpus[rank].scratch_chunks += size
