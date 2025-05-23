from mscclpp.language.src.program import MSCCLPPProgram
from mscclpp.language.src.types import BufferType, Chunk
from mscclpp.language.src.globals import get_program
from dataclasses import dataclass

@dataclass
class Rank:
    rank: int
    prog: MSCCLPPProgram

    def __init__(self, rank: int):
        self.rank = rank
        self.prog = get_program()
        if rank >= self.prog.num_ranks:
            raise RuntimeError(f"Rank {rank} is out of bounds. Number of ranks: {self.prog.num_ranks}")

    def _get_barrier_id(self, tb_list) -> int:
        return self.prog.ranks[self.rank].get_barrier_id(tb_list)

    def barrier(self, tb_list):
        barrier_id = self._get_barrier_id(tb_list)
        return self.prog.instr_dag.add_barrier(self.rank, tb_list, barrier_id)
    
    def get_input_buffer(self):
        buffer_size = self.prog.buffers_size[self.rank][BufferType.input]
        return Rank.Buffer(self.rank, BufferType.input, 0, buffer_size)
    
    def get_output_buffer(self):
        buffer_size = self.prog.buffers_size[self.rank][BufferType.output]
        return Rank.Buffer(self.rank, BufferType.output, 0, buffer_size)
    
    def generate_buffer(self, size):
        offset = self.prog.buffers_size[self.rank][BufferType.scratch]
        self.prog.buffers_size[self.rank][BufferType.scratch] += size
        return Rank.Buffer(self.rank, BufferType.scratch, offset, offset + size)
    
    class Buffer():
        def __init__(self, rank, buffer_type, offset, size):
            self.rank = rank
            self.buffer_type = buffer_type
            self.offset = offset
            self.size = offset + size

        def __getitem__(self, key):
            if self.offset + key.stop > self.size:
                raise RuntimeError(f"Index range from {self.offset + key.start} - {self.offset + key.stop} is out of bounds for buffer {self.buffer_type}. Buffer size: {self.size}")
            return Chunk(self.rank, self.buffer_type, self.offset + key.start, key.stop - key.start)
