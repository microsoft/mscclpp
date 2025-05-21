from mscclpp.channel_based_language.program import MSCCLPPProgram
from mscclpp.channel_based_language.types import BufferType, ChannelType, Instruction
from mscclpp.channel_based_language.channel import Channel
from mscclpp.channel_based_language.types import Chunk, Operation
from mscclpp.channel_based_language.world import _curr
from mscclpp.channel_based_language.json_generation.convert_to_json import generate_json
from dataclasses import dataclass

@dataclass
class RankRef:
    rank: int
    prog: MSCCLPPProgram

    def _get_barrier_id(self, tb_list) -> int:
        return self.prog.ranks[self.rank].get_barrier_id(tb_list)

    def barrier(self, tb_list):
        barrier_id = self._get_barrier_id(tb_list)
        return self.prog.instr_dag.add_barrier(self.rank, tb_list, barrier_id)
    
    def get_input_buffer(self):
        buffer_size = _curr().buffers_size[self.rank][BufferType.input]
        return ChunkRef(self.rank, BufferType.input, 0, buffer_size)
    
    def get_output_buffer(self):
        buffer_size = _curr().buffers_size[self.rank][BufferType.output]
        return ChunkRef(self.rank, BufferType.output, 0, buffer_size)
    
    def get_scratch_buffer(self, size):
        index = _curr().buffers_size[self.rank][BufferType.scratch]
        _curr().buffers_size[self.rank][BufferType.scratch] += size
        buffer_size = index + size
        return ChunkRef(self.rank, BufferType.scratch, index, buffer_size)

@dataclass
class ChannelRef(Channel):
    prog: MSCCLPPProgram

    def signal(self, tb, sync):
        op = Operation(Instruction.signal, self.src_rank, tb, channel_ids=[self.channel_id], channel_type=self.channel_type)
        self.prog.instr_dag.add_operation(op)

    def wait(self, tb, sync):
        op = Operation(Instruction.wait, self.src_rank, tb, channel_ids=[self.channel_id], channel_type=self.channel_type)
        self.prog.instr_dag.add_operation(op)

    def put(self, dst_chunk, src_chunk, tb):
        op = Operation(Instruction.put, self.src_rank, tb, local_chunks=[src_chunk], remote_chunks=[dst_chunk] ,channel_ids=[self.channel_id], channel_type=self.channel_type)
        self.prog.instr_dag.add_operation(op)

@dataclass
class ChunkRef(Chunk):

    def __getitem__(self, key):
        if self.index + key.stop > self.size:
            raise RuntimeError(f"Index range from {self.index + key.start} - {self.index + key.stop} is out of bounds for buffer {self.buffer}. Buffer size: {self.size}")
        return Chunk(self.rank, self.buffer, self.index + key.start, key.stop - key.start)

def channel(dst_rank: int, src_rank: int, channel_type: ChannelType) -> ChannelRef:
    num_ranks = _curr().num_ranks
    if src_rank >= num_ranks:
        raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
    if dst_rank >= num_ranks:
        raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")
    
    channel_id = len(_curr().channels[src_rank])
    _curr().channels[src_rank].append(Channel(channel_id, src_rank, dst_rank, channel_type))
    return ChannelRef(channel_id, src_rank, dst_rank, channel_type, _curr())

def rank(rank) -> RankRef:
    return RankRef(rank, _curr())

def JSON():
    return generate_json(_curr())