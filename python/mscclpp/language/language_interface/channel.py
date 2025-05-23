from mscclpp.language.src.program import MSCCLPPProgram
from mscclpp.language.src.types import Instruction
from mscclpp.language.src.channel import BaseChannel
from mscclpp.language.src.types import Operation
from mscclpp.language.src.globals import get_program
from dataclasses import dataclass


@dataclass
class Channel(BaseChannel):
    prog: MSCCLPPProgram

    def __init__(self, dst_rank, src_rank, channel_type):
        self.prog = get_program()
        num_ranks = self.prog.num_ranks
        if src_rank >= num_ranks:
            raise RuntimeError(f"Source rank {src_rank} is out of bounds. Number of ranks: {num_ranks}")
        if dst_rank >= num_ranks:
            raise RuntimeError(f"Destination rank {dst_rank} is out of bounds. Number of ranks: {num_ranks}")

        self.channel_id = len(self.prog.channels[src_rank][channel_type])
        self.dst_rank = dst_rank
        self.src_rank = src_rank
        self.channel_type = channel_type
        self.prog.add_channel(self.channel_id, dst_rank, src_rank, channel_type)

    def signal(self, tb, sync):
        op = Operation(
            Instruction.signal, self.src_rank, tb, channel_ids=[self.channel_id], channel_type=self.channel_type
        )
        self.prog.instr_dag.add_operation(op)

    def wait(self, tb, sync):
        op = Operation(
            Instruction.wait, self.src_rank, tb, channel_ids=[self.channel_id], channel_type=self.channel_type
        )
        self.prog.instr_dag.add_operation(op)

    def put(self, dst_chunk, src_chunk, tb):
        op = Operation(
            Instruction.put,
            self.src_rank,
            tb,
            local_chunks=[src_chunk],
            remote_chunks=[dst_chunk],
            channel_ids=[self.channel_id],
            channel_type=self.channel_type,
        )
        self.prog.instr_dag.add_operation(op)
