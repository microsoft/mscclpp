# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.channel_based_language.collectives import Collective
from mscclpp.channel_based_language.instruction_dag import InstructionDAG
from mscclpp.channel_based_language.rank import Rank
from mscclpp.channel_based_language.types import BufferType, ChannelType, Chunk, Operation, Instruction
from mscclpp.channel_based_language.channel import Channel
from mscclpp.channel_based_language.world import _curr, set_curr
from dataclasses import dataclass

class MSCCLPPProgram:
    def __init__(
        self,
        name: str,
        collective: Collective,
        num_ranks: int,
        protocol: str = "Simple",
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
    ):
        self.name = name
        self.collective = collective
        self.num_ranks = num_ranks
        self.protocol = protocol
        self.num_threads_per_block = num_threads_per_block
        self.use_double_scratch_buffer = use_double_scratch_buffer
        self.min_message_size = min_message_size
        self.max_message_size = max_message_size
        assert protocol == "Simple" or protocol == "LL", f"Given protocol: {protocol}. Must be either Simple, LL"
        self.buffers_size = collective.init_buffers()
        self.instr_dag = InstructionDAG(num_ranks)
        self.ranks = []
        self.channels = []
        for r in range(self.num_ranks):
            self.ranks.append(Rank(r))
            self.channels.append([])

    def __enter__(self):
        set_curr(self)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        set_curr(None)