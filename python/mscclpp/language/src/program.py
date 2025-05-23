# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.src.collectives import Collective
from mscclpp.language.src.instruction_dag import InstructionDAG
from mscclpp.language.src.channel import BaseChannel
from mscclpp.language.src.globals import set_program
from mscclpp.language.src.types import BufferType, ChannelType, Chunk, Operation, Instruction
from mscclpp.language.json_generation.types import JsonProgram, JsonGpu, RemoteBuffer
from mscclpp.language.json_generation.tb_class import JsonThreadblock

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
        self.channels = []
        for rank in range(self.num_ranks):
            self.channels.append({})
            self.channels[rank][ChannelType.memory] = []
            self.channels[rank][ChannelType.port] = []
            self.channels[rank][ChannelType.switch] = []

    def __enter__(self):
        set_program(self)

    def __exit__(self, exc_type, exc_value, traceback):
        set_program(None)

    def add_channel(self, channel_id, dst_rank, src_rank, channel_type):
        self.channels[src_rank][channel_type].append(BaseChannel(channel_id, dst_rank, src_rank, channel_type))

    def generating_execution_plan(self):
        prog = JsonProgram(
            self.name,
            self.collective.name,
            self.collective.inplace,
            self.protocol,
            num_threads_per_block=self.num_threads_per_block,
            use_double_scratch_buffer=self.use_double_scratch_buffer,
            min_message_size=self.min_message_size,
            max_message_size=self.max_message_size,
        )
        operations = self.instr_dag.retrieve_operations()

        gpus = []
        for rank in range(self.num_ranks):
            num_tb = self.instr_dag.retrieve_num_tb_per_rank(rank)
            used_remote_buffers = set()
            remote_buffers=[]
            remote_buffer_internal_ids = {}
            for tb in range(num_tb):
                for op in operations[rank][tb]:
                    for remote_chunk in op.remote_chunks:
                        if op.channel_type == ChannelType.memory or op.channel_type == ChannelType.port:
                            remote_buffer_id = len(remote_buffers)
                            remote_buffer = RemoteBuffer(
                               remote_chunk.rank, remote_chunk.buffer, op.channel_type
                            )
                            if remote_buffer not in used_remote_buffers:
                                used_remote_buffers.add(remote_buffer)
                                remote_buffers.append(remote_buffer)
                                remote_buffer_internal_ids[remote_buffer] = remote_buffer_id

            gpu = JsonGpu(
                rank,
                input_chunks=self.buffers_size[rank][BufferType.input],
                output_chunks=self.buffers_size[rank][BufferType.output],
                scratch_chunks=self.buffers_size[rank][BufferType.scratch],
                threadblocks=[],
                channels=[ch for sublist in self.channels[rank].values() for ch in sublist],
                remote_buffers=remote_buffers
                #buffer_aligment=
            )

            
            for tb in range(num_tb):
                tb_json = JsonThreadblock(
                    tb,
                    remote_buffer_internal_ids,
                )
                for op in operations[rank][tb]:
                    tb_json.add_channel(op.channel_type, op.channel_ids)
                    tb_json.add_operation(op)

                gpu.threadblocks.append(tb_json)

            gpus.append(gpu)
        prog.gpus = gpus
        return prog.to_json()