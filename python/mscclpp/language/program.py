# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.collectives import Collective
from mscclpp.language.internal.globals import set_program
from mscclpp.language.internal.types import BufferType, RemoteBuffer, ChannelType
from mscclpp.language.internal.gpu import Gpu
from typing import List
import json


class MSCCLPPProgram:
    def __init__(
        self,
        name: str,
        collective: Collective,
        num_ranks: int,
        protocol: str = "Simple",
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
        buffer_alignment: int = 16,
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
    ):
        self.name = name
        self.collective = collective
        self.num_ranks = num_ranks
        self.protocol = protocol
        self.num_threads_per_block = num_threads_per_block
        self.use_double_scratch_buffer = use_double_scratch_buffer
        self.buffer_alignment = buffer_alignment
        self.min_message_size = min_message_size
        self.max_message_size = max_message_size
        assert protocol == "Simple" or protocol == "LL", f"Given protocol: {protocol}. Must be either Simple, LL"
        self.buffers = collective.init_buffers()
        self.gpus: List[Gpu] = []
        for rank in range(self.num_ranks):
            self.gpus.append(
                Gpu(rank, self.buffers[rank][BufferType.input].size, self.buffers[rank][BufferType.output].size, 0)
            )

    def __enter__(self):
        set_program(self)

    def __exit__(self, exc_type, exc_value, traceback):
        set_program(None)

    def add_channel(self, channel):
        if channel.channel_type == ChannelType.switch:
            for gpu in channel.rank_group.ranks:
                self.gpus[gpu].add_channel(channel)
        else:
            self.gpus[channel.src_rank].add_channel(channel)

    def setup_channel(self, tb, channel):
        tb_channel_ids = []
        if channel.channel_type == ChannelType.switch:
            for gpu in channel.rank_group.ranks:
                tb_channel_ids.append(self.gpus[gpu].setup_channel(tb, channel))
        else:
            tb_channel_ids.append(self.gpus[channel.src_rank].setup_channel(tb, channel))
        return tb_channel_ids

    def setup_remote_chunk(self, rank, tb, remote_chunk: RemoteBuffer):
        return self.gpus[rank].add_remote_buffer(tb, remote_chunk)

    def add_operation(self, rank, tb, operation):
        self.gpus[rank].add_operation(tb, operation)

    def to_json(self):
        json_obj = {
            "name": self.name,
            "collective": self.collective.name,
            "inplace": self.collective.inplace,
            "protocol": self.protocol,
            "gpus": [gpu.to_json() for gpu in self.gpus],
            "num_threads_per_block": self.num_threads_per_block,
            "use_double_scratch_buffer": self.use_double_scratch_buffer,
            "buffer_alignment": self.buffer_alignment,
            "min_message_size": self.min_message_size,
            "max_message_size": self.max_message_size,
        }

        return json.dumps(json_obj, indent=2)
