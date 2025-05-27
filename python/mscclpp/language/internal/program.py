# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.collectives import Collective
from mscclpp.language.channel import Channel
from mscclpp.language.internal.globals import set_program
from mscclpp.language.internal.types import BufferType, RemoteBuffer
from mscclpp.language.json_generation.gpu import Gpu
from typing import List, Dict
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
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
        buffer_alignment: int = 64,
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
        self.gpus: List[Gpu] = []
        for rank in range(self.num_ranks):
            self.gpus.append(
                Gpu(
                    rank,
                    self.buffers_size[rank][BufferType.input],
                    self.buffers_size[rank][BufferType.output],
                    self.buffers_size[rank][BufferType.scratch],
                    buffer_alignment=buffer_alignment,
                )
            )

    def __enter__(self):
        set_program(self)

    def __exit__(self, exc_type, exc_value, traceback):
        set_program(None)

    def add_channel(self, channel: Channel):
        self.gpus[channel.src_rank].add_channel(channel)

    def setup_channel(self, tb, channel: Channel):
        return self.gpus[channel.src_rank].setup_channel(tb, channel)

    def setup_remote_chunk(self, tb, remote_chunk: RemoteBuffer):
        return self.gpus[remote_chunk.rank].add_remote_buffer(tb, remote_chunk)

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
            "min_message_size": self.min_message_size,
            "max_message_size": self.max_message_size,
        }

        return json.dumps(json_obj, indent=2)
