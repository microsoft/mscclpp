# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List

@dataclass
class Gpu:
    rank: int
    threadblocks: list = field(default_factory=list)

    # From ncclize
    precopies: list = field(default_factory=list)
    postcopies: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    scratch: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)

    def scratch_size(self):
        return max((idx for addr, idx in self.scratch.items()), default=-1) + 1


@dataclass
class Program:
    name: str
    collective: str
    inplace: bool
    protocol: str
    gpus: List[Gpu] = field(default_factory=list)
    num_chunk_groups: int = 1
    num_threads_per_block: int = 1024
    use_double_scratch_buffer: bool = False
    min_message_size: int = 0
    max_message_size: int = 2**64 - 1

class BufferType(Enum):
    input = "i"
    output = "o"
    scratch = "s"

    def __str__(self):
        return self.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value < other.value
    

class Instruction(Enum):
    start = "start"
    nop = "nop"
    read_reduce_copy = "rrc"
    read_reduce_copy_send = "rrcs"
    reduce_send = "rs"
    copy = "copy"
    reduce = "re"
    copy_packet = "cpkt"
    transform_to_packet = "tpkt"
    reduce_send_packet = "rspkt"
    reduce_packet = "rpkt"
    put = "put"
    read_put_packet = "rppkt"
    put_packet = "ppkt"
    put_with_signal = "pws"
    put_with_signal_and_flush = "pwsf"
    get = "get"
    wait = "wait"
    signal = "signal"
    flush = "flush"
    barrier = "barrier"
    group_store = "gstore"
    group_load_reduce = "glre"
    group_load_reduce_store = "glres"

    def __str__(self):
        return self.value


@dataclass
class Chunk:
    rank: int
    buffer: BufferType
    index: int
    size: int

    def __hash__(self):
        return hash((self.rank, self.buffer, self.index, self.size))


class ChannelType(Enum):
    port = "port"
    memory = "memory"
    switch = "switch"
    none = "none"

    # Deprecated
    proxy = "port"
    sm = "memory"

    def __str__(self):
        return self.value


@dataclass
class Operation:
    inst: Instruction
    rank: int
    tb: int
    local_chunks: List[Chunk] = field(default_factory=list)
    remote_chunks: List[Chunk] = field(default_factory=list)
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none