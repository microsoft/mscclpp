# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List


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
class Chunk:
    rank: int
    buffer: BufferType
    index: int
    size: int

    def __hash__(self):
        return hash((self.rank, self.buffer, self.index, self.size))


@dataclass
class Operation:
    inst: Instruction
    rank: int
    tb: int
    local_chunks: List[Chunk] = field(default_factory=list)
    remote_chunks: List[Chunk] = field(default_factory=list)
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none


@dataclass
class RemoteBuffer:
    rank: int
    type: BufferType
    channel_access: ChannelType

    def json_to_dict(self):
        return {"rank": self.rank, "type": self.type.value, "channel_access": self.channel_access.value}

    def __hash__(self):
        return hash((self.rank, self.type, self.channel_access))
