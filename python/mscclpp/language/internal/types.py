# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum
from typing import List
from collections import defaultdict

class ReduceOperation(Enum):
    sum = "sum"
    min = "min"
    max = "max"

    def __str__(self):
        return self.value

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
    nop = "nop" # V Doc
    copy = "copy" # V Doc
    copy_packet = "cpkt" # V Doc
    transform_to_packet = "tpkt" # V Doc
    reduce = "re" # V Doc
    reduce_packet = "rpkt" # V Doc
    signal = "signal" # V Doc
    wait = "wait" # V Doc
    relaxed_signal = "rlxsignal" # V Doc
    relaxed_wait = "rlxwait" # V Doc
    put = "put" # V Doc
    put_packet = "ppkt" # V Doc
    read_reduce_copy = "rrc"
    read_reduce_copy_send = "rrcs"
    reduce_send = "rs"
    reduce_send_packet = "rspkt"
    read_put_packet = "rppkt"
    put_with_signal = "pws"
    put_with_signal_and_flush = "pwsf"
    get = "get"
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
class RemoteBuffer:
    __remote_buffer_count = defaultdict(int)

    def __init__(self, rank: int, type: BufferType, channel_access: ChannelType, set_id: bool = False):
        if set_id:
            self.id = RemoteBuffer.__remote_buffer_count[rank]
            RemoteBuffer.__remote_buffer_count[rank] += 1
        else:
            self.id = -1

        self.rank: int = rank
        self.type: int = type
        self.channel_access: List[ChannelType] = [channel_access]

    def set_id(self):
        if self.id == -1:
            self.id = RemoteBuffer.__remote_buffer_count[self.rank]
            RemoteBuffer.__remote_buffer_count[self.rank] += 1

    def to_json(self):
        return {
            "rank": self.rank,
            "type": self.type.value,
            "access_channel_types": [ch.value for ch in self.channel_access],
        }

    def __hash__(self):
        return hash((self.rank, self.type))
