# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum
from typing import List
from collections import defaultdict


class SyncType(Enum):
    both = "both"
    before = "before"
    after = "after"
    none = "none"

    def __str__(self):
        return self.value

class ReduceOperationType(Enum):
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
    start = "start"  # ?
    nop = "nop" # OK
    copy = "copy" # OK
    copy_packet = "cpkt"  # packet -> regular OK
    transform_to_packet = "tpkt"  # regular -> packet OK
    reduce_copy = "rc" # OK
    reduce_copy_packet = "rcpkt"  # packet + packet -> packet OK
    signal = "signal" # OK
    wait = "wait" # OK
    relaxed_signal = "rlxsignal" # OK
    relaxed_wait = "rlxwait" # OK
    barrier = "barrier"  # To Doc
    flush = "flush"  # To Doc
    get = "get" # OK
    put = "put" # OK
    put_packet = "ppkt"  # regular => packet OK
    read_put_packet = "rppkt"  # packet => packet OK
    put_with_signal = "pws" # OK
    put_with_signal_and_flush = "pwsf" # OK 
    reduce_copy_send = "rcs"
    reduce_copy_send_packet = "rcspkt" # packet + packet -> packet => packet
    read_reduce_copy = "rrc"
    read_reduce_copy_send = "rrcs"
    group_store = "gstore"  # To Doc
    group_load_reduce = "glre"  # To Doc
    group_load_reduce_store = "glres"  # To Doc

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
