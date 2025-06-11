# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
from typing import List, Set
from collections import defaultdict


class SyncType(Enum):
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
    start = "start"
    nop = "nop"
    copy = "copy"
    copy_packet = "cpkt"
    transform_to_packet = "tpkt"
    reduce = "re"
    reduce_packet = "repkt"
    signal = "signal"
    wait = "wait"
    relaxed_signal = "rlxsignal"
    relaxed_wait = "rlxwait"
    barrier = "barrier"
    flush = "flush"
    get = "get"
    put = "put"
    put_packet = "ppkt"
    read_put_packet = "rppkt"
    put_with_signal = "pws"
    put_with_signal_and_flush = "pwsf"
    reduce_send = "res"
    reduce_send_packet = "respkt"
    read_reduce = "rre"
    read_reduce_send = "rres"
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

    def to_json(self):
        return {"rank": self.rank, "type": self.buffer.value, "index": self.index, "size": self.size}


@dataclass
class RemoteBuffer:
    __remote_buffer_count = defaultdict(int)

    def __init__(
        self, local_rank: int, remote_rank: int, type: BufferType, channel_access: ChannelType, set_id: bool = False
    ):
        if set_id:
            self.id = RemoteBuffer.__remote_buffer_count[local_rank]
            RemoteBuffer.__remote_buffer_count[local_rank] += 1
        else:
            self.id = -1

        self.local_rank: int = local_rank
        self.remote_rank: int = remote_rank
        self.type: int = type
        self.channel_access: Set[ChannelType] = {channel_access}

    def set_id(self):
        if self.id == -1:
            self.id = RemoteBuffer.__remote_buffer_count[self.local_rank]
            RemoteBuffer.__remote_buffer_count[self.local_rank] += 1

    def to_json(self):
        return {
            "rank": self.remote_rank,
            "type": self.type.value,
            "access_channel_types": [ch.value for ch in self.channel_access],
        }

    def __hash__(self):
        return hash((self.remote_rank, self.type))


@dataclass
class RankGroup:
    size: int
    ranks: List[int]

    def to_json(self):
        return {
            "size": self.size,
            "ranks": self.ranks,
        }
