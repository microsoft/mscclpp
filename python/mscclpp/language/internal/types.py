# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
from typing import List, Set


class SyncType(Enum):
    none = "none"
    before = "before"
    after = "after"
    both = "both"

    def __str__(self):
        return self.value

    def __or__(self, other):
        if not isinstance(other, SyncType):
            return NotImplemented

        map_num = {SyncType.none: 0, SyncType.before: 1, SyncType.after: 2, SyncType.both: 3}
        return list(map_num.keys())[map_num[self] | map_num[other]]

    def __and__(self, other):
        if not isinstance(other, SyncType):
            return NotImplemented

        map_num = {SyncType.none: 0, SyncType.before: 1, SyncType.after: 2, SyncType.both: 3}
        return list(map_num.keys())[map_num[self] & map_num[other]]

    def __xor__(self, other):
        if not isinstance(other, SyncType):
            return NotImplemented

        map_num = {SyncType.none: 0, SyncType.before: 1, SyncType.after: 2, SyncType.both: 3}
        return list(map_num.keys())[map_num[self] ^ map_num[other]]


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
    unpack_packet = "upkt"
    reduce = "re"
    reduce_packet = "repkt"
    reduce_copy_packet = "recpkt"
    sem_acquire = "sem_acquire"
    sem_release = "sem_release"
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
    reduce_copy_send_packet = "recspkt"
    read_reduce = "rre"
    read_reduce_send = "rres"
    group_store = "gstore"
    group_load_reduce = "glre"
    group_load_reduce_store = "glres"
    pipeline = "pipeline"

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

    def to_dict(self):
        return {"rank": self.rank, "type": self.buffer.value, "index": self.index, "size": self.size}


@dataclass
class RemoteBuffer:
    def __init__(
        self, local_rank: int, remote_rank: int, type: BufferType, channel_access: ChannelType, set_id: bool = False
    ):
        self.local_rank: int = local_rank
        self.remote_rank: int = remote_rank
        self.type: int = type
        self.channel_access: Set[ChannelType] = {channel_access}

    def to_dict(self):
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

    def to_dict(self):
        return {
            "size": self.size,
            "ranks": self.ranks,
        }


class DataAccessType(Enum):
    read = "r"
    write = "w"
    both = "b"

    def __or__(self, other):
        if not isinstance(other, DataAccessType):
            return NotImplemented

        map_num = {DataAccessType.read: 1, DataAccessType.write: 2, DataAccessType.both: 3}
        return list(map_num.keys())[map_num[self] | map_num[other]]

    def __str__(self):
        return self.value


@dataclass
class DataAccess:
    operation_id: int
    start: int
    end: int
    buffer_type: BufferType
    data_access_type: DataAccessType

    def __lt__(self, other):
        if self.start != other.start:
            return self.start < other.start
        return self.end < other.end

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    def overlaps(self, other) -> bool:
        return self.start <= other.end and other.start <= self.end

    def check_conflict(self, other) -> bool:
        return (
            self.overlaps(other)
            and self.operation_id != other.operation_id
            and (self.data_access_type != DataAccessType.read or other.data_access_type != DataAccessType.read)
        )


class ReplicationPolicy(Enum):
    interleaved = "interleaved"
    none = "none"

    def __str__(self):
        return self.value
