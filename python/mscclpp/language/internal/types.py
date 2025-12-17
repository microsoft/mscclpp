# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.thread_block_group import ThreadBlockGroup
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set
from collections import defaultdict
import uuid

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
    rank: int
    threadblock: int
    operation_global_id: uuid.UUID
    operation_order_id: int
    start: float
    end: float
    buffer_type: BufferType
    data_access_type: DataAccessType
    tb_group: ThreadBlockGroup = None
    
    def __lt__(self, other):
        if self.start != other.start:
            return self.start < other.start
        return self.end < other.end

    def __eq__(self, other, tolerance=1e-5):
        return (abs(self.start - other.start) < tolerance and 
                abs(self.end - other.end) < tolerance)

    def __hash__(self):
        return hash((self.start, self.end))

    def lower_overlaps(self, other, tolerance=1e-5) -> bool:
        return (self.start + tolerance < other.end)

    def overlaps(self, other, tolerance=1e-5) -> bool:
        return (self.start + tolerance < other.end) and (other.start + tolerance < self.end)

    def check_conflict(self, other) -> bool:
        if (
            self.overlaps(other)
            and self.operation_global_id != other.operation_global_id
            and (self.data_access_type != DataAccessType.read or other.data_access_type != DataAccessType.read)
        ):
            if self.threadblock == other.threadblock:
                return DataAccessConflict(self.rank, {(other.threadblock, other.operation_order_id, True)}, DataAccessConflictType.intra_threadblock)
            else:
                is_order_defined = ((self.tb_group is not None and other.tb_group is not None and self.tb_group.tbg_overlap(other.tb_group))
                or (self.tb_group is not None and other.tb_group is None and self.tb_group.tb_overlap(other.threadblock))
                or (self.tb_group is None and other.tb_group is not None and other.tb_group.tb_overlap(self.threadblock)))
                return DataAccessConflict(self.rank, {(self.threadblock, other.operation_order_id, True), (other.threadblock, other.operation_order_id, is_order_defined)}, DataAccessConflictType.inter_threadblock)
        else:
            return DataAccessConflict(self.rank)

class DataAccessConflictType(Enum):
    inter_threadblock = "inter_tb"
    intra_threadblock = "intra_tb"
    none = "none"

    def __add__(self, other):
        if not isinstance(other, DataAccessConflictType):
            return NotImplemented

        map_to_num = {DataAccessConflictType.none: 0, DataAccessConflictType.intra_threadblock: 1, DataAccessConflictType.inter_threadblock: 3}
        map_to_dact = {0: DataAccessConflictType.none, 1: DataAccessConflictType.intra_threadblock, 3: DataAccessConflictType.inter_threadblock}
        return map_to_dact[map_to_num[self] | map_to_num[other]]

    def __str__(self):
        return self.value

@dataclass
class DataAccessConflict():
    rank: int
    threadblocks: Set[int]  = field(default_factory=set)
    conflict_type: DataAccessConflictType = DataAccessConflictType.none

    def __add__(self, other):
        if not isinstance(other, DataAccessConflict):
            return NotImplemented

        return DataAccessConflict(self.rank, self.threadblocks | other.threadblocks, self.conflict_type + other.conflict_type)

class ReplicationPolicy(Enum):
    interleaved = "interleaved"
    none = "none"

    def __str__(self):
        return self.value