# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List

from mscclpp.language.buffer import Buffer


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


@dataclass
class Threadblock:
    channel: int = -1
    send: int = -1
    recv: int = -1
    ops: list = field(default_factory=list)
    rbid: int = -1  # threadblock id of the receiver
    id: int = -1
    channels: list = field(default_factory=list)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class ReplicationPolicy(Enum):
    # this means each instance deal with the different chunk
    # Chunk A, Chunk B -> Chunk A0, Chunk B0, Chunk A1, Chunk B1
    duplicated = "duplicated"
    # this means each instance deal with the different chunk in interleaved way
    # Chunk A, Chunk B -> Chunk A0, Chunk A1, Chunk B0, Chunk B1
    interleaved = "interleaved"
    # this means pack multi instrances to deal with the same chunk and share the channels
    packed = "packed"

    def __str__(self):
        return self.value


class Instruction(Enum):
    start = "start"
    nop = "nop"
    read_reduce_copy = "rrc"
    read_reduce_copy_send = "rrcs"
    reduce_send = "rs"
    copy = "copy"
    reduce = "reduce"
    copy_packet = "cpkt"
    transform_to_packet = "tpkt"
    reduce_send_packet = "rspkt"
    reduce_packet = "rpkt"
    put = "put"
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
class ChunkRef:
    rank: int
    buffer: Buffer
    index: int
    size: int

    def __hash__(self):
        return hash((self.rank, self.buffer, self.index, self.size))


class ChannelType(Enum):
    port = "port"
    memory = "memory"
    none = "none"
    nvls = "nvls"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Channel:
    srcBuffer: Buffer
    dstBuffer: Buffer
    type: ChannelType
    connected_to: Union[int, List[int]]

    def __hash__(self):
        # Ensure connected_to is converted to a tuple if it's a list
        connected_to_hashable = tuple(self.connected_to) if isinstance(self.connected_to, list) else self.connected_to
        return hash((self.srcBuffer, self.dstBuffer, self.type, connected_to_hashable))


@dataclass
class Op:
    inst: Instruction
    rank: int
    src: ChunkRef
    dst: ChunkRef
    depends: list = field(default_factory=list)
    step: int = -1  # Step in the TB
    tb: int = -1  # TB this op is assigned to
    prev: list = field(default_factory=list)  # List of instructions that happen before
    next: list = field(default_factory=list)  # List of instructions that happen after
    channel: int = -1
    channel_type: ChannelType = ChannelType.none
    srcs: list = field(default_factory=list)
    dsts: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def cnt(self):
        if self.src:
            if self.dst:
                assert self.src.size == self.dst.size
            return self.src.size
        elif self.dst:
            return self.dst.size
        else:
            return 0

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Op({self.inst}, {self.rank}, {self.src}, {self.dst}, step:{self.step}, tb:{self.tb})"
