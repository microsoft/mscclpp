# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import RemoteBuffer, ChannelType, BufferType, RankGroup
from mscclpp.language.internal.threadblock import ThreadBlock
from mscclpp.language.internal.operations import BaseOperation
from dataclasses import dataclass, field
from collections import *
from typing import List
import copy


@dataclass
class Gpu:
    id: int
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    threadblocks: list = field(default_factory=list)
    remote_buffers: OrderedDict = field(default_factory=OrderedDict)
    semaphores: list = field(default_factory=list)

    _channels: dict = field(default_factory=dict, init=False)
    _nvls_channels: list = field(default_factory=list, init=False)

    def add_channel(self, channel):
        if channel.channel_type == ChannelType.switch:
            self._nvls_channels.append(
                Gpu.NVLSChannel(buffer_type=channel.buffer_type, rank_groups=[channel.rank_group])
            )
        else:
            if channel.channel_type not in self._channels:
                self._channels[channel.channel_type] = Gpu.Channel(channel_type=channel.channel_type)
            self._channels[channel.channel_type].connected_to.append(channel.dst_rank)

    def setup_channel(self, tb: int, channel) -> int:
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_channel(channel)

    def add_remote_buffer(self, tb: int, remote_buffer: RemoteBuffer, channel_access: ChannelType) -> int:
        if remote_buffer not in self.remote_buffers:
            remote_buffer_id = len(self.remote_buffers)
        else:
            remote_buffer_id, existing_remote_buffer = self.remote_buffers[remote_buffer]
            remote_buffer.channel_access |= existing_remote_buffer.channel_access
        self.remote_buffers[remote_buffer] = (remote_buffer_id, remote_buffer)

        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_remote_buffer(remote_buffer_id, channel_access)

    def add_semaphore(self, semaphore):
        self.semaphores.append(Gpu.Semaphore(semaphore.initial_value))

    def add_operation(self, tb: int, operation: BaseOperation):
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        self.threadblocks[tb].add_operation(operation)

    def optimize_operations(self):
        for tb in self.threadblocks:
            tb.optimize_operations()

    def adding_data_sync(self):
        for tb in self.threadblocks:
            tb.adding_data_sync()

    def resolve_data_dependency(self):
        for tb in self.threadblocks:
            tb.resolve_data_dependency()

    def replicate_instances(self, instances, default_replication_function, buffer_replication_function):
        threadblocks = []

        self.input_chunks *= instances
        self.output_chunks *= instances
        self.scratch_chunks *= instances

        new_channels = {ChannelType.memory: [], ChannelType.port: [], ChannelType.switch: []}
        new_semaphores = []

        if ChannelType.memory in self._channels:
            for rank in self._channels[ChannelType.memory].connected_to:
                for _ in range(instances):
                    new_channels[ChannelType.memory].append(rank)
            self._channels[ChannelType.memory].connected_to = new_channels[ChannelType.memory]
        if ChannelType.port in self._channels:
            for rank in self._channels[ChannelType.port].connected_to:
                for _ in range(instances):
                    new_channels[ChannelType.port].append(rank)
            self._channels[ChannelType.port].connected_to = new_channels[ChannelType.port]
        for channel in self._nvls_channels:
            for _ in range(instances):
                new_channels[ChannelType.switch].append(channel)
        self._nvls_channels = new_channels[ChannelType.switch]

        for sempahore in self.semaphores:
            for _ in range(instances):
                new_semaphores.append(sempahore)
        self.semaphores = new_semaphores

        for threadblock in self.threadblocks:
            for instance in range(instances):
                tb = copy.deepcopy(threadblock)
                tb.id = default_replication_function(threadblock.id, instance, instances)

                tb.shift_channels(instance, instances, default_replication_function)
                tb.shift_buffers(instance, instances, buffer_replication_function)
                tb.shift_ids(instance, instances, default_replication_function)

                threadblocks.append(tb)

        self.threadblocks = threadblocks

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_dict() for tb in self.threadblocks],
            "channels": [ch.to_dict() for ch in self._channels.values()] + [ch.to_dict() for ch in self._nvls_channels],
            "remote_buffers": [rb[1].to_dict() for rb in self.remote_buffers.values()],
            "semaphores": [sm.to_dict() for sm in self.semaphores],
        }

    @dataclass
    class Channel:
        channel_type: ChannelType
        connected_to: List[int] = field(default_factory=list)

        def to_dict(self):
            return {"channel_type": self.channel_type.value, "connected_to": self.connected_to}

    @dataclass
    class NVLSChannel:
        buffer_type: BufferType
        channel_type: ChannelType = ChannelType.switch
        rank_groups: List[RankGroup] = field(default_factory=list)

        def to_dict(self):
            return {
                "channel_type": self.channel_type.value,
                "buffer_type": self.buffer_type.value,
                "rank_groups": [rg.to_dict() for rg in self.rank_groups],
            }

    @dataclass
    class Semaphore:
        init_value: int

        def to_dict(self):
            return {"init_value": self.init_value}
