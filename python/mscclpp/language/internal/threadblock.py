# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import ChannelType, RemoteBuffer, BufferType
from mscclpp.language.internal.optimizer import *
from mscclpp.language.internal.buffer_access import *
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import List


@dataclass
class ThreadBlock:
    def __init__(self, rank: int, id: int):
        self.rank = rank
        self.id = id
        self.ops = []

        self._remote_buffers = OrderedDict()
        self._intra_remote_buffer_ids = {ChannelType.memory: {}, ChannelType.port: {}}
        self._channels = OrderedDict()
        self._intra_channel_ids = {ChannelType.memory: {}, ChannelType.port: {}, ChannelType.switch: {}}

    def add_channel(self, channel):
        if channel.channel_type not in self._channels:
            self._channels[channel.channel_type] = ThreadBlock.Channel(channel_type=channel.channel_type)

        if channel.channel_type == ChannelType.switch:
            channel_id = channel.channel_ids[channel.src_rank]
        else:
            channel_id = channel.channel_id

        if channel_id not in self._intra_channel_ids[channel.channel_type]:
            self._intra_channel_ids[channel.channel_type][channel_id] = len(
                self._channels[channel.channel_type].channel_ids
            )
            self._channels[channel.channel_type].channel_ids.append(channel_id)
        return self._intra_channel_ids[channel.channel_type][channel_id]

    def add_remote_buffer(self, remote_buffer_id: int, access_channel_type: ChannelType) -> int:
        if access_channel_type not in self._remote_buffers:
            self._remote_buffers[access_channel_type] = ThreadBlock.RemoteBuffer(
                access_channel_type=access_channel_type
            )

        if remote_buffer_id not in self._intra_remote_buffer_ids[access_channel_type]:
            self._intra_remote_buffer_ids[access_channel_type][remote_buffer_id] = len(
                self._intra_remote_buffer_ids[access_channel_type]
            )
            self._remote_buffers[access_channel_type].remote_buffer_ids.append(remote_buffer_id)
        return self._intra_remote_buffer_ids[access_channel_type][remote_buffer_id]

    def add_operation(self, op):
        self.ops.append(op)

    def optimize_operations(self):
        self.ops = fuse_operations(self.ops)

    def adding_data_sync(self):
        self.ops = add_data_sync(self.ops)

    def resolve_data_dependency(self):
        interval_map = BuffersAccess()
        self.ops = interval_map.process_operations(self.ops)

    def shift_channels(self, instance, num_instances, replication_function):
        for channel in self._channels.values():
            for i in range(len(channel.channel_ids)):
                channel.channel_ids[i] = replication_function(channel.channel_ids[i], instance, num_instances)

    def shift_buffers(self, instance, num_instances, replication_function):
        for op in self.ops:
            op.shift_buffers(instance, num_instances, replication_function)

    def shift_ids(self, instance, num_instances, replication_function):
        for op in self.ops:
            op.shift_ids(instance, num_instances, replication_function)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ops": [op.to_dict() for op in self.ops],
            "channels": [ch.to_dict() for ch in self._channels.values() if len(ch.channel_ids) > 0],
            "remote_buffer_refs": (
                [rb.to_dict() for rb in self._remote_buffers.values()] if self._remote_buffers else []
            ),
        }

    @dataclass
    class Channel:
        channel_type: ChannelType
        channel_ids: List[int] = field(default_factory=list)

        def to_dict(self) -> dict:
            return {"channel_type": self.channel_type.value, "channel_ids": self.channel_ids}

    @dataclass
    class RemoteBuffer:
        access_channel_type: ChannelType
        remote_buffer_ids: List[int] = field(default_factory=list)

        def to_dict(self) -> dict:
            return {
                "access_channel_type": self.access_channel_type.value,
                "remote_buffer_ids": self.remote_buffer_ids,
            }
