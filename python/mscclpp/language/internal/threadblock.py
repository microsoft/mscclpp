from mscclpp.language.internal.dsl_types import ChannelType, RemoteBuffer, BufferType
from mscclpp.language.internal.optmizer import *
from mscclpp.language.internal.buffer_access import *
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class ThreadBlock:
    def __init__(self, rank: int, id: int):
        self.rank = rank
        self.id = id
        self.ops = []

        self.__remote_buffers = OrderedDict()
        self.__intra_remote_buffer_ids = {ChannelType.memory: {}, ChannelType.port: {}}
        self.__channels = OrderedDict()
        self.__intra_channel_ids = {ChannelType.memory: {}, ChannelType.port: {}, ChannelType.switch: {}}

    def add_channel(self, channel):
        if channel.channel_type not in self.__channels:
            self.__channels[channel.channel_type] = ThreadBlock.Channel(channel_type=channel.channel_type)

        if channel.channel_type == ChannelType.switch:
            channel_id = channel.channel_ids[channel.src_rank]
        else:
            channel_id = channel.channel_id

        if channel_id not in self.__intra_channel_ids[channel.channel_type]:
            self.__intra_channel_ids[channel.channel_type][channel_id] = len(
                self.__channels[channel.channel_type].channel_ids
            )
            self.__channels[channel.channel_type].channel_ids.append(channel_id)
        return self.__intra_channel_ids[channel.channel_type][channel_id]

    def add_remote_buffer(self, remote_buffer_id: int, access_channel_type: ChannelType) -> int:
        if access_channel_type not in self.__remote_buffers:
            self.__remote_buffers[access_channel_type] = ThreadBlock.RemoteBuffer(
                access_channel_type=access_channel_type
            )

        if remote_buffer_id not in self.__intra_remote_buffer_ids[access_channel_type]:
            self.__intra_remote_buffer_ids[access_channel_type][remote_buffer_id] = len(
                self.__intra_remote_buffer_ids[access_channel_type]
            )
            self.__remote_buffers[access_channel_type].remote_buffer_ids.append(remote_buffer_id)
        return self.__intra_remote_buffer_ids[access_channel_type][remote_buffer_id]

    def add_operation(self, op):
        self.ops.append(op)

    def optimize_operations(self):
        self.ops = fuse_instructions(self.ops)

    def adding_data_sync(self):
        self.ops = adding_data_sync(self.ops)

    def resolve_data_dependency(self):
        interval_map = BuffersAccess()
        self.ops = interval_map.process_operations(self.ops)

    def shift_channels(self, shift):
        for channel in self.__channels.values():
            for i in range(len(channel.channel_ids)):
                channel.channel_ids[i] += shift[channel.channel_type]

    def shift_buffers(self, instance, num_instances):
        for op in self.ops:
            op.shift_buffers(instance, num_instances)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "ops": [op.to_json() for op in self.ops],
            "channels": [ch.to_json() for ch in self.__channels.values() if len(ch.channel_ids) > 0],
            "remote_buffer_refs": (
                [rb.to_json() for rb in self.__remote_buffers.values()] if self.__remote_buffers else []
            ),
        }

    @dataclass
    class Channel:
        channel_type: ChannelType
        channel_ids: list[int] = field(default_factory=list)

        def to_json(self) -> dict:
            return {"channel_type": self.channel_type.value, "channel_ids": self.channel_ids}

    @dataclass
    class RemoteBuffer:
        access_channel_type: ChannelType
        remote_buffer_ids: list[int] = field(default_factory=list)

        def to_json(self) -> dict:
            return {
                "access_channel_type": self.access_channel_type.value,
                "remote_buffer_ids": self.remote_buffer_ids,
            }
