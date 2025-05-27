from mscclpp.language.channel import Channel
from mscclpp.language.internal.types import ChannelType, RemoteBuffer
from dataclasses import dataclass, field


@dataclass
class ThreadBlockChannel:
    channel_type: ChannelType
    channel_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"channel_type": self.channel_type.value, "channel_ids": self.channel_ids}


@dataclass
class ThreadBlockRemoteBuffer:
    access_channel_type: ChannelType
    remote_buffer_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"accessChannelType": self.access_channel_type.value, "remoteBufferIds": self.remote_buffer_ids}


@dataclass
class Threadblock:
    id: int
    ops: list = field(default_factory=list)

    __remote_buffers = {
        ChannelType.memory: ThreadBlockRemoteBuffer(ChannelType.memory),
        ChannelType.port: ThreadBlockRemoteBuffer(ChannelType.port),
    }
    __intra_remote_buffer_ids = {}

    __channels = {
        ChannelType.memory: ThreadBlockChannel(ChannelType.memory),
        ChannelType.port: ThreadBlockChannel(ChannelType.port),
        ChannelType.switch: ThreadBlockChannel(ChannelType.switch),
    }
    __intra_channel_ids = {
        ChannelType.memory: {},
        ChannelType.port: {},
        ChannelType.switch: {},
    }

    def to_json(self) -> dict:
        channels = []
        for ch in self.__channels.values():
            if len(ch.channel_ids) > 0:
                channels.append({"channel_type": ch.channel_type.value, "channel_ids": list(ch.channel_ids)})
        remote_buffers = []
        for rb in self.__remote_buffers.values():
            if len(rb.remote_buffer_ids) > 0:
                remote_buffers.append(rb.to_json())

        return {
            "id": self.id,
            "ops": [op.to_json() for op in self.ops],
            "channels": channels,
            "remoteBufferIds": remote_buffers,
        }

    def add_channel(self, channel: Channel):
        if channel.channel_id not in self.__intra_channel_ids[channel.channel_type]:
            self.__intra_channel_ids[channel.channel_type][channel.channel_id] = len(
                self.__channels[channel.channel_type].channel_ids
            )
            self.__channels[channel.channel_type].channel_ids.append(channel.channel_id)
        return self.__intra_channel_ids[channel.channel_type][channel.channel_id]

    def add_remote_buffer(self, remote_buffer: RemoteBuffer):
        if remote_buffer.id not in self.__intra_remote_buffer_ids:
            self.__intra_remote_buffer_ids[remote_buffer.id] = len(self.__intra_remote_buffer_ids)
            self.__remote_buffers[remote_buffer.channel_access].remote_buffer_ids.append(remote_buffer.id)
        return self.__intra_remote_buffer_ids[remote_buffer.id]

    def add_operation(self, op):
        self.ops.append(op)
