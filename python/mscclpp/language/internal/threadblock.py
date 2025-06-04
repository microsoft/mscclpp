from mscclpp.language.internal.types import ChannelType, RemoteBuffer, BufferType
from dataclasses import dataclass, field


@dataclass
class ThreadBlock:
    def __init__(self, rank: int, id: int):
        self.rank = rank
        self.id = id
        self.ops = []

        self.__remote_buffers = {}
        self.__intra_remote_buffer_ids = {}
        self.__channels = {}
        self.__nvls_channels = {}
        self.__intra_channel_ids = {ChannelType.memory: {}, ChannelType.port: {}}
        self.__intra_nvls_channel_ids = {BufferType.input: {}, BufferType.output: {}, BufferType.scratch: {}}

    def add_channel(self, channel):
        if channel.channel_type is ChannelType.switch:
            if channel.buffer_type not in self.__nvls_channels:
                self.__nvls_channels[channel.buffer_type] = ThreadBlock.NVLSChannel(
                    channel_type=ChannelType.switch, buffer_type=channel.buffer_type
                )

            channel_id = channel.channel_ids[self.rank]
            if channel_id not in self.__intra_nvls_channel_ids[channel.buffer_type]:
                self.__intra_nvls_channel_ids[channel.buffer_type][channel_id] = len(
                    self.__nvls_channels[channel.buffer_type].channel_ids
                )
                self.__nvls_channels[channel.buffer_type].channel_ids.append(channel_id)
            return self.__intra_nvls_channel_ids[channel.buffer_type][channel_id]
        else:
            if channel.channel_type not in self.__channels:
                self.__channels[channel.channel_type] = ThreadBlock.Channel(channel_type=channel.channel_type)

            channel_id = channel.channel_id
            if channel_id not in self.__intra_channel_ids[channel.channel_type]:
                self.__intra_channel_ids[channel.channel_type][channel_id] = len(
                    self.__channels[channel.channel_type].channel_ids
                )
                self.__channels[channel.channel_type].channel_ids.append(channel_id)
            return self.__intra_channel_ids[channel.channel_type][channel_id]

    def add_remote_buffer(self, remote_buffer: RemoteBuffer):
        for access_channel_type in remote_buffer.channel_access:
            if access_channel_type not in self.__remote_buffers:
                self.__remote_buffers[access_channel_type] = ThreadBlock.RemoteBuffer(
                    access_channel_type=access_channel_type
                )

        if remote_buffer.id not in self.__intra_remote_buffer_ids:
            self.__intra_remote_buffer_ids[remote_buffer.id] = len(self.__intra_remote_buffer_ids)
        for channel_access in remote_buffer.channel_access:
            self.__remote_buffers[channel_access].remote_buffer_ids.add(remote_buffer.id)
        return self.__intra_remote_buffer_ids[remote_buffer.id]

    def add_operation(self, op):
        self.ops.append(op)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "ops": [op.to_json() for op in self.ops],
            "channels": [ch.to_json() for ch in self.__channels.values() if len(ch.channel_ids) > 0]
            + [ch.to_json() for ch in self.__nvls_channels.values() if len(ch.channel_ids) > 0],
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
    class NVLSChannel:
        channel_type: ChannelType
        buffer_type: BufferType
        channel_ids: list[int] = field(default_factory=list)

        def to_json(self) -> dict:
            return {
                "channel_type": self.channel_type.value,
                "buffer_type": self.buffer_type.value,
                "channel_ids": self.channel_ids,
            }

    @dataclass
    class RemoteBuffer:
        access_channel_type: ChannelType
        remote_buffer_ids: set[int] = field(default_factory=set)

        def to_json(self) -> dict:
            return {
                "access_channel_type": self.access_channel_type.value,
                "remote_buffer_ids": list(self.remote_buffer_ids),
            }
