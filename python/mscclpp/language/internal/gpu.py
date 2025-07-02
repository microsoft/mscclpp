from mscclpp.language.internal.types import RemoteBuffer, ChannelType, BufferType, RankGroup
from mscclpp.language.internal.threadblock import ThreadBlock
from mscclpp.language.internal.operations import BaseOperation
from dataclasses import dataclass, field
from typing import List


@dataclass
class Gpu:
    id: int
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    threadblocks: list = field(default_factory=list)
    remote_buffers: dict = field(default_factory=dict)

    __channels: dict = field(default_factory=dict, init=False)
    __nvls_channels: list = field(default_factory=list, init=False)

    def add_channel(self, channel):
        if channel.channel_type == ChannelType.switch:
            self.__nvls_channels.append(
                Gpu.NVLSChannel(buffer_type=channel.buffer_type, rank_groups=[channel.rank_group])
            )
        else:
            if channel.channel_type not in self.__channels:
                self.__channels[channel.channel_type] = Gpu.Channel(channel_type=channel.channel_type)
            self.__channels[channel.channel_type].connected_to.append(channel.dst_rank)

    def setup_channel(self, tb: int, channel) -> int:
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_channel(channel)

    def add_remote_buffer(self, tb: int, remote_buffer: RemoteBuffer, channel_access: ChannelType) -> int:
        if remote_buffer not in self.remote_buffers:
            remote_buffer_id = len(self.remote_buffers)
        else:
            remote_buffer_key = self.remote_buffers.pop(remote_buffer)
            remote_buffer.channel_access |= remote_buffer_key[1].channel_access
            remote_buffer_id = remote_buffer_key[0]
        self.remote_buffers[remote_buffer] = (remote_buffer_id, remote_buffer)

        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_remote_buffer(remote_buffer_id, channel_access)

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

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_json() for tb in self.threadblocks],
            "channels": [ch.to_json() for ch in self.__channels.values()]
            + [ch.to_json() for ch in self.__nvls_channels],
            "remote_buffers": [rb.to_json() for rb in self.remote_buffers.keys()],
        }

    @dataclass
    class Channel:
        channel_type: ChannelType
        connected_to: List[int] = field(default_factory=list)

        def to_json(self):
            return {"channel_type": self.channel_type.value, "connected_to": self.connected_to}

    @dataclass
    class NVLSChannel:
        buffer_type: BufferType
        channel_type: ChannelType = ChannelType.switch
        rank_groups: List[RankGroup] = field(default_factory=list)

        def to_json(self):
            return {
                "channel_type": self.channel_type.value,
                "buffer_type": self.buffer_type.value,
                "rank_groups": [rg.to_json() for rg in self.rank_groups],
            }
