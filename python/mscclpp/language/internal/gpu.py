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
    __nvls_channels: dict = field(default_factory=dict, init=False)

    def add_channel(self, channel):
        if channel.channel_type == ChannelType.switch:
            if channel.buffer_type not in self.__nvls_channels:
                self.__nvls_channels[channel.buffer_type] = Gpu.NVLSChannel(buffer_type=channel.buffer_type)
            self.__nvls_channels[channel.buffer_type].rank_groups.append(channel.rank_group)
        else:
            if channel.channel_type not in self.__channels:
                self.__channels[channel.channel_type] = Gpu.Channel(channel_type=channel.channel_type)
            self.__channels[channel.channel_type].connected_to.append(channel.dst_rank)

    def setup_channel(self, tb: int, channel) -> int:
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_channel(channel)

    def add_remote_buffer(self, tb: int, remote_buffer: RemoteBuffer) -> int:
        if (remote_buffer.remote_rank, remote_buffer.type) not in self.remote_buffers:
            remote_buffer.set_id()
            self.remote_buffers[(remote_buffer.remote_rank, remote_buffer.type)] = remote_buffer
        else:
            gpu_remote_buffer = self.remote_buffers[(remote_buffer.remote_rank, remote_buffer.type)]
            gpu_remote_buffer.channel_access.update(remote_buffer.channel_access)
            remote_buffer = gpu_remote_buffer

        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        return self.threadblocks[tb].add_remote_buffer(remote_buffer)

    def add_operation(self, tb: int, operation: BaseOperation):
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(ThreadBlock(self.id, i))

        self.threadblocks[tb].add_operation(operation)

    def optimize_operations(self):
        for tb in self.threadblocks:
            tb.optimize_operations()

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_json() for tb in self.threadblocks],
            "channels": [ch.to_json() for ch in self.__channels.values()]
            + [ch.to_json() for ch in self.__nvls_channels.values()],
            "remote_buffers": [rb.to_json() for rb in self.remote_buffers.values()],
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
