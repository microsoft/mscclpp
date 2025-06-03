from mscclpp.language.channel import Channel
from dataclasses import dataclass, field
from mscclpp.language.internal.threadblock import Threadblock
from mscclpp.language.internal.operations import BaseOperation
from mscclpp.language.internal.types import RemoteBuffer, ChannelType


@dataclass
class Gpu:
    id: int
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    threadblocks: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    remote_buffers: set = field(default_factory=set)

    def add_channel(self, channel: Channel):
        self.channels.append(channel)

    def setup_channel(self, tb: int, channel: Channel) -> int:
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(Threadblock(i))

        return self.threadblocks[tb].add_channel(channel)

    def add_remote_buffer(self, tb: int, remote_buffer: RemoteBuffer) -> int:
        if remote_buffer not in self.remote_buffers:
            remote_buffer.set_id()
            self.remote_buffers.add(remote_buffer)
        else:
            gpu_remote_buffer = self.remote_buffer.find(remote_buffer)
            gpu_remote_buffer.channel_access.update(remote_buffer.channel_access)
            remote_buffer = gpu_remote_buffer

        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(Threadblock(i))

        return self.threadblocks[tb].add_remote_buffer(remote_buffer)

    def add_operation(self, tb: int, operation: BaseOperation):
        for i in range(len(self.threadblocks), tb + 1):
            self.threadblocks.append(Threadblock(i))

        self.threadblocks[tb].add_operation(operation)

    def to_json(self) -> dict:
        channels = {}
        nvls_channels = {}
        for ch in self.channels:
            if ch.channel_type != ChannelType.switch:
                if ch.channel_type not in channels:
                    channels[ch.channel_type] = []
                channels[ch.channel_type].append(ch.dst_rank)
            else:
                if ch.buffer_type not in nvls_channels:
                    nvls_channels[ch.buffer_type] = []
                nvls_channels[ch.buffer_type].append(ch)

        json_channels = []
        for ch_type, dst_ranks in channels.items():
            json_channels.append({"channel_type": ch_type.value, "connected_to": dst_ranks})

        for buffer_type, nvls_channels in nvls_channels.items():
            json_channels.append(
                {
                    "buffer_type": buffer_type.value,
                    "channel_type": ChannelType.switch.value,
                    "rank_group": [ch.rank_group.to_json() for ch in nvls_channels],
                }
            )

        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_json() for tb in self.threadblocks],
            "channels": json_channels,
            "remote_buffers": [rb.to_json() for rb in self.remote_buffers],
        }
