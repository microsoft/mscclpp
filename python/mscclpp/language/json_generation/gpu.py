from mscclpp.language.channel import Channel
from dataclasses import dataclass, field
from mscclpp.language.json_generation.threadblock import Threadblock
from mscclpp.language.json_generation.operations import BaseOperation
from mscclpp.language.internal.types import RemoteBuffer

@dataclass
class Gpu:
    id: int
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    threadblocks: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    remote_buffers: set = field(default_factory=set)
    buffer_alignment: int = 0

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
        for ch in self.channels:
            if ch.channel_type not in channels:
                channels[ch.channel_type] = []
            channels[ch.channel_type].append(ch.dst_rank)
        json_channels = []
        for ch_type, dst_ranks in channels.items():
            json_channels.append({"type": ch_type.value, "connectedTo": dst_ranks})
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_json() for tb in self.threadblocks],
            "channels": json_channels,
            "remote_buffers": [rb.to_json() for rb in self.remote_buffers],
            "buffer_alignment": self.buffer_alignment,
        }
