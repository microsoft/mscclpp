from mscclpp.language.src.types import BufferType, ChannelType
from typing import List
from dataclasses import dataclass, field
import json


@dataclass
class RemoteBuffer:
    rank: int
    type: BufferType
    channel_access: ChannelType

    def json_to_dict(self):
        return {
            "rank": self.rank,
            "type": self.type.value,
            "channel_access": self.channel_access.value
        }
    
    def __hash__(self):
        return hash((self.rank, self.type, self.channel_access))

@dataclass
class JsonGpu:
    id: int
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    threadblocks: list = field(default_factory=list)
    channels: list = field(default_factory=dict)
    remote_buffers: list = field(default_factory=list)
    buffer_alignment: int = 0

    def to_json_dict(self) -> dict:
        channels = {}
        for ch in self.channels:
            if ch.channel_type not in channels:
                channels[ch.channel_type] = []
            channels[ch.channel_type].append(ch.dst_rank)
        json_channels = []
        for ch_type, dst_ranks in channels.items():
            json_channels.append({
                "type": ch_type.value,
                "connectedTo": dst_ranks
            })
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.to_json_dict() for tb in self.threadblocks],
            "channels": json_channels,
            "remote_buffers": [rb.json_to_dict() for rb in self.remote_buffers],
            "buffer_alignment": self.buffer_alignment
        }


@dataclass
class JsonProgram:
    name: str
    collective: str
    inplace: bool
    protocol: str
    gpus: List[JsonGpu] = field(default_factory=list)
    num_chunk_groups: int = 1
    num_threads_per_block: int = 1024
    use_double_scratch_buffer: bool = False
    min_message_size: int = 0
    max_message_size: int = 2**64 - 1

    def to_json(self) -> str:
        json_obj = {
            "name": self.name,
            "collective": self.collective,
            "inplace": self.inplace,
            "protocol": self.protocol,
            "gpus": [gpu.to_json_dict() for gpu in self.gpus],
            "num_threads_per_block": self.num_threads_per_block,
            "use_double_scratch_buffer": self.use_double_scratch_buffer,
            "min_message_size": self.min_message_size,
            "max_message_size": self.max_message_size,
        }

        return json.dumps(json_obj, indent=2)