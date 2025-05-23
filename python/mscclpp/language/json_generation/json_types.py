from mscclpp.language.src.types import BufferType, ChannelType, Instruction, RemoteBuffer
from mscclpp.language.json_generation.json_operations import create_json_operation
from typing import List, Set
from dataclasses import dataclass, field
import json


@dataclass
class JsonChannel:
    channel_type: ChannelType
    channel_ids: list[int] = field(default_factory=list)


@dataclass
class JsonThreadblock:
    id: int
    map_remote_buffer_ids: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)
    remote_buffer_ids: set = field(default_factory=set)

    __map_intra_node_remote_buffer_ids: dict = field(default_factory=dict)

    __channels = {
        ChannelType.memory: JsonChannel(ChannelType.memory),
        ChannelType.port: JsonChannel(ChannelType.port),
        ChannelType.switch: JsonChannel(ChannelType.switch),
    }
    __intranode_channel_ids = {
        ChannelType.memory: {},
        ChannelType.port: {},
        ChannelType.switch: {},
    }

    def json_to_dict(self) -> dict:
        channels = []
        for ch in self.__channels.values():
            if len(ch.channel_ids) > 0:
                channels.append({"channel_type": ch.channel_type.value, "channel_ids": list(ch.channel_ids)})
        return {
            "id": self.id,
            "ops": [op.json_to_dict() for op in self.ops],
            "channels": channels,
            "remoteBufferIds": list(self.remote_buffer_ids),
        }

    def add_channel(self, channel_type: ChannelType, channel_ids: List[int]):
        if channel_type != ChannelType.none:
            for id in channel_ids:
                if id not in self.__intranode_channel_ids[channel_type]:
                    self.__intranode_channel_ids[channel_type][id] = len(self.__channels[channel_type].channel_ids)
                    self.__channels[channel_type].channel_ids.append(channel_ids)

    def add_operation(self, op):
        for chunk in op.remote_chunks:
            remote_buffer = RemoteBuffer(chunk.rank, chunk.buffer, op.channel_type)
            if self.map_remote_buffer_ids[remote_buffer] not in self.remote_buffer_ids:
                self.__map_intra_node_remote_buffer_ids[remote_buffer] = len(self.remote_buffer_ids)
                self.remote_buffer_ids.add(self.map_remote_buffer_ids[remote_buffer])

        self.ops.append(create_json_operation(op, self.__map_intra_node_remote_buffer_ids))


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
            json_channels.append({"type": ch_type.value, "connectedTo": dst_ranks})
        return {
            "id": self.id,
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "scratch_chunks": self.scratch_chunks,
            "threadblocks": [tb.json_to_dict() for tb in self.threadblocks],
            "channels": json_channels,
            "remote_buffers": [rb.json_to_dict() for rb in self.remote_buffers],
            "buffer_alignment": self.buffer_alignment,
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
