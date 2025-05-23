from mscclpp.language.src.types import BufferType, ChannelType, Instruction
from mscclpp.language.json_generation.types import RemoteBuffer
from typing import List, Set
from dataclasses import dataclass, field
import json

@dataclass
class JsonChannel:
    channel_type: ChannelType
    channel_ids: list[int] = field(default_factory=list)

@dataclass
class JsonOperation:
    name: str
    src_buff: list = field(default_factory=list)
    dst_buff: list = field(default_factory=list)
    channel_ids: list = field(default_factory=list)

    def json_to_dict(self):
        result = {"name": self.name}
        for chunk in self.src_buff:
            result["src_buff"].append({
                "type": chunk.buffer.value,
                "offset": chunk.index,
                "size": chunk.size
            })
        for chunk in self.src_buff:
            result["src_buff"].append({
                "buffer_id": 0,
                "offset": chunk.index,
                "size": chunk.size
            })
        result["cids"] = self.channel_ids
        return result

@dataclass
class JsonThreadblock:
    id: int
    map_remote_buffer_ids: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)
    remote_buffer_ids: set = field(default_factory=set)

    __channels={
        ChannelType.memory: JsonChannel(ChannelType.memory),
        ChannelType.port: JsonChannel(ChannelType.port),
        ChannelType.switch: JsonChannel(ChannelType.switch),
    }
    __intranode_channel_ids={
        ChannelType.memory: {},
        ChannelType.port: {},
        ChannelType.switch: {},
    }

    def to_json_dict(self) -> dict:
        channels = []
        for ch in self.__channels.values():
            if len(ch.channel_ids) > 0:
                channels.append({
                    "channel_type": ch.channel_type.value,
                    "channel_ids": list(ch.channel_ids)
                })
        return{
            "id": self.id,
            "ops": self.ops,
            "channels": channels,
            "remoteBufferIds": list(self.remote_buffer_ids)
        }
    
    def add_channel(self, channel_type: ChannelType, channel_ids: List[int]):
        if channel_type != ChannelType.none:
            for id in channel_ids:
                if id not in self.__intranode_channel_ids[channel_type]:
                    self.__intranode_channel_ids[channel_type][id] = len(self.__channels[channel_type].channel_ids)
                    self.__channels[channel_type].channel_ids.append(channel_ids)

    def add_operation(self, op):
        if op.inst == Instruction.put:
            src_buff = []
            for chunk in op.local_chunks:
                src_buff.append({
                    "type": chunk.buffer.value,
                    "offset": chunk.index,
                    "size": chunk.size
                })
            dst_buff = []
            for chunk in op.remote_chunks:
                remote_buffer = RemoteBuffer(
                        chunk.rank, chunk.buffer, op.channel_type
                    )
                if self.map_remote_buffer_ids[remote_buffer] not in self.remote_buffer_ids:
                    internal_buff_id = len(self.remote_buffer_ids)
                    self.remote_buffer_ids.add(self.map_remote_buffer_ids[remote_buffer])
                else:
                    internal_buff_id = self.remote_buffer_ids.index(self.map_remote_buffer_ids[remote_buffer])
                dst_buff.append({
                    "buffer_id": internal_buff_id,
                    "offset": chunk.index,
                    "size": chunk.size
                })
            self.ops.append({
                "name": op.inst.value,
                "src_buff": src_buff,
                "dst_buff": dst_buff,
                "cids": op.channel_ids,
                "ctype": op.channel_type.value
            })
        elif op.inst == Instruction.signal:
            self.ops.append({
                "name": op.inst.value,
                "cids": op.channel_ids,
                "ctype": op.channel_type.value
            })
        elif op.inst == Instruction.wait:
            self.ops.append({
                "name": op.inst.value,
                "cids": op.channel_ids,
                "ctype": op.channel_type.value
            })

        
        
