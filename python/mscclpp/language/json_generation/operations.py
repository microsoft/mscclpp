from dataclasses import dataclass, field
from typing import List, Dict, Set
from mscclpp.language.internal.types import ChannelType, Instruction, BufferType


@dataclass
class BaseOperation:
    name: str


@dataclass
class LocalChunk:
    type: BufferType
    index: int
    size: int

    def to_json(self):
        return {"type": self.type.value, "index": self.index, "size": self.size}


@dataclass
class RemoteChunk:
    buffer_id: int
    index: int
    size: int

    def to_json(self):
        return {"buffer_id": self.buffer_id, "index": self.index, "size": self.size}


@dataclass
class PutOperation(BaseOperation):

    def __init__(
        self,
        src_buff: List[LocalChunk],
        dst_buff: List[RemoteChunk],
        channel_ids: List[int],
        channel_type: ChannelType,
    ):
        self.name = Instruction.put.value
        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type

    def to_json(self):
        result = {"name": self.name}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_json())
        result["cids"] = self.channel_ids
        return result


@dataclass
class SignalOperation(BaseOperation):
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none

    def __init__(self, channels_ids: List[int], channel_type: ChannelType):
        self.name = Instruction.signal.value
        self.channel_ids = channels_ids
        self.channel_type = channel_type

    def to_json(self):
        result = {"name": self.name}
        result["cids"] = self.channel_ids
        result["ctype"] = self.channel_type.value
        return result


@dataclass
class WaitOperation(BaseOperation):
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none

    def __init__(self, channels_ids: List[int], channel_type: ChannelType):
        self.name = Instruction.wait.value
        self.channel_ids = channels_ids
        self.channel_type = channel_type

    def to_json(self):
        result = {"name": self.name}
        result["cids"] = self.channel_ids
        result["ctype"] = self.channel_type.value
        return result
