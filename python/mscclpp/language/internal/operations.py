from dataclasses import dataclass, field
from typing import List
from mscclpp.language.internal.types import ChannelType, Instruction, BufferType, ReduceOperation


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
        result["ctype"] = self.channel_type.value
        return result


@dataclass
class SignalOperation(BaseOperation):
    def __init__(self, channels_ids: List[int], channel_type: ChannelType, relaxed: bool = False):
        if relaxed:
            self.name = Instruction.relaxed_signal.value
        else:
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
    def __init__(self, channels_ids: List[int], channel_type: ChannelType, relaxed: bool = False):
        if relaxed:
            self.name = Instruction.relaxed_wait.value
        else:
            self.name = Instruction.wait.value
        self.channel_ids = channels_ids
        self.channel_type = channel_type

    def to_json(self):
        result = {"name": self.name}
        result["cids"] = self.channel_ids
        result["ctype"] = self.channel_type.value
        return result


@dataclass
class SyncOperation(BaseOperation):
    def __init__(self):
        self.name = Instruction.nop.value

    def to_json(self):
        result = {"name": self.name}
        return result
    

@dataclass
class ReduceOperation(BaseOperation):
    def __init__(self, src_buff: List[LocalChunk], dst_buff: List[LocalChunk], reduce_operation: ReduceOperation):
        self.name = Instruction.reduce.value
        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.reduce_operation = reduce_operation

    def to_json(self):
        result = {"name": self.name}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_json())
        result["reduce_op"] = self.reduce_operation.value
        return result


@dataclass
class CopyOperation(BaseOperation):
    def __init__(self, src_buff: List[LocalChunk], dst_buff: List[LocalChunk], ):
        self.name = Instruction.copy.value
        self.src_buff = src_buff
        self.dst_buff = dst_buff

    def to_json(self):
        result = {"name": self.name}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_json())
        return result
