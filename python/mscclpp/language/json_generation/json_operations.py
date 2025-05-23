from dataclasses import dataclass, field
from typing import List, Dict, Set
from mscclpp.language.src.types import ChannelType, Instruction, Operation, BufferType
from mscclpp.language.src.types import RemoteBuffer


@dataclass
class JsonBaseOperation:
    name: str


@dataclass
class JsonLocalChunk:
    type: BufferType
    index: int
    size: int

    def json_to_dict(self):
        return {"type": self.type.value, "index": self.index, "size": self.size}


@dataclass
class JsonRemoteChunk:
    buffer_id: int
    index: int
    size: int

    def json_to_dict(self):
        return {"buffer_id": self.buffer_id, "index": self.index, "size": self.size}


@dataclass
class JsonPutOperation(JsonBaseOperation):
    src_buff: List[JsonLocalChunk] = field(default_factory=list)
    dst_buff: List[JsonRemoteChunk] = field(default_factory=list)
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none

    def __init__(self, operation: Operation, remote_buffers: Dict[RemoteBuffer, int]):
        self.name = operation.inst.value
        self.src_buff = []
        self.dst_buff = []
        self.channel_ids = []
        for chunk in operation.local_chunks:
            self.src_buff.append(JsonLocalChunk(chunk.buffer, chunk.index, chunk.size))
        for chunk in operation.remote_chunks:
            remote_buffer = RemoteBuffer(chunk.rank, chunk.buffer, operation.channel_type)
            remote_buffer_id = remote_buffers[remote_buffer]
            self.dst_buff.append(JsonRemoteChunk(remote_buffer_id, chunk.index, chunk.size))
        self.channel_ids = operation.channel_ids
        self.channel_type = operation.channel_type

    def json_to_dict(self):
        result = {"name": self.name}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.json_to_dict())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.json_to_dict())
        result["cids"] = self.channel_ids
        return result


@dataclass
class JsonSignalOperation(JsonBaseOperation):
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none

    def __init__(self, operation: Operation, remote_buffers: Dict[RemoteBuffer, int]):
        self.name = operation.inst.value
        self.channel_ids = operation.channel_ids
        self.channel_type = operation.channel_type

    def json_to_dict(self):
        result = {"name": self.name}
        result["cids"] = self.channel_ids
        result["ctype"] = self.channel_type.value
        return result


@dataclass
class JsonWaitOperation(JsonBaseOperation):
    channel_ids: List[int] = field(default_factory=list)
    channel_type: ChannelType = ChannelType.none

    def __init__(self, operation: Operation, remote_buffers: Dict[RemoteBuffer, int]):
        self.name = operation.inst.value
        self.channel_ids = operation.channel_ids
        self.channel_type = operation.channel_type

    def json_to_dict(self):
        result = {"name": self.name}
        result["cids"] = self.channel_ids
        result["ctype"] = self.channel_type.value
        return result


_map_operation_to_json = {
    Instruction.put: JsonPutOperation,
    Instruction.signal: JsonSignalOperation,
    Instruction.wait: JsonWaitOperation,
}


def create_json_operation(op: Operation, remote_buffers: Dict[RemoteBuffer, int]) -> JsonBaseOperation:
    if op.inst in _map_operation_to_json:
        return _map_operation_to_json[op.inst](op, remote_buffers)
    else:
        raise ValueError(f"Unsupported operation: {op.inst}")
