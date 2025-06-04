from mscclpp.language.internal.types import ChannelType, Instruction, BufferType, ReduceOperationType
from dataclasses import dataclass
from typing import List


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
class NVLSChunk:
    rank: int
    type: BufferType
    index: int
    size: int

    def to_json(self):
        return {"rank": self.rank, "type": self.type.value, "index": self.index, "size": self.size}


@dataclass
class SyncOperation(BaseOperation):
    def __init__(self):
        self.name = Instruction.nop.value

    def to_json(self):
        result = {"name": self.name}
        return result


@dataclass
class CopyOperation(BaseOperation):
    def __init__(
        self,
        src_buff: List[LocalChunk],
        dst_buff: List[LocalChunk],
        from_packet: bool = False,
        to_packet: bool = False,
    ):
        if from_packet and to_packet:
            raise RuntimeError(f"Copy Operation from Packet to Packet is not Supported.")
        elif from_packet:
            self.name = Instruction.copy_packet.value
        elif to_packet:
            self.name = Instruction.transform_to_packet.value
        else:
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
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
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
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class BarrierOperation(BaseOperation):
    __current_barriers = []

    def __init__(self, rank: int, tb_list: List[int]):
        for _ in range(len(BarrierOperation.__current_barriers), rank + 1):
            BarrierOperation.__current_barriers.append({})
        barrier_info = BarrierOperation.BarrierInfo(tb_list)

        if barrier_info not in BarrierOperation.__current_barriers[rank]:
            self.barrier_id = len(BarrierOperation.__current_barriers[rank])
            BarrierOperation.__current_barriers[rank][barrier_info] = self.barrier_id
        else:
            self.barrier_id = BarrierOperation.__current_barriers[rank][barrier_info]

        self.name = Instruction.barrier.value
        self.barrier_info = barrier_info

    def to_json(self):
        result = {"name": self.name}
        result["barrier_id"] = self.barrier_id
        result["num_threadblocks"] = len(self.barrier_info.tb_list)

        return result

    class BarrierInfo:
        def __init__(self, tb_list):
            self.tb_list = tb_list

        def __eq__(self, other):
            return self.tb_list == other.tb_list

        def __hash__(self):
            return hash(tuple(self.tb_list))


@dataclass
class FlushOperation(BaseOperation):
    def __init__(self, channels_ids: List[int], channel_type: ChannelType):
        self.name = Instruction.flush.value
        self.channel_ids = channels_ids
        self.channel_type = channel_type

    def to_json(self):
        result = {"name": self.name}
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class GetOperation(BaseOperation):
    def __init__(
        self,
        src_buff: List[RemoteChunk],
        dst_buff: List[LocalChunk],
        channel_ids: List[int],
        channel_type: ChannelType,
    ):
        self.name = Instruction.get.value
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
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class PutOperation(BaseOperation):
    def __init__(
        self,
        src_buff: List[LocalChunk],
        dst_buff: List[RemoteChunk],
        channel_ids: List[int],
        channel_type: ChannelType,
        from_packet: bool = False,
        to_packet: bool = False,
        with_signal: bool = False,
        with_signal_and_flush: bool = False,
    ):
        if from_packet and to_packet:
            self.name = Instruction.read_put_packet.value
        elif to_packet:
            self.name = Instruction.put_packet.value
        elif from_packet:
            raise RuntimeError(f"Put Operation from Packet is not Supported.")
        else:
            if with_signal:
                if with_signal_and_flush:
                    self.name = Instruction.put_with_signal_and_flush.value
                else:
                    self.name = Instruction.put_with_signal.value
            elif with_signal_and_flush:
                self.name = Instruction.put_with_signal_and_flush.value
            else:
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
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class ReduceOperation(BaseOperation):
    def __init__(
        self,
        local_src_buff: List[LocalChunk],
        local_dst_buff: List[LocalChunk],
        remote_src_buff: List[RemoteChunk] = [],
        remote_dst_buff: List[RemoteChunk] = [],
        channel_ids: List[int] = [],
        channel_type: ChannelType = ChannelType.none,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
        packet: bool = False,
    ):
        if len(remote_src_buff) == 0 and len(remote_dst_buff) == 0:
            if packet:
                self.name = Instruction.reduce_copy_packet.value
            else:
                self.name = Instruction.reduce_copy.value
        elif len(remote_src_buff) == 0:
            if packet:
                self.name = Instruction.reduce_copy_send_packet.value
            else:
                self.name = Instruction.reduce_copy_send.value
        elif len(remote_dst_buff) == 0 and not packet:
            self.name = Instruction.read_reduce_copy.value
        elif not packet:
            self.name = Instruction.read_reduce_copy_send.value
        else:
            raise RuntimeError(f"Reduce Operation invalid parameters.")

        self.local_src_buff = local_src_buff
        self.local_dst_buff = local_dst_buff
        self.remote_src_buff = remote_src_buff
        self.remote_dst_buff = remote_dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def to_json(self):
        result = {"name": self.name}
        result["src_buff"] = []
        for chunk in self.local_src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.local_dst_buff:
            result["dst_buff"].append(chunk.to_json())

        if len(self.remote_src_buff) > 0:
            for chunk in self.remote_src_buff:
                result["src_buff"].append(chunk.to_json())
        if len(self.remote_dst_buff) > 0:
            for chunk in self.remote_dst_buff:
                result["dst_buff"].append(chunk.to_json())

        if len(self.channel_ids) > 0:
            result["channel_ids"] = self.channel_ids
        if self.channel_type != ChannelType.none:
            result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result


@dataclass
class GroupLoadReduce(BaseOperation):
    def __init__(
        self,
        buffer_type: BufferType,
        buffer_offset: int,
        size: int,
        dst_chunk: NVLSChunk,
        tb_channel_id: List[int] = [],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        self.name = Instruction.group_load_reduce.value
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.dst_chunk = dst_chunk
        self.tb_channel_id = tb_channel_id
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def to_json(self):
        result = {"name": self.name}
        result["buffer_type"] = self.buffer_type.value
        result["buffer_offset"] = self.buffer_offset
        result["size"] = self.size
        result["dst_chunk"] = self.dst_chunk.to_json()
        result["channel_ids"] = self.tb_channel_id
        result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result


@dataclass
class GroupStore(BaseOperation):
    def __init__(
        self,
        src_chunk: NVLSChunk,
        buffer_type: BufferType,
        buffer_offset: int,
        size: int,
        tb_channel_id: List[int] = [],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        self.name = Instruction.group_store.value
        self.src_chunk = src_chunk
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.tb_channel_id = tb_channel_id
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def to_json(self):
        result = {"name": self.name}
        result["src_chunk"] = self.src_chunk.to_json()
        result["buffer_type"] = self.buffer_type.value
        result["buffer_offset"] = self.buffer_offset
        result["size"] = self.size
        result["channel_ids"] = self.tb_channel_id
        result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result
