from mscclpp.language.internal.dsl_types import (
    ChannelType,
    Instruction,
    BufferType,
    ReduceOperationType,
    Chunk,
    SyncType,
    DataAccess,
    DataAccessType,
)
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
class SyncOperation(BaseOperation):
    def __init__(self):
        self.name = Instruction.nop

    def __add__(self, other):
        fused_operation = None
        if isinstance(other, SyncOperation):
            fused_operation = SyncOperation()

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
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
            self.name = Instruction.copy_packet
        elif to_packet:
            self.name = Instruction.transform_to_packet
        else:
            self.name = Instruction.copy

        self.src_buff = src_buff
        self.dst_buff = dst_buff

    def local_data_access(self):
        access = {}
        for chunk in self.src_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.read
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.read
        for chunk in self.dst_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.write
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.write
        return [DataAccess(*key, value) for key, value in access.items()]

    def __add__(self, other):
        return None

    def to_json(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_json())
        return result


@dataclass
class SignalOperation(BaseOperation):
    def __init__(
        self,
        channels_ids: List[int],
        channel_type: ChannelType,
        data_sync: SyncType = SyncType.none,
        relaxed: bool = False,
    ):
        if relaxed:
            self.name = Instruction.relaxed_signal
        else:
            self.name = Instruction.signal
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, SignalOperation)
            and self.channel_type == other.channel_type
            and self.name == other.name
            and not self.channel_ids & other.channel_ids
        ):
            fused_operation = SignalOperation(
                channels_ids=self.channel_ids | other.channel_ids,
                channel_type=self.channel_type,
                data_sync=self.data_sync | other.data_sync,
                relaxed=(self.name == Instruction.relaxed_signal),
            )
        elif (
            isinstance(other, SignalOperation) or isinstance(other, WaitOperation) or isinstance(other, FlushOperation)
        ):
            if other.data_sync == SyncType.before or other.data_sync == SyncType.both:
                self.data_sync = self.data_sync ^ ((SyncType.after & self.data_sync))
        elif isinstance(other, SyncOperation) or isinstance(other, BarrierOperation):
            self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class WaitOperation(BaseOperation):
    def __init__(
        self,
        channels_ids: List[int],
        channel_type: ChannelType,
        data_sync: SyncType = SyncType.none,
        relaxed: bool = False,
    ):
        if relaxed:
            self.name = Instruction.relaxed_wait
        else:
            self.name = Instruction.wait
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, WaitOperation)
            and self.name == other.name
            and not self.channel_ids & other.channel_ids
            and self.channel_type == other.channel_type
        ):
            fused_operation = WaitOperation(
                channels_ids=self.channel_ids | other.channel_ids,
                channel_type=self.channel_type,
                data_sync=self.data_sync | other.data_sync,
                relaxed=(self.name == Instruction.relaxed_wait),
            )
        elif (
            isinstance(other, SignalOperation) or isinstance(other, WaitOperation) or isinstance(other, FlushOperation)
        ):
            if other.data_sync == SyncType.before or other.data_sync == SyncType.both:
                self.data_sync = self.data_sync ^ ((SyncType.after & self.data_sync))
        elif isinstance(other, SyncOperation) or isinstance(other, BarrierOperation):
            self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
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

        self.name = Instruction.barrier
        self.barrier_info = barrier_info

    def __add__(self, other):
        fused_operation = None
        if isinstance(other, SignalOperation) or isinstance(other, WaitOperation) or isinstance(other, FlushOperation):
            other.data_sync = other.data_sync ^ (SyncType.before & other.data_sync)

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
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
    def __init__(self, channels_ids: List[int], channel_type: ChannelType, data_sync: SyncType = SyncType.none):
        self.name = Instruction.flush
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if isinstance(other, FlushOperation) and self.channel_type == other.channel_type:
            fused_operation = FlushOperation(
                channels_ids=self.channel_ids | other.channel_ids,
                channel_type=self.channel_type,
                data_sync=self.data_sync | other.data_sync,
            )
        elif (
            isinstance(other, SignalOperation) or isinstance(other, WaitOperation) or isinstance(other, FlushOperation)
        ):
            if other.data_sync == SyncType.before or other.data_sync == SyncType.both:
                self.data_sync = self.data_sync ^ ((SyncType.after & self.data_sync))
        elif isinstance(other, SyncOperation) or isinstance(other, BarrierOperation):
            self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
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
        self.name = Instruction.get
        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type

    def local_data_access(self):
        access = {}
        for chunk in self.dst_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.write
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.write
        return [DataAccess(*key, value) for key, value in access.items()]

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, GetOperation)
            and self.src_buff[0].size == other.src_buff[0].size
            and self.channel_type == other.channel_type
        ):
            fused_operation = GetOperation(
                src_buff=self.src_buff + other.src_buff,
                dst_buff=self.dst_buff + other.dst_buff,
                channel_ids=self.channel_ids + other.channel_ids,
                channel_type=self.channel_type,
            )

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
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
            self.name = Instruction.read_put_packet
        elif to_packet:
            self.name = Instruction.put_packet
        elif from_packet:
            raise RuntimeError(f"Put Operation from Packet is not Supported.")
        else:
            if with_signal:
                if with_signal_and_flush:
                    self.name = Instruction.put_with_signal_and_flush
                else:
                    self.name = Instruction.put_with_signal
            elif with_signal_and_flush:
                self.name = Instruction.put_with_signal_and_flush
            else:
                self.name = Instruction.put

        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.to_packet = to_packet
        self.with_signal = with_signal
        self.with_signal_and_flush = with_signal_and_flush

    def local_data_access(self):
        access = {}
        for chunk in self.src_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.read
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.read
        return [DataAccess(*key, value) for key, value in access.items()]

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, PutOperation)
            and (
                self.name == Instruction.put
                or self.name == Instruction.put_packet
                or self.name == Instruction.put_with_signal
                or self.name == Instruction.put_with_signal_and_flush
            )
            and self.name == other.name
            and self.src_buff[0].size == other.src_buff[0].size
            and self.channel_type == other.channel_type
        ):
            fused_operation = PutOperation(
                src_buff=self.src_buff + other.src_buff,
                dst_buff=self.dst_buff + other.dst_buff,
                channel_ids=self.channel_ids + other.channel_ids,
                channel_type=self.channel_type,
                to_packet=self.to_packet,
                with_signal=self.with_signal,
                with_signal_and_flush=self.with_signal_and_flush,
            )

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_json())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_json())
        if self.channel_ids == ChannelType.port:
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
        put_channel_ids: List[int] = [],
        channel_type: ChannelType = ChannelType.none,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
        packet: bool = False,
    ):
        if len(remote_src_buff) == 0 and len(remote_dst_buff) == 0:
            if packet:
                self.name = Instruction.reduce_packet
            else:
                self.name = Instruction.reduce
        elif len(remote_src_buff) == 0:
            if packet:
                self.name = Instruction.reduce_send_packet
            else:
                self.name = Instruction.reduce_send
        elif len(remote_dst_buff) == 0 and not packet:
            self.name = Instruction.read_reduce
        elif not packet:
            self.name = Instruction.read_reduce_send
        else:
            raise RuntimeError(f"Reduce Operation invalid parameters.")

        self.local_src_buff = local_src_buff
        self.local_dst_buff = local_dst_buff
        self.remote_src_buff = remote_src_buff
        self.remote_dst_buff = remote_dst_buff
        self.channel_ids = channel_ids
        self.put_channel_ids = put_channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation
        self.packet = packet

    def local_data_access(self):
        access = {}
        for chunk in self.local_src_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.read
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.read
        for chunk in self.local_dst_buff:
            if chunk.type not in access:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] = DataAccessType.write
            else:
                access[(chunk.index, chunk.index + chunk.size - 1, chunk.type)] |= DataAccessType.write
        return [DataAccess(*key, value) for key, value in access.items()]

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, ReduceOperation)
            and (
                self.name == Instruction.reduce
                or self.name == Instruction.reduce_packet
                or self.name == Instruction.read_reduce
            )
            and self.name == other.name
            and self.local_src_buff[0] == other.local_src_buff[0]
            and self.local_dst_buff == other.local_dst_buff
            and self.channel_type == other.channel_type
            and self.reduce_operation == other.reduce_operation
        ):
            fused_operation = ReduceOperation(
                self.local_src_buff + other.local_src_buff[1:],
                self.local_dst_buff,
                remote_src_buff=self.remote_src_buff + other.remote_src_buff,
                channel_ids=self.channel_ids + other.channel_ids,
                channel_type=self.channel_type,
                reduce_operation=self.reduce_operation,
                packet=self.packet,
            )
        if (
            isinstance(other, PutOperation)
            and (
                self.name == Instruction.reduce
                or self.name == Instruction.reduce_send
                or self.name == Instruction.read_reduce
                or self.name == Instruction.read_reduce_send
            )
            and other.name == Instruction.put
            and self.local_dst_buff[0] == other.src_buff[0]
            and other.channel_type == ChannelType.memory
        ):
            fused_operation = ReduceOperation(
                self.local_src_buff,
                self.local_dst_buff,
                remote_src_buff=self.remote_src_buff,
                remote_dst_buff=self.remote_dst_buff + other.dst_buff,
                channel_ids=self.channel_ids,
                put_channel_ids=self.put_channel_ids + other.channel_ids,
                channel_type=self.channel_type,
                reduce_operation=self.reduce_operation,
                packet=self.packet,
            )
        if (
            isinstance(other, PutOperation)
            and (self.name == Instruction.reduce_packet or self.name == Instruction.reduce_send_packet)
            and other.name == Instruction.put_packet
            and self.local_dst_buff[0] == other.src_buff[0]
            and other.channel_type == ChannelType.memory
        ):
            fused_operation = ReduceOperation(
                self.local_src_buff,
                self.local_dst_buff,
                remote_src_buff=self.remote_src_buff,
                remote_dst_buff=self.remote_dst_buff + other.dst_buff,
                channel_ids=self.channel_ids,
                put_channel_ids=self.put_channel_ids + other.channel_ids,
                channel_type=other.channel_type,
                reduce_operation=self.reduce_operation,
                packet=self.packet,
            )

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
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
        dst_chunk: Chunk,
        channel_ids: List[int] = [],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        self.name = Instruction.group_load_reduce
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.dst_chunk = dst_chunk
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def __add__(self, other):
        fused_operation = None
        if (
            isinstance(other, GroupStore)
            and self.buffer_type == other.buffer_type
            and self.size == other.size
            and self.dst_chunk == other.src_chunk
            and self.channel_ids == other.channel_ids
            and self.channel_type == other.channel_type
        ):
            fused_operation = GroupLoadReduceStore(
                buffer_type=self.buffer_type,
                size=self.size,
                src_index=[self.buffer_offset],
                dst_index=[other.buffer_offset],
                channel_ids=self.channel_ids,
                channel_type=self.channel_type,
                reduce_operation=self.reduce_operation,
            )

        return fused_operation

    def to_json(self):
        result = {"name": self.name.value}
        result["buffer_type"] = self.buffer_type.value
        result["buffer_offset"] = self.buffer_offset
        result["size"] = self.size
        result["dst_chunk"] = self.dst_chunk.to_json()
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result


@dataclass
class GroupStore(BaseOperation):
    def __init__(
        self,
        src_chunk: Chunk,
        buffer_type: BufferType,
        buffer_offset: int,
        size: int,
        channel_ids: List[int] = [],
        channel_type: ChannelType = ChannelType.switch,
    ):
        self.name = Instruction.group_store
        self.src_chunk = src_chunk
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.channel_ids = channel_ids
        self.channel_type = channel_type

    def __add__(self, other):
        return None

    def to_json(self):
        result = {"name": self.name.value}
        result["src_chunk"] = self.src_chunk.to_json()
        result["buffer_type"] = self.buffer_type.value
        result["buffer_offset"] = self.buffer_offset
        result["size"] = self.size
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        return result


@dataclass
class GroupLoadReduceStore(BaseOperation):
    def __init__(
        self,
        buffer_type: BufferType,
        size: int,
        src_index: List[int],
        dst_index: List[int],
        channel_ids: List[int] = [],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        self.name = Instruction.group_load_reduce_store
        self.buffer_type = buffer_type
        self.size = size
        self.src_index = src_index
        self.dst_index = dst_index
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def __add__(self, other):
        return None

    def to_json(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for i in range(len(self.src_index)):
            result["src_buff"].append(
                {"switch_channel_id": self.channel_ids[i], "index": self.src_index[i], "size": self.size}
            )
        result["dst_buff"] = []
        for i in range(len(self.dst_index)):
            result["dst_buff"].append(
                {"switch_channel_id": self.channel_ids[i], "index": self.src_index[i], "size": self.size}
            )
        result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result
