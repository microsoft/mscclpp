# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import (
    ChannelType,
    Instruction,
    BufferType,
    ReduceOperationType,
    Chunk,
    SyncType,
    DataAccess,
    DataAccessType,
)
from mscclpp.language.thread_block_group import ThreadBlockGroup
from mscclpp.language.loop import LoopIterationContext
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class BaseOperation(ABC):
    """Abstract base class for all MSCCLPP operations.

    This class provides the foundation for all operations that can be performed
    in MSCCLPP programs, including communication operations, synchronization
    operations, and data manipulation operations.

    Attributes:
        id (uuid.UUID): Unique identifier for this operation instance, automatically
            generated using UUID4.
        name (str): The name/type of the operation, typically from the Instruction enum.
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    rank: int
    threadblock: int
    name: str
    # TODO: Only fuse operation with the same pipeline_context
    pipeline_context: LoopIterationContext = field(default=None)

    def local_data_access(self, sync_purpose=True):
        """Get list of local data accesses performed by this operation.

        Returns information about which local memory regions (buffers/chunks)
        this operation reads from or writes to. This is used for dependency
        analysis and optimization.

        Args:
            sync_purpose (bool, optional): Whether this access info is being used
                for synchronization analysis. Defaults to True.

        Returns:
            List[DataAccess]: List of DataAccess objects describing the memory
                regions accessed by this operation. Returns empty list if no
                local data access occurs.
        """
        return []

    def shift_buffers(self, instance, num_instances, replication_function):
        """Shift buffer indices for operation replication across instances.

        When operations are replicated across multiple instances, buffer indices
        need to be adjusted to avoid conflicts. This method applies the replication
        function to all buffer indices used by this operation.

        Args:
            instance (int): The current instance number (0-based).
            num_instances (int): Total number of instances being created.
            replication_function (callable): Function that takes (original_index,
                instance, num_instances) and returns the new index for this instance.
        """
        return

    def shift_ids(self, instance, num_instances, replication_function):
        """Shift resource IDs for operation replication across instances.

        Similar to shift_buffers, but operates on resource IDs like channel IDs,
        semaphore IDs, or barrier IDs that need to be unique across instances.

        Args:
            instance (int): The current instance number (0-based).
            num_instances (int): Total number of instances being created.
            replication_function (callable): Function that takes (original_id,
                instance, num_instances) and returns the new ID for this instance.
        """
        return

    def set_pipeline_context(self, pipeline_context):
        self.pipeline_context = pipeline_context

    def basic_fusion_check(self, other_op):
        return (
            self.rank == other_op.rank
            and self.threadblock == other_op.threadblock
            and self.pipeline_context is other_op.pipeline_context
        )

    def __add__(self, other):
        """Attempt to fuse this operation with another operation.

        Operation fusion is an optimization technique where compatible operations
        can be combined into a single operation to reduce overhead. This method
        implements the fusion logic specific to each operation type.

        Args:
            other (BaseOperation): Another operation to potentially fuse with.
        """
        return None


@dataclass
class LocalChunk:
    type: BufferType
    index: int
    size: int

    def to_dict(self):
        return {"type": self.type.value, "index": self.index, "size": self.size}


@dataclass
class RemoteChunk(LocalChunk):
    buffer_id: int

    def to_dict(self):
        return {"buffer_id": self.buffer_id, "index": self.index, "size": self.size}


class SyncOperation(BaseOperation):
    def __init__(self, rank: int, threadblock: int):
        super().__init__(rank, threadblock, Instruction.nop)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if isinstance(other, SyncOperation):
                fused_operation = SyncOperation(self.rank, self.threadblock)
            elif isinstance(other, BarrierOperation):
                fused_operation = other
            elif isinstance(other, PipelineOperation) and (other.get_data_sync() & SyncType.before) == SyncType.before:
                fused_operation = other
            elif check_data_sync_op(other):
                other.data_sync = other.data_sync ^ (SyncType.before & other.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        return result


class CopyOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        src_buff: List[LocalChunk],
        dst_buff: List[LocalChunk],
        tbg: ThreadBlockGroup = None,
        from_packet: bool = False,
        to_packet: bool = False,
    ):
        if from_packet and to_packet:
            raise RuntimeError(f"Copy Operation from Packet to Packet is not Supported.")
        elif from_packet:
            super().__init__(rank, threadblock, Instruction.unpack_packet)
        elif to_packet:
            super().__init__(rank, threadblock, Instruction.copy_packet)
        else:
            super().__init__(rank, threadblock, Instruction.copy)

        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.tbg = tbg

    def local_data_access(self, order_id, sync_purpose=True):
        data_access = []
        if self.name != Instruction.unpack_packet or not sync_purpose:
            for chunk in self.src_buff:
                data_access.append(
                    DataAccess(
                        self.rank,
                        self.threadblock,
                        self.id,
                        order_id,
                        (
                            chunk.index + self.tbg.start_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else 0
                        ),
                        (
                            chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else chunk.size
                        ),
                        chunk.type,
                        DataAccessType.read,
                        self.tbg,
                        self.pipeline_context,
                    )
                )
        if self.name != Instruction.copy_packet or not sync_purpose:
            for chunk in self.dst_buff:
                data_access.append(
                    DataAccess(
                        self.rank,
                        self.threadblock,
                        self.id,
                        order_id,
                        (
                            chunk.index + self.tbg.start_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else 0
                        ),
                        (
                            chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else chunk.size
                        ),
                        chunk.type,
                        DataAccessType.write,
                        self.tbg,
                        self.pipeline_context,
                    )
                )
        return data_access

    def shift_buffers(self, instance, num_instances, replication_function):
        for chunk in self.src_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.dst_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)

    def to_dict(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_dict())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_dict())
        if self.tbg is not None:
            result["tbg"] = self.tbg.to_dict(self.threadblock)
        return result


class SemaphoreAcquireOperation(BaseOperation):
    def __init__(self, rank: int, threadblock: int, semaphore_ids: List[int], data_sync: SyncType = SyncType.none):
        super().__init__(rank, threadblock, Instruction.sem_acquire)
        self.semaphore_ids = semaphore_ids
        self.data_sync = data_sync
        self.tb_sync = set()

    def add_tb_sync(self, tb):
        self.tb_sync.add(tb)

    def shift_ids(self, instance, num_instances, replication_function):
        for i in range(len(self.semaphore_ids)):
            self.semaphore_ids[i] = replication_function(self.semaphore_ids[i], instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if isinstance(other, SemaphoreAcquireOperation):
                fused_operation = SemaphoreAcquireOperation(
                    self.rank,
                    self.threadblock,
                    semaphore_ids=self.semaphore_ids + other.semaphore_ids,
                    data_sync=self.data_sync | other.data_sync,
                )
            elif (
                (check_data_sync_op(other) and (other.data_sync & SyncType.before) == SyncType.before)
                or (
                    isinstance(other, PipelineOperation)
                    and (other.get_data_sync() & SyncType.before) == SyncType.before
                )
                or isinstance(other, SyncOperation)
                or isinstance(other, BarrierOperation)
            ):
                self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["semaphore_ids"] = list(self.semaphore_ids)
        return result


class SemaphoreReleaseOperation(BaseOperation):
    def __init__(self, rank: int, threadblock: int, semaphore_ids: List[int], data_sync: SyncType = SyncType.none):
        super().__init__(rank, threadblock, Instruction.sem_release)
        self.semaphore_ids = semaphore_ids
        self.data_sync = data_sync

    def shift_ids(self, instance, num_instances, replication_function):
        for i in range(len(self.semaphore_ids)):
            self.semaphore_ids[i] = replication_function(self.semaphore_ids[i], instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if isinstance(other, SemaphoreReleaseOperation):
                fused_operation = SemaphoreReleaseOperation(
                    self.rank,
                    self.threadblock,
                    semaphore_ids=self.semaphore_ids + other.semaphore_ids,
                    data_sync=self.data_sync | other.data_sync,
                )
            elif (
                (check_data_sync_op(other) and (other.data_sync & SyncType.before) == SyncType.before)
                or (
                    isinstance(other, PipelineOperation)
                    and (other.get_data_sync() & SyncType.before) == SyncType.before
                )
                or isinstance(other, SyncOperation)
                or isinstance(other, BarrierOperation)
            ):
                self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["semaphore_ids"] = list(self.semaphore_ids)
        return result


class SignalOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        channels_ids: List[int],
        channel_type: ChannelType,
        data_sync: SyncType = SyncType.none,
        relaxed: bool = False,
    ):
        if relaxed:
            super().__init__(rank, threadblock, Instruction.relaxed_signal)
        else:
            super().__init__(rank, threadblock, Instruction.signal)
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if (
                isinstance(other, SignalOperation)
                and self.channel_type == other.channel_type
                and self.name == other.name
                and not self.channel_ids & other.channel_ids
            ):
                fused_operation = SignalOperation(
                    self.rank,
                    self.threadblock,
                    channels_ids=self.channel_ids | other.channel_ids,
                    channel_type=self.channel_type,
                    data_sync=self.data_sync | other.data_sync,
                    relaxed=(self.name == Instruction.relaxed_signal),
                )
            elif (
                (check_data_sync_op(other) and (other.data_sync & SyncType.before) == SyncType.before)
                or (
                    isinstance(other, PipelineOperation)
                    and (other.get_data_sync() & SyncType.before) == SyncType.before
                )
                or isinstance(other, SyncOperation)
                or isinstance(other, BarrierOperation)
            ):
                self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
        result["channel_type"] = self.channel_type.value
        return result


class WaitOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        channels_ids: List[int],
        channel_type: ChannelType,
        data_sync: SyncType = SyncType.none,
        relaxed: bool = False,
    ):
        if relaxed:
            super().__init__(rank, threadblock, Instruction.relaxed_wait)
        else:
            super().__init__(rank, threadblock, Instruction.wait)
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if (
                isinstance(other, WaitOperation)
                and self.name == other.name
                and not self.channel_ids & other.channel_ids
                and self.channel_type == other.channel_type
            ):
                fused_operation = WaitOperation(
                    self.rank,
                    self.threadblock,
                    channels_ids=self.channel_ids | other.channel_ids,
                    channel_type=self.channel_type,
                    data_sync=self.data_sync | other.data_sync,
                    relaxed=(self.name == Instruction.relaxed_wait),
                )
            elif (
                (check_data_sync_op(other) and (other.data_sync & SyncType.before) == SyncType.before)
                or (
                    isinstance(other, PipelineOperation)
                    and (other.get_data_sync() & SyncType.before) == SyncType.before
                )
                or isinstance(other, SyncOperation)
                or isinstance(other, BarrierOperation)
            ):
                self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
        result["channel_type"] = self.channel_type.value
        return result


class BarrierOperation(BaseOperation):
    __current_barriers = []

    def __init__(self, rank: int, threadblock: int, tb_list: List[int]):
        for _ in range(len(BarrierOperation.__current_barriers), rank + 1):
            BarrierOperation.__current_barriers.append({})
        barrier_info = BarrierOperation.BarrierInfo(tb_list)

        if barrier_info not in BarrierOperation.__current_barriers[rank]:
            self.barrier_id = len(BarrierOperation.__current_barriers[rank])
            BarrierOperation.__current_barriers[rank][barrier_info] = self.barrier_id
        else:
            self.barrier_id = BarrierOperation.__current_barriers[rank][barrier_info]

        super().__init__(rank, threadblock, Instruction.barrier)
        self.barrier_info = barrier_info

    def shift_ids(self, instance, num_instances, replication_function):
        self.barrier_id = replication_function(self.barrier_id, instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if check_data_sync_op(other):
                other.data_sync = other.data_sync ^ (SyncType.before & other.data_sync)
            elif isinstance(other, SyncOperation):
                fused_operation = self

        return fused_operation

    def to_dict(self):
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


class FlushOperation(BaseOperation):
    def __init__(
        self,
        rank: int,
        threadblock: int,
        channels_ids: List[int],
        channel_type: ChannelType,
        data_sync: SyncType = SyncType.none,
    ):
        super().__init__(rank, threadblock, Instruction.flush)
        self.channel_ids = set(channels_ids)
        self.channel_type = channel_type
        self.data_sync = data_sync

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if isinstance(other, FlushOperation) and self.channel_type == other.channel_type:
                fused_operation = FlushOperation(
                    self.rank,
                    self.threadblock,
                    channels_ids=self.channel_ids | other.channel_ids,
                    channel_type=self.channel_type,
                    data_sync=self.data_sync | other.data_sync,
                )
            elif (
                (check_data_sync_op(other) and (other.data_sync & SyncType.before) == SyncType.before)
                or (
                    isinstance(other, PipelineOperation)
                    and (other.get_data_sync() & SyncType.before) == SyncType.before
                )
                or isinstance(other, SyncOperation)
                or isinstance(other, BarrierOperation)
            ):
                self.data_sync = self.data_sync ^ (SyncType.after & self.data_sync)

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["channel_ids"] = list(self.channel_ids)
        result["channel_type"] = self.channel_type.value
        return result


class GetOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        src_buff: List[RemoteChunk],
        dst_buff: List[LocalChunk],
        channel_ids: List[int],
        channel_type: ChannelType,
        tbg: ThreadBlockGroup = None,
    ):
        super().__init__(rank, threadblock, Instruction.get)
        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.tbg = tbg

    def local_data_access(self, order_id, sync_purpose=True):
        data_access = []
        for chunk in self.dst_buff:
            data_access.append(
                DataAccess(
                    self.rank,
                    self.threadblock,
                    self.id,
                    order_id,
                    chunk.index + self.tbg.start_offset(self.threadblock, chunk.size) if self.tbg is not None else 0,
                    (
                        chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                        if self.tbg is not None
                        else chunk.size
                    ),
                    chunk.type,
                    DataAccessType.write,
                    self.tbg,
                    self.pipeline_context,
                )
            )
        return data_access

    def shift_buffers(self, instance, num_instances, replication_function):
        for chunk in self.src_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.dst_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if (
                isinstance(other, GetOperation)
                and self.src_buff[0].size == other.src_buff[0].size
                and self.channel_type == other.channel_type
                and self.tbg == other.tbg
            ):
                fused_operation = GetOperation(
                    self.rank,
                    self.threadblock,
                    src_buff=self.src_buff + other.src_buff,
                    dst_buff=self.dst_buff + other.dst_buff,
                    channel_ids=self.channel_ids + other.channel_ids,
                    channel_type=self.channel_type,
                    tbg=self.tbg,
                )

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_dict())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_dict())
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        if self.tbg is not None:
            result["tbg"] = self.tbg.to_dict(self.threadblock)
        return result


class PutOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        src_buff: List[LocalChunk],
        dst_buff: List[RemoteChunk],
        channel_ids: List[int],
        channel_type: ChannelType,
        tbg: ThreadBlockGroup = None,
        from_packet: bool = False,
        to_packet: bool = False,
        with_signal: bool = False,
        with_signal_and_flush: bool = False,
    ):
        if from_packet and to_packet:
            super().__init__(rank, threadblock, Instruction.read_put_packet)
        elif to_packet:
            super().__init__(rank, threadblock, Instruction.put_packet)
        elif from_packet:
            raise RuntimeError(f"Put Operation from Packet is not Supported.")
        else:
            if with_signal:
                if with_signal_and_flush:
                    super().__init__(rank, threadblock, Instruction.put_with_signal_and_flush)
                else:
                    super().__init__(rank, threadblock, Instruction.put_with_signal)
            elif with_signal_and_flush:
                super().__init__(rank, threadblock, Instruction.put_with_signal_and_flush)
            else:
                super().__init__(rank, threadblock, Instruction.put)

        self.src_buff = src_buff
        self.dst_buff = dst_buff
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.to_packet = to_packet
        self.with_signal = with_signal
        self.with_signal_and_flush = with_signal_and_flush
        self.tbg = tbg

    def local_data_access(self, order_id, sync_purpose=True):
        data_access = []
        if self.name != Instruction.read_put_packet or not sync_purpose:
            for chunk in self.src_buff:
                data_access.append(
                    DataAccess(
                        self.rank,
                        self.threadblock,
                        self.id,
                        order_id,
                        (
                            chunk.index + self.tbg.start_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else 0
                        ),
                        (
                            chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else chunk.size
                        ),
                        chunk.type,
                        DataAccessType.read,
                        self.tbg,
                        self.pipeline_context,
                    )
                )
        return data_access

    def shift_buffers(self, instance, num_instances, replication_function):
        for chunk in self.src_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.dst_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
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
                and self.tbg == other.tbg
            ):
                fused_operation = PutOperation(
                    self.rank,
                    self.threadblock,
                    src_buff=self.src_buff + other.src_buff,
                    dst_buff=self.dst_buff + other.dst_buff,
                    channel_ids=self.channel_ids + other.channel_ids,
                    channel_type=self.channel_type,
                    tbg=self.tbg,
                    to_packet=self.to_packet,
                    with_signal=self.with_signal,
                    with_signal_and_flush=self.with_signal_and_flush,
                )

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.src_buff:
            result["src_buff"].append(chunk.to_dict())
        result["dst_buff"] = []
        for chunk in self.dst_buff:
            result["dst_buff"].append(chunk.to_dict())
        if self.channel_type == ChannelType.port:
            result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        if self.tbg is not None:
            result["tbg"] = self.tbg.to_dict(self.threadblock)
        return result


@dataclass
class ReduceOperation(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        local_src_buff: List[LocalChunk],
        local_dst_buff: List[LocalChunk],
        local_pkt_dst_buff: List[LocalChunk] = None,
        remote_src_buff: List[RemoteChunk] = None,
        remote_dst_buff: List[RemoteChunk] = None,
        channel_ids: List[int] = None,
        put_channel_ids: List[int] = None,
        channel_type: ChannelType = ChannelType.none,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
        tbg: ThreadBlockGroup = None,
        packet: bool = False,
    ):
        local_pkt_dst_buff = local_pkt_dst_buff if local_pkt_dst_buff is not None else []
        remote_src_buff = remote_src_buff if remote_src_buff is not None else []
        remote_dst_buff = remote_dst_buff if remote_dst_buff is not None else []
        channel_ids = channel_ids if channel_ids is not None else []
        put_channel_ids = put_channel_ids if put_channel_ids is not None else []

        if len(remote_src_buff) == 0 and len(remote_dst_buff) == 0:
            if packet:
                if len(local_pkt_dst_buff) == 0:
                    super().__init__(rank, threadblock, Instruction.reduce_packet)
                else:
                    super().__init__(rank, threadblock, Instruction.reduce_copy_packet)
            else:
                super().__init__(rank, threadblock, Instruction.reduce)
        elif len(remote_src_buff) == 0:
            if packet:
                if len(local_pkt_dst_buff) == 0:
                    super().__init__(rank, threadblock, Instruction.reduce_send_packet)
                else:
                    super().__init__(rank, threadblock, Instruction.reduce_copy_send_packet)
            else:
                super().__init__(rank, threadblock, Instruction.reduce_send)
        elif len(remote_dst_buff) == 0 and not packet:
            super().__init__(rank, threadblock, Instruction.read_reduce)
        elif not packet:
            super().__init__(rank, threadblock, Instruction.read_reduce_send)
        else:
            raise RuntimeError(f"Reduce Operation invalid parameters.")

        self.local_src_buff = local_src_buff
        self.local_dst_buff = local_dst_buff
        self.local_pkt_dst_buff = local_pkt_dst_buff
        self.remote_src_buff = remote_src_buff
        self.remote_dst_buff = remote_dst_buff
        self.channel_ids = channel_ids
        self.put_channel_ids = put_channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation
        self.tbg = tbg
        self.packet = packet

    def local_data_access(self, order_id, sync_purpose=True):
        data_access = []
        for i in range(len(self.local_src_buff)):
            chunk = self.local_src_buff[i]
            if not self.packet or i != 0 or not sync_purpose:
                data_access.append(
                    DataAccess(
                        self.rank,
                        self.threadblock,
                        self.id,
                        order_id,
                        (
                            chunk.index + self.tbg.start_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else 0
                        ),
                        (
                            chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                            if self.tbg is not None
                            else chunk.size
                        ),
                        chunk.type,
                        DataAccessType.read,
                        self.tbg,
                        self.pipeline_context,
                    )
                )
        for chunk in self.local_dst_buff:
            data_access.append(
                DataAccess(
                    self.rank,
                    self.threadblock,
                    self.id,
                    order_id,
                    chunk.index + self.tbg.start_offset(self.threadblock, chunk.size) if self.tbg is not None else 0,
                    (
                        chunk.index + self.tbg.end_offset(self.threadblock, chunk.size)
                        if self.tbg is not None
                        else chunk.size
                    ),
                    chunk.type,
                    DataAccessType.write,
                    self.tbg,
                    self.pipeline_context,
                )
            )
        return data_access

    def shift_buffers(self, instance, num_instances, replication_function):
        for chunk in self.local_src_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.local_dst_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.remote_src_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)
        for chunk in self.remote_dst_buff:
            chunk.index = replication_function(chunk.index, chunk.size, instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
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
                and self.tbg == other.tbg
            ):
                fused_operation = ReduceOperation(
                    self.rank,
                    self.threadblock,
                    self.local_src_buff + other.local_src_buff[1:],
                    self.local_dst_buff,
                    remote_src_buff=self.remote_src_buff + other.remote_src_buff,
                    channel_ids=self.channel_ids + other.channel_ids,
                    channel_type=self.channel_type,
                    reduce_operation=self.reduce_operation,
                    tbg=self.tbg,
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
                and self.tbg == other.tbg
            ):
                fused_operation = ReduceOperation(
                    self.rank,
                    self.threadblock,
                    self.local_src_buff,
                    self.local_dst_buff,
                    remote_src_buff=self.remote_src_buff,
                    remote_dst_buff=self.remote_dst_buff + other.dst_buff,
                    channel_ids=self.channel_ids,
                    put_channel_ids=self.put_channel_ids + other.channel_ids,
                    channel_type=self.channel_type,
                    reduce_operation=self.reduce_operation,
                    tbg=self.tbg,
                    packet=self.packet,
                )
            if (
                isinstance(other, PutOperation)
                and (self.name == Instruction.reduce_packet or self.name == Instruction.reduce_send_packet)
                and other.name == Instruction.put_packet
                and self.local_dst_buff[0] == other.src_buff[0]
                and other.channel_type == ChannelType.memory
                and self.tbg == other.tbg
            ):
                fused_operation = ReduceOperation(
                    self.rank,
                    self.threadblock,
                    self.local_src_buff,
                    self.local_dst_buff,
                    remote_src_buff=self.remote_src_buff,
                    remote_dst_buff=self.remote_dst_buff + other.dst_buff,
                    channel_ids=self.channel_ids,
                    put_channel_ids=self.put_channel_ids + other.channel_ids,
                    channel_type=other.channel_type,
                    reduce_operation=self.reduce_operation,
                    tbg=self.tbg,
                    packet=self.packet,
                )
            if (
                isinstance(other, CopyOperation)
                and self.name == Instruction.reduce_packet
                and other.name == Instruction.copy_packet
                and self.local_dst_buff[0] == other.src_buff[0]
                and self.tbg_info == other.tbg_info
            ):
                fused_operation = ReduceOperation(
                    self.rank,
                    self.threadblock,
                    self.local_src_buff,
                    self.local_dst_buff,
                    local_pkt_dst_buff=other.dst_buff,
                    remote_src_buff=self.remote_src_buff,
                    remote_dst_buff=self.remote_dst_buff,
                    channel_ids=self.channel_ids,
                    put_channel_ids=self.put_channel_ids,
                    channel_type=self.channel_type,
                    reduce_operation=self.reduce_operation,
                    tbg_info=self.tbg_info,
                    packet=self.packet,
                )
            if (
                isinstance(other, PutOperation)
                and (self.name == Instruction.reduce_copy_packet or self.name == Instruction.reduce_copy_send_packet)
                and (
                    (other.name == Instruction.put_packet and self.local_dst_buff[0] == other.src_buff[0])
                    or (other.name == Instruction.read_put_packet and self.local_pkt_dst_buff[0] == other.src_buff[0])
                )
                and other.channel_type == ChannelType.memory
                and self.tbg_info == other.tbg_info
            ):
                fused_operation = ReduceOperation(
                    self.rank,
                    self.threadblock,
                    self.local_src_buff,
                    self.local_dst_buff,
                    local_pkt_dst_buff=self.local_pkt_dst_buff,
                    remote_src_buff=self.remote_src_buff,
                    remote_dst_buff=self.remote_dst_buff + other.dst_buff,
                    channel_ids=self.channel_ids,
                    put_channel_ids=self.put_channel_ids + other.channel_ids,
                    channel_type=other.channel_type,
                    reduce_operation=self.reduce_operation,
                    tbg_info=self.tbg_info,
                    packet=self.packet,
                )

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["src_buff"] = []
        for chunk in self.local_src_buff:
            result["src_buff"].append(chunk.to_dict())
        result["dst_buff"] = []
        for chunk in self.local_dst_buff:
            result["dst_buff"].append(chunk.to_dict())
        for chunk in self.local_pkt_dst_buff:
            result["dst_buff"].append(chunk.to_dict())

        if len(self.remote_src_buff) > 0:
            for chunk in self.remote_src_buff:
                result["src_buff"].append(chunk.to_dict())
        if len(self.remote_dst_buff) > 0:
            for chunk in self.remote_dst_buff:
                result["dst_buff"].append(chunk.to_dict())

        if self.channel_type != ChannelType.none:
            result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        if self.tbg is not None:
            result["tbg"] = self.tbg.to_dict(self.threadblock)
        return result


@dataclass
class GroupLoadReduce(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        buffer_type: BufferType,
        buffer_offset: int,
        size: int,
        dst_chunk: Chunk,
        channel_ids: List[int],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        super().__init__(rank, threadblock, Instruction.group_load_reduce)
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.dst_chunk = dst_chunk
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def shift_buffers(self, instance, num_instances, replication_function):
        self.buffer_offset = replication_function(self.buffer_offset, self.size, instance, num_instances)
        self.dst_chunk.index = replication_function(self.dst_chunk.index, self.size, instance, num_instances)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if (
                isinstance(other, GroupStore)
                and self.buffer_type == other.buffer_type
                and self.size == other.size
                and self.dst_chunk == other.src_chunk
                and self.channel_ids == other.channel_ids
                and self.channel_type == other.channel_type
            ):
                fused_operation = GroupLoadReduceStore(
                    self.rank,
                    self.threadblock,
                    buffer_type=self.buffer_type,
                    size=self.size,
                    src_index=[self.buffer_offset],
                    dst_index=[other.buffer_offset],
                    channel_ids=self.channel_ids,
                    channel_type=self.channel_type,
                    reduce_operation=self.reduce_operation,
                )

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["buffer_type"] = self.buffer_type.value
        result["buffer_offset"] = self.buffer_offset
        result["size"] = self.size
        result["dst_chunk"] = self.dst_chunk.to_dict()
        result["channel_ids"] = self.channel_ids
        result["channel_type"] = self.channel_type.value
        result["reduce_op"] = self.reduce_operation.value
        return result


@dataclass
class GroupStore(BaseOperation):
    def __init__(
        self,
        rank,
        threadblock: int,
        src_chunk: Chunk,
        buffer_type: BufferType,
        buffer_offset: int,
        size: int,
        channel_ids: List[int],
        channel_type: ChannelType = ChannelType.switch,
    ):
        super().__init__(rank, threadblock, Instruction.group_store)
        self.src_chunk = src_chunk
        self.buffer_type = buffer_type
        self.buffer_offset = buffer_offset
        self.size = size
        self.channel_ids = channel_ids
        self.channel_type = channel_type

    def shift_buffers(self, instance, num_instances, replication_function):
        self.buffer_offset = replication_function(self.buffer_offset, self.size, instance, num_instances)
        self.src_chunk.index = replication_function(self.src_chunk.index, self.size, instance, num_instances)

    def to_dict(self):
        result = {"name": self.name.value}
        result["src_chunk"] = self.src_chunk.to_dict()
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
        rank,
        threadblock: int,
        buffer_type: BufferType,
        size: int,
        src_index: List[int],
        dst_index: List[int],
        channel_ids: List[int],
        channel_type: ChannelType = ChannelType.switch,
        reduce_operation: ReduceOperationType = ReduceOperationType.sum,
    ):
        super().__init__(rank, threadblock, Instruction.group_load_reduce_store)
        self.buffer_type = buffer_type
        self.size = size
        self.src_index = src_index
        self.dst_index = dst_index
        self.channel_ids = channel_ids
        self.channel_type = channel_type
        self.reduce_operation = reduce_operation

    def shift_buffers(self, instance, num_instances, replication_function):
        for i in range(len(self.src_index)):
            self.src_index[i] = replication_function(self.src_index[i], self.size, instance, num_instances)
        for i in range(len(self.dst_index)):
            self.dst_index[i] = replication_function(self.dst_index[i], self.size, instance, num_instances)

    def to_dict(self):
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


@dataclass
class PipelineOperation(BaseOperation):
    def __init__(self, rank: int, threadblock: int, unit_size: int, num_chunks: int, operations=None):
        super().__init__(rank, threadblock, Instruction.pipeline)
        self.unit_size = unit_size
        self.num_chunks = num_chunks
        self.operations = operations if operations is not None else []

    def _check_sync(self, operation, data_sync):
        result = SyncType.none
        if isinstance(operation, SyncOperation) or isinstance(operation, BarrierOperation):
            result |= SyncType.before
        if check_data_sync_op(operation):
            result |= data_sync & operation.data_sync

        return result

    def add_operation(self, operation):
        self.operations.append(operation)

    def get_data_sync(self):
        data_sync = SyncType.none
        if len(self.operations) > 0:
            data_sync |= self._check_sync(self.operations[0], SyncType.before)
            data_sync |= self._check_sync(self.operations[-1], SyncType.after)

        return data_sync

    def local_data_access(self, sync_purpose=True):
        data_access = []
        for operation in self.operations:
            for operation_data_access in operation.local_data_access(sync_purpose):
                operation_data_access.operation_id = self.id
                data_access.append(operation_data_access)

        return data_access

    def shift_buffers(self, instance, num_instances, replication_function):
        for operation in self.operations:
            operation.shift_buffers(instance, num_instances, replication_function)

    def shift_ids(self, instance, num_instances, replication_function):
        for operation in self.operations:
            operation.shift_ids(instance, num_instances, replication_function)

    def __add__(self, other):
        fused_operation = None
        if self.basic_fusion_check(other):
            if (self.get_data_sync() & SyncType.after) == SyncType.after and check_data_sync_op(other):
                other.data_sync = other.data_sync ^ (SyncType.before & other.data_sync)
            elif isinstance(other, SyncOperation) and (self.get_data_sync() & SyncType.after) == SyncType.after:
                fused_operation = self

        return fused_operation

    def to_dict(self):
        result = {"name": self.name.value}
        result["iter_context"] = {"unit_size": self.unit_size, "num_chunks": self.num_chunks}
        result["ops"] = []
        for operation in self.operations:
            result["ops"].append(operation.to_dict())
        return result


def check_data_sync_op(operation):
    return (
        isinstance(operation, SemaphoreAcquireOperation)
        or isinstance(operation, SemaphoreReleaseOperation)
        or isinstance(operation, SignalOperation)
        or isinstance(operation, WaitOperation)
        or isinstance(operation, FlushOperation)
    )


def add_data_sync(operations):
    result_operations = []
    data_sync_operations = {
        Instruction.sem_acquire,
        Instruction.sem_release,
        Instruction.signal,
        Instruction.wait,
        Instruction.relaxed_signal,
        Instruction.relaxed_wait,
        Instruction.flush,
    }

    for operation in operations:
        if operation.name == Instruction.pipeline:
            pipeline_result_operations = add_data_sync(operation.operations)
            operation.operations = pipeline_result_operations

        if operation.name in data_sync_operations and (
            operation.data_sync == SyncType.before or operation.data_sync == SyncType.both
        ):
            result_operations.append(SyncOperation(operation.rank, operation.threadblock))
        result_operations.append(operation)
        if operation.name in data_sync_operations and (
            operation.data_sync == SyncType.after or operation.data_sync == SyncType.both
        ):
            result_operations.append(SyncOperation(operation.rank, operation.threadblock))

    return result_operations
