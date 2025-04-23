# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from mscclpp.language.collectives import Collective
from mscclpp.language.buffer import *
from mscclpp.language.types import DataFormat, ChannelType, ChunkRef, ReplicationPolicy, Threadblock
from mscclpp.language.ir import *
from mscclpp.language.dag import DagOptimizer, DagLower, InstructionDAG
from mscclpp.language.rank import Rank
from mscclpp.language.topo_sort import SortDAG
from mscclpp.language.testator import Testator

_current_program = None


def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program


# For msccl++ program, we have one assumption that for channel can be identified by (send_buffer, recv_buffer, type, send_tb/recv_tb)
# which means the send_tb and recv_tb should be the same for a pair of signal and wait, also same for put/get operation.
# If one sender what to send data to peer want to use different tb in receiver side. We need to send to same tb in receiver side first,
# then performance a across tb sync. This is a limitation of current implementation.
class MSCCLPPProgram:
    def __init__(
        self,
        name: str,
        collective: Collective,
        num_ranks: int,
        instances: int,
        protocol: str = "Simple",
        instr_fusion: bool = True,
        replication_policy: ReplicationPolicy = ReplicationPolicy.duplicated,
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
    ):
        self.name = name
        self.collective = collective
        self.num_ranks = num_ranks
        self.instances = instances
        self.protocol = protocol
        self.instr_fusion = instr_fusion
        self.replication_policy = replication_policy
        self.num_threads_per_block = num_threads_per_block
        self.use_double_scratch_buffer = use_double_scratch_buffer
        self.min_message_size = min_message_size
        self.max_message_size = max_message_size
        assert protocol == "Simple" or protocol == "LL", f"Given protocol: {protocol}. Must be either Simple, LL"
        self.run_opt = True  # Runs optimization passes
        # Initialize the input buffers
        self.buffers = collective.init_buffers()
        self.instr_dag = InstructionDAG(self.num_ranks, self.buffers)
        self.sort_dag = SortDAG()
        self.testator = Testator()
        self.ranks = []
        for r in range(self.num_ranks):
            self.ranks.append(Rank(r))
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                ref = self.get_ref(r, buffer, index, 1)
                # self.chunk_dag.init_chunk(chunk, ref)
                self.instr_dag.add_start(r, buffer, index, ref)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCLPP Program in context")
        _current_program = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

    def _convert_to_execution_plan(self):
        ops = self.instr_dag.convert_set_list()
        ops = sorted(ops, key=lambda x: x.step)
        for op in ops:
            rank = op.rank
            tbid = op.tb
            if tbid not in self.instr_dag.tbs[rank]:
                self.instr_dag.tbs[rank][tbid] = Threadblock(id=tbid)
            tb = self.instr_dag.tbs[rank][tbid]
            tb.ops.append(op)

    def get_rank_ref(self, rank):
        return RankRef(rank, self)

    def get_ref(self, rank, buffer, index, size):
        buffer, index = self.collective.get_buffer_index(rank, buffer, index)
        return Ref(rank, buffer, index, size, self)

    def get_chunks(self, rank, buffer, index, size=1):
        chunks = [None] * size
        for i in range(0, size):
            if self.buffers[rank][buffer] and index + i < len(self.buffers[rank][buffer]):
                chunks[i] = self.buffers[rank][buffer][index + i]
            else:
                chunks[i] = None
        return chunks

    def check_buffer_exists(self, rank, name):
        if name not in self.buffers[rank]:
            self.buffers[rank][name] = BufferSlice(Buffer.scratch, name)

    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def check(self):
        return self.testator.check(self.collective, self.sort_dag.operation_order())

    # Lower program to MSCCLPP
    def lower(self):
        self._convert_to_execution_plan()
        self.instr_dag.complete_channels()
        dag_optimizer = DagOptimizer(self.instr_dag)
        dag_optimizer.remove_redundant_signal_wait()
        if self.instr_fusion:
            dag_optimizer.fuse_instructions()
        dag_lower = DagLower(self.instr_dag)
        gpu_prgms = dag_lower.lower(self.instances, self.replication_policy)
        program = Program(
            self.name,
            self.collective.name,
            self.collective.inplace,
            self.protocol,
            gpu_prgms,
            self.collective.num_chunk_groups * self.instances,
            self.num_threads_per_block,
            self.use_double_scratch_buffer,
            self.min_message_size,
            self.max_message_size,
        )
        for gpu in program.gpus:
            gpu.input_chunks = len(self.buffers[gpu.rank][Buffer.input]) * self.instances
            gpu.output_chunks = self.collective.get_output_chunk_count(
                len(self.buffers[gpu.rank][Buffer.output]), self.instances
            )
        return program

    def generate_json(self):
        return ir_to_json(self.lower())


def Json():
    print(_curr().generate_json())


@dataclass
class RankRef:
    rank: int
    prog: MSCCLPPProgram

    def _get_barrier_id(self, tb_list) -> int:
        return self.prog.ranks[self.rank].get_barrier_id(tb_list)

    def barrier(self, tb_list):
        barrier_id = self._get_barrier_id(tb_list)
        return self.prog.instr_dag.add_barrier(self.rank, tb_list, barrier_id)


@dataclass
class Ref(ChunkRef):
    prog: MSCCLPPProgram

    def __repr__(self):
        return f"Ref(Buffer:{self.buffer}, Index:{self.index}, Size:{self.size}, Rank:{self.rank})"

    def _end(self):
        return self.index + self.size

    def _get_chunk(self, index):
        return self.prog.buffers[self.rank][self.buffer][index]

    def split(self, num):
        assert self.size % num == 0, f"Trying to split a chunk of {self.size} elements into {num} parts"
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = self.prog.get_ref(self.rank, self.buffer, index, size)
        return chunks

    def group(self, other):
        assert self.rank == other.rank, f"Trying to concatenate chunks on ranks {self.rank} and {other.rank}"
        assert self.buffer == other.buffer, f"Trying to concatenate chunks in {self.buffer} and {other.buffer}"
        if self.index < other.index:
            first = self
            second = other
        else:
            first = other
            second = self

        end = max(first._end(), second._end())
        return Ref(self.rank, self.buffer, first.index, end - first.index, self.prog)

    def _get_buffer_index(self, remote_rank, buffer, index):
        if index == -1 and buffer == None:
            return self.buffer, self.index
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            return buffer, self.prog.buffers[remote_rank][buffer].instance_size()
        return buffer, index

    def _put(
        self,
        dst,
        buffer=None,
        index=-1,
        sendtb=-1,
        src_format=DataFormat.raw,
        chan_type=ChannelType.memory,
        use_packet=False,
    ):
        self.prog.check_buffer_exists(dst, buffer)
        assert self.rank != dst, "Cannot put to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        if use_packet:
            self.prog.instr_dag.add_put(self.rank, self, dst_chunkref, sendtb, src_format, chan_type, True)
            self.prog.instr_dag.add_signal(self.rank, self, dst_chunkref, -1, ChannelType.none)
            self.prog.instr_dag.add_wait(dst, dst_chunkref, self, -1, ChannelType.none)
        else:
            self.prog.instr_dag.add_put(self.rank, self, dst_chunkref, sendtb, src_format, chan_type)
        return dst_chunkref

    def put(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.memory):
        op = Op(inst=Instruction.put, rank=self.rank, src=self, dst=ChunkRef(dst, buffer, index, self.size))
        self.prog.sort_dag.insert_operation(op)

        return self._put(dst, buffer, index, sendtb, DataFormat.raw, chan_type)

    def put_packet(
        self,
        dst,
        buffer=None,
        index=-1,
        sendtb=-1,
        src_format=DataFormat.raw,
        chan_type=ChannelType.memory,
        temp_buffer=None,
        temp_buffer_index=-1,
    ):
        extra = {"src_format": src_format, "temp_buffer": temp_buffer, "temp_buffer_index": temp_buffer_index}
        op = Op(
            inst=Instruction.put_packet,
            rank=self.rank,
            src=self,
            dst=ChunkRef(dst, buffer, index, self.size),
            extra=extra,
        )
        self.prog.sort_dag.insert_operation(op)

        chunk_ref = self
        if chan_type == ChannelType.port and src_format == DataFormat.raw:
            assert temp_buffer is not None, "Need to specify a temporary buffer for port channels"
            chunk_ref = self._copy(
                self.rank, temp_buffer, temp_buffer_index, sendtb, trans_from_packet=False, trans_to_packet=True
            )
        return chunk_ref._put(dst, buffer, index, sendtb, src_format, chan_type, True)

    def get(self, src, buffer=None, index=-1, recvtb=-1, chan_type=ChannelType.memory):
        op = Op(inst=Instruction.get, rank=self.rank, src=ChunkRef(src, buffer, index, self.size), dst=self)
        self.prog.sort_dag.insert_operation(op)

        self.prog.check_buffer_exists(src, buffer)
        sender = src
        receiver = self.rank
        assert sender != receiver, "Cannot get from the same rank"
        buffer, index = self._get_buffer_index(src, buffer, index)

        src_chunkref = self.prog.get_ref(src, buffer, index, self.size)

        self.prog.instr_dag.add_get(receiver, src_chunkref, self, recvtb, chan_type)

    # for signal and wait, currently we assuem the pair will use the same tb index. In future we need
    # to infer the tb index from the instruction DAG Add a channel is define as (send_tb, src_buffer, recv_tb, dst_buffer, type).
    # Then we can use DAG info to reduce the number of channels.
    def signal(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.memory):
        op = Op(inst=Instruction.signal, rank=self.rank, src=self, dst=ChunkRef(dst, buffer, index, self.size))
        self.prog.sort_dag.insert_operation(op)

        sender = self.rank
        receiver = dst
        assert sender != receiver, "Cannot signal to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        self.prog.instr_dag.add_signal(sender, self, dst_chunkref, sendtb, chan_type)

    # only port channel need to use this function
    def flush(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.port):
        op = Op(inst=Instruction.flush, rank=self.rank, src=self, dst=ChunkRef(dst, buffer, index, self.size))
        self.prog.sort_dag.insert_operation(op)

        assert chan_type == ChannelType.port, "Only port channel can use flush"
        sender = self.rank
        receiver = dst
        assert sender != receiver, "Cannot flush to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        self.prog.instr_dag.add_flush(sender, self, dst_chunkref, sendtb)

    def wait(self, src, buffer=None, index=-1, recvtb=-1, chan_type=ChannelType.memory):
        op = Op(inst=Instruction.wait, rank=self.rank, src=ChunkRef(src, buffer, index, self.size), dst=self)
        self.prog.sort_dag.insert_operation(op)

        sender = src
        receiver = self.rank
        assert sender != receiver, "Cannot wait on the same rank"
        buffer, index = self._get_buffer_index(src, buffer, index)

        src_chunkref = self.prog.get_ref(src, buffer, index, self.size)
        self.prog.instr_dag.add_wait(receiver, self, src_chunkref, recvtb, chan_type)

    def _copy(self, dst, buffer=None, index=-1, sendtb=-1, trans_from_packet=False, trans_to_packet=False):
        self.prog.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(dst, buffer, index)

        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        # Check if we are copying the chunk to the same index (easy mistake when we are using inplace)
        if dst_chunkref == self:
            return

        assert self.rank == dst, "Chunk copy only supports intra-rank communication"
        self.prog.instr_dag.add_copy(self.rank, self, dst_chunkref, sendtb, trans_from_packet, trans_to_packet)

        return dst_chunkref

    # Copies the chunk(s) referenced by this chunkref onto Rank dst at location (buffer, index)
    def copy(self, dst, buffer=None, index=-1, sendtb=-1):
        op = Op(inst=Instruction.copy, rank=self.rank, src=self, dst=ChunkRef(dst, buffer, index, self.size))
        self.prog.sort_dag.insert_operation(op)

        return self._copy(dst, buffer, index, sendtb)

    def copy_packet(self, dst, buffer=None, index=-1, sendtb=-1):
        op = Op(inst=Instruction.copy_packet, rank=self.rank, src=self)
        self.prog.sort_dag.insert_operation(op)

        return self._copy(dst, buffer, index, sendtb, trans_from_packet=True, trans_to_packet=False)

    def _reduce(self, other_chunkref, recvtb=-1, channel_type=ChannelType.memory, use_packet=False):
        dst = self.rank
        src = other_chunkref.rank

        if use_packet:
            assert src == dst, "Packet reduce only supports intra-rank communication"

        if src != dst:
            self.prog.instr_dag.add_read_reduce(dst, other_chunkref, self, recvtb, channel_type)
        else:
            self.prog.instr_dag.add_reduce(src, other_chunkref, self, recvtb, use_packet)

        return self

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce(self, other_chunkref, recvtb=-1, channel_type=ChannelType.memory):
        op = Op(inst=Instruction.reduce, rank=self.rank, src=self, dst=other_chunkref)
        self.prog.sort_dag.insert_operation(op)

        return self._reduce(other_chunkref, recvtb, channel_type)

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce_packet(self, other_chunkref, recvtb=-1):
        op = Op(inst=Instruction.reduce_packet, rank=self.rank, src=self, dst=other_chunkref)
        self.prog.sort_dag.insert_operation(op)

        return self._reduce(other_chunkref, recvtb, use_packet=True)

    # """
    # Group operations. These operations are used to perform collective operations across multiple chunks.
    # For now, all chunks must has the same buffer type and offset.
    # """
    # Reads the chunk(s) referenced by other_chunkref and reduce into the chunk referenced by this chunkref
    def group_load_reduce(self, other_chunkrefs: list, recvtb=-1, chan_type=ChannelType.nvls):
        op = Op(inst=Instruction.group_load_reduce, rank=self.rank, src=self, dst=None, srcs=other_chunkrefs)
        self.prog.sort_dag.insert_operation(op)

        assert (
            len(other_chunkrefs) > 0 and chan_type == ChannelType.nvls
        ), "Group load reduce only supports nvls channel"
        nranks_per_node = self.prog.collective.num_ranks_per_node
        for other_chunkref in other_chunkrefs:
            assert (
                self.rank // nranks_per_node == other_chunkref.rank // nranks_per_node
            ), "Group load reduce only supports chunks on the same node"
            assert self.buffer == other_chunkref.buffer, "Group load reduce only supports chunks with the same buffer"
            assert self.index == other_chunkref.index, "Group load reduce only supports chunks with the same index"

        self.prog.instr_dag.add_group_load_reduce(self.rank, other_chunkrefs, self, recvtb, chan_type)
        return self

    # Copies the chunk(s) referenced by this chunkref onto other_chunkrefs
    def group_store(self, dsts: list, index=-1, buffer=None, sendtb=-1, chan_type=ChannelType.nvls):
        extra = {"dsts": dsts, "index": index, "buffer": buffer}
        op = Op(inst=Instruction.group_store, rank=self.rank, src=self, dst=None, extra=extra)
        self.prog.sort_dag.insert_operation(op)

        for dst in dsts:
            self.prog.check_buffer_exists(dst, buffer)
        assert index == -1 or self.index == index, "Group store only supports chunks with the same index"
        assert chan_type == ChannelType.nvls, "Group store only supports nvls channel"

        other_chunkrefs = []
        nrank_per_node = self.prog.collective.num_ranks_per_node
        for dst in dsts:
            # Direct linked
            buffer, index = self._get_buffer_index(dst, buffer, index)

            assert self.buffer == buffer, "Group store only supports chunks with the same buffer"
            assert (
                self.rank // nrank_per_node == dst // nrank_per_node
            ), "Group store only supports chunks on the same node"

            dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
            other_chunkrefs.append(dst_chunkref)
        # add new op here
        self.prog.instr_dag.add_group_store(self.rank, self, other_chunkrefs, sendtb, chan_type)

    def get_origin_index(self, index=0):
        return self._get_chunk(index + self.index).origin_index

    def get_origin_rank(self, index=0):
        return self._get_chunk(index + self.index).origin_rank

    def get_dst_index(self, index=0):
        return self._get_chunk(index + self.index).dst_index

    def get_dst_rank(self, index=0):
        return self._get_chunk(index + self.index).dst_rank

    def print_chunk_info(self, index=0):
        print(self._get_chunk(index + self.index))


def chunk(rank, buffer, index, size=1) -> Ref:
    if _curr().buffers[rank][buffer][index] is None:
        return None
    return _curr().get_ref(rank, buffer, index, size)


def rank(rank) -> RankRef:
    return _curr().get_rank_ref(rank)


def Check():
    return _curr().check()
