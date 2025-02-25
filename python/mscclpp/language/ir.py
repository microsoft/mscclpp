# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional, Union

from mscclpp.language.types import Buffer, ChannelType, Op, Program, Instruction

_local_src_insts_mscclpp: set = {
    Instruction.put,
    Instruction.put_packet,
    Instruction.signal,
    Instruction.flush,
    Instruction.put_with_signal,
    Instruction.put_with_signal_and_flush,
    Instruction.copy,
    Instruction.copy_packet,
    Instruction.transform_to_packet,
    Instruction.reduce,
    Instruction.reduce_packet,
    Instruction.reduce_send,
    Instruction.reduce_send_packet,
    Instruction.group_load_reduce_store,
    Instruction.group_store,
}
_local_dst_insts_mscclpp: set = {
    Instruction.get,
    Instruction.wait,
    Instruction.read_reduce_copy,
    Instruction.copy,
    Instruction.copy_packet,
    Instruction.transform_to_packet,
    Instruction.reduce,
    Instruction.read_reduce_copy_send,
    Instruction.reduce_send,
    Instruction.reduce_packet,
    Instruction.reduce_send_packet,
    Instruction.group_load_reduce_store,
    Instruction.group_load_reduce,
}

_insts_no_need_sync_barrier: set = {
    Instruction.copy_packet,
    Instruction.reduce_packet,
    Instruction.reduce_send_packet,
    Instruction.barrier,
}


def ir_to_json(program: Program):
    # Figure out sizes of buffers based on usage
    buffer_sizes = defaultdict(lambda: 0)
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst in _local_src_insts_mscclpp:
                    key = (gpu.rank, op.src.buffer)
                    buffer_sizes[key] = max(buffer_sizes[key], op.src.index + op.src.size)
                    for src in op.srcs:
                        key = (gpu.rank, src.buffer)
                        buffer_sizes[key] = max(buffer_sizes[key], src.index + src.size)
                if op.inst in _local_dst_insts_mscclpp:
                    key = (gpu.rank, op.dst.buffer)
                    buffer_sizes[key] = max(buffer_sizes[key], op.dst.index + op.dst.size)
                    # ignore remote buffers
                    if (
                        op.inst != Instruction.read_reduce_copy_send
                        and op.inst != Instruction.reduce_send
                        and op.inst != Instruction.reduce_send_packet
                    ):
                        for dst in op.dsts:
                            key = (gpu.rank, dst.buffer)
                            buffer_sizes[key] = max(buffer_sizes[key], dst.index + dst.size)
    for gpu in program.gpus:
        gpu.input_chunks = max(buffer_sizes[(gpu.rank, Buffer.input)], gpu.input_chunks)
        gpu.output_chunks = max(buffer_sizes[(gpu.rank, Buffer.output)], gpu.output_chunks)
        gpu.scratch_chunks = max(buffer_sizes[(gpu.rank, Buffer.scratch)], gpu.scratch_chunks)

    # Since LL protocol will double the scratch size. We need to make sure all GPUs have the same scratch size.
    # Otherwise the offset calculation will be wrong.
    if program.protocol == "LL":
        max_scratch = max(gpu.scratch_chunks for gpu in program.gpus)
        for gpu in program.gpus:
            gpu.scratch_chunks = max_scratch

    # get channel info for each GPU and threadblock
    for gpu in program.gpus:
        gpu.threadblocks = sorted(gpu.threadblocks, key=lambda tb: tb.id)
        chan_dict = {}
        # the channel key is the tuple (srcBuffer, dstBuffer, type)
        for tb in gpu.threadblocks:
            for ch in tb.channels:
                key = (ch.srcBuffer, ch.dstBuffer, ch.type)
                if key not in chan_dict:
                    chan_dict[key] = [(tb.id, ch.connected_to)]
                else:
                    chan_dict[key].append((tb.id, ch.connected_to))
        for key, value in chan_dict.items():
            chan_dict[key] = sorted(value)
        gpu.channels = chan_dict

    # Remove the dependencies of wait after signal. They are actually depends on remote chunk
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst == Instruction.wait:
                    op.depends = list(filter(lambda dep: dep.inst != Instruction.signal, op.depends))

    # Filter out redundant dependencies
    # e.g. if op1 and op2 depend on op, and op1 happens before op2
    # then op2 does not need to explicitly depend on op
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            running_depends = []
            for op in tb.ops:
                op.depends = list(filter(lambda dep: dep not in running_depends, op.depends))
                running_depends = running_depends + op.depends

    # Do some additional postprocessing of operations:
    # - Expand operations with dependencies with no-ops
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            new_ops = []
            for op in tb.ops:
                if op.inst in _insts_no_need_sync_barrier:
                    new_ops.append(op)
                    continue
                # Expand extra dependencies into nop operations
                nop = Op(Instruction.nop, -1, None, None, [])
                for i, dep in enumerate(op.depends):
                    # barrier already syncs all threads, only sync within the same threadblock
                    if dep.inst != Instruction.barrier and dep.tb == op.tb:
                        nop.depends.append(dep)
                if len(new_ops) > 0 and (
                    new_ops[-1].inst == Instruction.barrier or new_ops[-1].inst == Instruction.nop
                ):
                    new_ops[-1].depends.extend(nop.depends)
                elif len(nop.depends) > 0:
                    new_ops.append(nop)
                new_ops.append(op)
            tb.ops = new_ops

    # update step and tid for ops
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for i, op in enumerate(tb.ops):
                op.step = i
                op.tb = tb.id

    # Need to calculate channel info for each GPU
    nchannels = 0
    for gpu in program.gpus:
        max_tb_channels = 0
        if len(gpu.threadblocks) > 0:
            max_tb_channels = max(tb.channel + 1 for tb in gpu.threadblocks)
        nchannels = max(nchannels, max_tb_channels)
    return _dump_to_json(program)


@dataclass
class _JsonInstruction:
    name: str
    i_buff: Optional[Dict[str, str]] = None
    i_cids: Optional[List[Dict[str, Union[int, List[int]]]]] = None
    o_buff: Optional[Dict[str, str]] = None
    o_cids: Optional[List[Dict[str, Union[int, List[int]]]]] = None
    src: Optional[int] = None
    srcs: Optional[List[Dict[str, Union[int, str]]]] = None
    srcbuff: Optional[str] = None
    srcoff: Optional[int] = None
    dst: Optional[int] = None
    dsts: Optional[List[Dict[str, Union[int, str]]]] = None
    dstbuff: Optional[str] = None
    dstoff: Optional[int] = None
    ctype: Optional[str] = None
    cnt: Optional[int] = None
    deps: Optional[List[Dict[str, int]]] = None
    nthread_blocks: Optional[int] = None
    barrier_id: Optional[int] = None


class _OpConverter(ABC):
    def get_channel_ids(self, rank_list, offset_list, tb_channel_dict, src_buffer, dst_buffer, chan_type):
        channel_ids = []
        key = (src_buffer, dst_buffer, chan_type)
        if chan_type == ChannelType.nvls:
            ranks = []
            for c in rank_list:
                ranks.append(c.rank)
            channel_ids.extend(
                [{"id": id} for id, ele in enumerate(tb_channel_dict[key]["connectedTo"]) if set(ele) == set(ranks)]
            )
        else:
            for c in range(len(rank_list)):
                rank = rank_list[c].rank
                index = offset_list[c].index
                channel_ids.extend(
                    [
                        {"id": id, "off": index}
                        for id, ele in enumerate(tb_channel_dict[key]["connectedTo"])
                        if ele == rank
                    ]
                )
        return channel_ids

    @abstractmethod
    def to_json(self, op: Op) -> _JsonInstruction:
        pass


class _SignalFlushConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        dst_channel_ids = self.get_channel_ids(
            op.dsts, op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        o_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        return _JsonInstruction(
            name=op.inst.value,
            o_buff=o_buff,
            o_cids=dst_channel_ids,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _WaitConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src_channel_ids = self.get_channel_ids(
            op.srcs, op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        return _JsonInstruction(
            name=op.inst.value,
            i_buff=i_buff,
            i_cids=src_channel_ids,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _ReadReduceCopyConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src_channel_ids = self.get_channel_ids(
            op.srcs, op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        dst = op.dst
        src = op.dst  # TODO(binyli): fix this
        return _JsonInstruction(
            name=op.inst.value,
            i_buff=i_buff,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            i_cids=src_channel_ids,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _ReadReduceCopySendConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src_channel_ids = self.get_channel_ids(
            op.srcs, op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        dst_channel_ids = self.get_channel_ids(
            op.dsts, tb_channel_dict, op.dst.buffer, op.dsts[0].buffer, op.channel_type
        )
        i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        o_buff = {"src": op.dst.buffer.value, "dst": op.dsts[0].buffer.value}
        dst = op.dst
        src = op.dst  # TODO(binyli): fix this
        return _JsonInstruction(
            name=op.inst.value,
            i_buff=i_buff,
            i_cids=src_channel_ids,
            o_buff=o_buff,
            o_cids=dst_channel_ids,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _ReduceSendConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        dst_channel_ids = self.get_channel_ids(
            op.dsts, op.srcs, tb_channel_dict, op.dst.buffer, op.dsts[0].buffer, ChannelType.memory
        )
        o_buff = {"src": op.dst.buffer.value, "dst": op.dsts[0].buffer.value}
        srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
        dst = op.dst
        src = op.dst  # TODO(binyli): fix this
        return _JsonInstruction(
            name=op.inst.value,
            o_buff=o_buff,
            o_cids=dst_channel_ids,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            srcs=srcs,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _ReduceConverters(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
        dst = op.dst
        src = op.dst
        return _JsonInstruction(
            name=op.inst.value,
            srcs=srcs,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _NopConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        return _JsonInstruction(
            name=op.inst.value,
            deps=sorted(
                list(map(lambda dep: {"tb": dep.tb, "step": dep.step}, op.depends)), key=lambda x: (x["tb"], x["step"])
            ),
        )


class _BarrierConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        return _JsonInstruction(
            name=op.inst.value,
            nthread_blocks=len(op.extra["tb_list"]),
            barrier_id=op.extra["barrier_id"],
        )


class _PutConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        dst_channel_ids = self.get_channel_ids(
            op.dsts, op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        o_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
        return _JsonInstruction(
            name=op.inst.value,
            o_buff=o_buff,
            o_cids=dst_channel_ids,
            srcs=srcs,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _GetConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src_channel_ids = self.get_channel_ids(
            op.dsts, op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
        dsts = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.dsts))
        return _JsonInstruction(
            name=op.inst.value,
            i_buff=i_buff,
            i_cids=src_channel_ids,
            dsts=dsts,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _CopyConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src = op.src
        dst = op.dst
        return _JsonInstruction(
            name=op.inst.value,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


class _GroupLoadReduceStoreConverter(_OpConverter):
    def to_json(self, op: Op, tb_channel_dict: dict) -> _JsonInstruction:
        src = op.src
        dst = op.dst
        src_channel_ids = self.get_channel_ids(
            op.srcs, op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        dst_channel_ids = self.get_channel_ids(
            op.dsts, op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
        )
        return _JsonInstruction(
            name=op.inst.value,
            src=src.rank,
            srcbuff=src.buffer.value,
            srcoff=src.index,
            dst=dst.rank,
            dstbuff=dst.buffer.value,
            dstoff=dst.index,
            i_cids=src_channel_ids,
            o_cids=dst_channel_ids,
            ctype=op.channel_type.value,
            cnt=op.cnt(),
        )


_json_converter_map: Dict[Instruction, _OpConverter] = {
    Instruction.signal: _SignalFlushConverter(),
    Instruction.flush: _SignalFlushConverter(),
    Instruction.wait: _WaitConverter(),
    Instruction.read_reduce_copy: _ReadReduceCopyConverter(),
    Instruction.read_reduce_copy_send: _ReadReduceCopySendConverter(),
    Instruction.reduce_send: _ReduceSendConverter(),
    Instruction.reduce_send_packet: _ReduceSendConverter(),
    Instruction.reduce: _ReduceConverters(),
    Instruction.reduce_packet: _ReduceConverters(),
    Instruction.nop: _NopConverter(),
    Instruction.barrier: _BarrierConverter(),
    Instruction.put: _PutConverter(),
    Instruction.put_packet: _PutConverter(),
    Instruction.put_with_signal: _PutConverter(),
    Instruction.put_with_signal_and_flush: _PutConverter(),
    Instruction.get: _GetConverter(),
    Instruction.copy: _CopyConverter(),
    Instruction.copy_packet: _CopyConverter(),
    Instruction.transform_to_packet: _CopyConverter(),
    Instruction.group_load_reduce_store: _GroupLoadReduceStoreConverter(),
}


def _dump_to_json(program: Program):
    gpus = []

    def remove_empty_fields(d):
        return {k: v for k, v in d.items() if v not in [None, "", [], {}]}

    max_scratch = max(gpu.scratch_chunks for gpu in program.gpus)
    max_input = max(gpu.input_chunks for gpu in program.gpus)
    max_output = max(gpu.output_chunks for gpu in program.gpus)

    for id, gpu in enumerate(program.gpus):
        gpu_instance = {
            "id": id,
            "inputChunks": gpu.input_chunks,
            "outputChunks": gpu.output_chunks,
            "scratchChunks": gpu.scratch_chunks,
            "chunkGroups": program.num_chunk_groups,
            "threadblocks": [],
            "channels": [],
        }
        for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
            obj = {
                "srcbuff": srcBuffer.value if hasattr(srcBuffer, "value") else srcBuffer,
                "dstbuff": dstBuffer.value if hasattr(dstBuffer, "value") else dstBuffer,
                "type": type.value,
                "connectedTo": [ch[1] for ch in channels],
            }
            if type == ChannelType.nvls:
                obj["connectedTo"] = [sorted(list(peers)) for peers in obj["connectedTo"]]
            gpu_instance["channels"].append(obj)
        gpu_instance["channels"] = list(filter(lambda x: x["type"] != "none", gpu_instance["channels"]))
        gpu_instance["channels"] = sorted(
            gpu_instance["channels"], key=lambda x: (x["srcbuff"], x["dstbuff"], x["type"])
        )

        # render for GPU NVLS channels
        for i, chan in enumerate(gpu_instance["channels"]):
            if chan["type"] == "nvls":
                buff = chan["srcbuff"]
                buffer_size = (
                    max_input
                    if buff == Buffer.input.value
                    else max_output if buff == Buffer.output.value else max_scratch
                )
                gpu_instance["channels"][i] = {
                    "buff": chan["srcbuff"],
                    "type": chan["type"],
                    "rankGroups": [{"size": buffer_size, "ranks": ranks} for ranks in chan["connectedTo"]],
                }

        for tb in gpu.threadblocks:
            if tb.id < 0:
                continue
            ops = []
            tb_channels = []
            tb_channel_dict = {}
            for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
                obj = {
                    "srcbuff": srcBuffer.value if hasattr(srcBuffer, "value") else srcBuffer,
                    "dstbuff": dstBuffer.value if hasattr(dstBuffer, "value") else dstBuffer,
                    "type": type.value,
                    "chanIds": [id for id, ele in enumerate(channels) if ele[0] == tb.id],
                    "connectedTo": [ele[1] for ele in channels if ele[0] == tb.id],
                }
                if len(obj["chanIds"]) > 0:
                    tb_channel_dict[(srcBuffer, dstBuffer, type)] = obj
                    tb_channels.append(obj)
            tb_channels = filter(lambda x: x["type"] != "none", tb_channels)
            tb_channels = sorted(tb_channels, key=lambda x: (x["srcbuff"], x["dstbuff"], x["type"]))
            for op in tb.ops:
                if op.tb == -1:
                    continue
                instr = _json_converter_map[op.inst].to_json(op, tb_channel_dict)
                ops.append(remove_empty_fields(asdict(instr)))
            threadblock = {
                "id": tb.id,
                "ops": ops,
                "channels": list(
                    map(
                        lambda x: {"src": x["srcbuff"], "dst": x["dstbuff"], "ctype": x["type"], "cids": x["chanIds"]},
                        tb_channels,
                    )
                ),
            }
            gpu_instance["threadblocks"].append(threadblock)
        gpus.append(gpu_instance)
    obj = {
        "name": program.name,
        "collective": program.collective,
        "protocol": program.protocol,
        "inplace": program.inplace,
        "gpus": gpus,
        "num_threads_per_block": program.num_threads_per_block,
        "use_double_scratch_buffer": program.use_double_scratch_buffer,
        "min_message_size": program.min_message_size,
        "max_message_size": program.max_message_size,
    }
    return json.dumps(obj, indent=2)
