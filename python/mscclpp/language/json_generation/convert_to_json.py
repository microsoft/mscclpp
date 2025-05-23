from mscclpp.channel_based_language.program import MSCCLPPProgram
from mscclpp.channel_based_language.types import BufferType, ChannelType, Instruction
from mscclpp.channel_based_language.json_generation.types import InfoLocation, RemoteBuffer
import json

class _SignalConverter():
    def to_json(self, op, remote_buffer_internal_ids):
        cids = []
        for channel_id in op.channel_ids:
            cids.append(channel_id)

        return {
            "name": "signal",
            "cids": cids,
            "channel_type": op.channel_type.value,
        }
    
class _WaitConverter():
    def to_json(self, op, remote_buffer_internal_ids):
        cids = []
        for channel_id in op.channel_ids:
            cids.append(channel_id)

        return {
            "name": "wait",
            "cids": cids,
            "channel_type": op.channel_type.value,
        }

class _PutConverter():
    def to_json(self, op, remote_buffer_internal_ids):
        src_buffs = []
        for buff in op.local_chunks:
            src_buffs.append({
                "type": buff.buffer.value,
                "offset": buff.index,
                "size": buff.size
            })
        dst_buffs = []
        for buff in op.remote_chunks:
            remotebuff = RemoteBuffer(buff.rank, buff.buffer, InfoLocation.gpu)
            buff_id = remote_buffer_internal_ids[remotebuff]
            dst_buffs.append({
                "buff_id": buff_id,
                "offset": buff.index,
                "size": buff.size
            })
        cids = []
        for channel_id in op.channel_ids:
            cids.append(channel_id)

        return {
            "name": "put",
            "src_buff": src_buffs,
            "dst_buff": dst_buffs,
            "cids": cids,
            "channel_type": op.channel_type.value,
        }

_json_converter_map: {} = {
    Instruction.signal: _SignalConverter(),
    Instruction.wait: _WaitConverter(),
    Instruction.put: _PutConverter()
}

def generate_json(program: MSCCLPPProgram):
    operations = program.instr_dag.retrieve_operations()
    gpus = []
    for rank in range(program.num_ranks):
        id = rank
        input_buffer_size = program.buffers_size[rank][BufferType.input]
        output_buffers_size = program.buffers_size[rank][BufferType.output]
        #scratch_buffers_size = program.buffers_size[rank][BufferType.scratch]
        channels = program.channels[rank]
        remote_buffers = program.instr_dag.retrieve_remote_buffers(rank)
        remote_buffers_json = []
        for remote_buffer in remote_buffers:
            remote_buffers_json.append(remote_buffer.convert_to_json())
        #buffer_alignment = program.collective.get_buffer_alignment(rank)

        remote_buffer_internal_ids = {}
        position = 0
        for remote_buffer in remote_buffers:
            remote_buffer_internal_ids[remote_buffer] = position
            position += 1
        


        num_tb = program.instr_dag.retrieve_num_tb_per_rank(rank)
        threadblock = []
        for tb in range(num_tb):
            thread_block_id = tb
            thread_block_channel_ids = []
            thread_block_remote_buffer_ids = []
            threadblock_ops = []
            for op in operations[rank][tb]:
                if len(op.channel_ids) > 0:
                    thread_block_channel_ids.append(op.channel_ids)
                for remote_chunk in op.remote_chunks:
                    if op.channel_type == ChannelType.port:
                        remote_buffer = RemoteBuffer(remote_chunk.rank, remote_chunk.buffer, InfoLocation.cpu)
                        thread_block_remote_buffer_ids.append(remote_buffer_internal_ids[remote_buffer])
                    else:
                        remote_buffer = RemoteBuffer(remote_chunk.rank, remote_chunk.buffer, InfoLocation.gpu)
                        thread_block_remote_buffer_ids.append(remote_buffer_internal_ids[remote_buffer])

                threadblock_ops.append(_json_converter_map[op.inst].to_json(op, remote_buffer_internal_ids))

            threadblockc_el = {
                "id": thread_block_id,
                "ops": threadblock_ops,
                "channels": thread_block_channel_ids,
                "remoteBuffersIds": thread_block_remote_buffer_ids,
            }
            threadblock.append(threadblockc_el)

        gpu = {
            "id": id,
            "input_buffer_size": input_buffer_size,
            "output_buffers_size": output_buffers_size,
            #"scratch_buffers_size": scratch_buffers_size,
            "threadblocks": threadblock,
            #"channels": channels,
            "remoteBuffers": remote_buffers_json,
            #"bufferAlignment": buffer_alignment,
        }
        gpus.append(gpu)

    obj = {
        "name": program.name,
        "collective": program.collective.name,
        "protocol": program.protocol,
        "inplace": True,
        "gpus": gpus,
        "num_threads_per_block": program.num_threads_per_block,
        "use_double_scratch_buffer": program.use_double_scratch_buffer,
        "min_message_size": program.min_message_size,
        "max_message_size": program.max_message_size,
    }

    return json.dumps(obj, indent=2)
