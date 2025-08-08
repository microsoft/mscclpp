# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.pipeline import *


def alltoall_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    last_step_process = True
    steps_pipeline = 2
    chunks_per_step = 2
    chunksperloop = chunks_per_step * steps_pipeline
    collective = AllToAll(gpu_size, chunksperloop, True)
    with MSCCLPPProgram(
        name,
        collective,
        gpu_size,
        instances=2,
        protocol="Simple",
        reuse_resources=True,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels and Scratch Buffer
        channels = {}
        first_step_channel = {}
        last_step_channel = {}
        sync_channels = {}
        semaphores = {}
        scratch_buffer = {}
        tb_offset = gpu_size - 1
        for gpu in range(gpu_size):
            src_rank_id = gpu
            scratch_buffer[src_rank_id] = Buffer(src_rank_id, chunksperloop * (gpu_size - 1))
            for tb in range(gpu_size):
                semaphores[src_rank_id, tb] = Semaphore(src_rank_id, initial_value=0)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)
                    first_step_channel[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)
                    sync_channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)
                    last_step_channel[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)

        # Initial Synchronization
        for gpus in range(gpu_size):
            src_rank_id = gpus
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != peer:
                    tb = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                    sync_channels[dst_rank_id, src_rank_id].signal(tb=tb, relaxed=True)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    tb = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                    sync_channels[dst_rank_id, src_rank_id].wait(tb=tb, relaxed=True, data_sync=SyncType.after)

        # Copy Data to Scratch Buffer and Put Remote Rank
        for i in range(steps_pipeline):
            if i == 0:
                for gpu in range(gpu_size):
                    src_rank_id = gpu
                    src_rank = Rank(src_rank_id)
                    input_buffer = src_rank.get_input_buffer()
                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            local_index = dst_rank_id * chunksperloop + i * chunks_per_step + chunks_per_step // 2
                            remote_index = (src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1) * chunksperloop + i * chunks_per_step + chunks_per_step // 2
                            tb = tb_offset + (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1)
                            first_step_channel[dst_rank_id, src_rank_id].put(scratch_buffer[dst_rank_id][remote_index: remote_index + chunks_per_step // 2],  input_buffer[local_index: local_index + chunks_per_step // 2], tb=tb)
                            first_step_channel[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)

                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            local_index = dst_rank_id * chunksperloop + i * chunks_per_step
                            remote_index = (src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1) * chunksperloop + i * chunks_per_step
                            tb = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                            channels[dst_rank_id, src_rank_id].put(scratch_buffer[dst_rank_id][remote_index: remote_index + chunks_per_step // 2],  input_buffer[local_index: local_index + chunks_per_step // 2], tb=tb)
                            channels[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)
                            semaphores[gpu, tb].release(tb=tb)

                for gpu in range(gpu_size):
                    src_rank_id = gpu
                    src_rank = Rank(src_rank_id)
                    input_buffer = src_rank.get_input_buffer()
                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            src_index = (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1) * chunksperloop + i * chunks_per_step + chunks_per_step // 2
                            dst_index = dst_rank_id * chunksperloop + i * chunks_per_step + chunks_per_step // 2
                            tb = tb_offset + (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1)
                            first_step_channel[dst_rank_id, src_rank_id].wait(tb=tb, data_sync=SyncType.after)
                            src_rank.copy(input_buffer[dst_index: dst_index + chunks_per_step // 2], scratch_buffer[src_rank_id][src_index: src_index + chunks_per_step // 2], tb=tb)

                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            src_index = (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1) * chunksperloop + i * chunks_per_step
                            dst_index = dst_rank_id * chunksperloop + i * chunks_per_step
                            tb = tb_offset + (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1)
                            semaphores[gpu, tb - tb_offset].acquire(tb=tb)
                            channels[dst_rank_id, src_rank_id].wait(tb=tb, data_sync=SyncType.after)
                            src_rank.copy(input_buffer[dst_index: dst_index + chunks_per_step // 2], scratch_buffer[src_rank_id][src_index: src_index + chunks_per_step // 2], tb=tb)
                    
            else:
                for gpu in range(gpu_size):
                    src_rank_id = gpu
                    src_rank = Rank(src_rank_id)
                    input_buffer = src_rank.get_input_buffer()
                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            local_index = dst_rank_id * chunksperloop + i * chunks_per_step
                            remote_index = (src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1) * chunksperloop + i * chunks_per_step
                            tb = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                            if i == steps_pipeline - 1 and last_step_process:
                                channels[dst_rank_id, src_rank_id].put(scratch_buffer[dst_rank_id][remote_index + chunks_per_step // 2: remote_index + chunks_per_step],  input_buffer[local_index + chunks_per_step // 2: local_index + chunks_per_step], tb=tb)
                                last_step_channel[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)

                                channels[dst_rank_id, src_rank_id].put(scratch_buffer[dst_rank_id][remote_index: remote_index + chunks_per_step // 2],  input_buffer[local_index: local_index + chunks_per_step // 2], tb=tb)
                                channels[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)
                                semaphores[gpu, tb].release(tb=tb)
                            else:
                                channels[dst_rank_id, src_rank_id].put(scratch_buffer[dst_rank_id][remote_index: remote_index + chunks_per_step],  input_buffer[local_index: local_index + chunks_per_step], tb=tb)
                                channels[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)
                                semaphores[gpu, tb].release(tb=tb)

                # Copy Data From Scratch Buffer
                for gpu in range(gpu_size):
                    src_rank_id = gpu
                    src_rank = Rank(src_rank_id)
                    input_buffer = src_rank.get_input_buffer()
                    for peer in range(gpu_size):
                        dst_rank_id = peer
                        if dst_rank_id != src_rank_id:
                            src_index = (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1) * chunksperloop + i * chunks_per_step
                            dst_index = dst_rank_id * chunksperloop + i * chunks_per_step
                            tb = tb_offset + (dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1)
                            if i == steps_pipeline - 1 and last_step_process:
                                last_step_channel[dst_rank_id, src_rank_id].wait(tb=tb - tb_offset, data_sync=SyncType.after)
                                src_rank.copy(input_buffer[dst_index + chunks_per_step // 2: dst_index + chunks_per_step], scratch_buffer[src_rank_id][src_index + chunks_per_step // 2: src_index + chunks_per_step], tb=tb - tb_offset)
                            
                                semaphores[gpu, tb - tb_offset].acquire(tb=tb)
                                channels[dst_rank_id, src_rank_id].wait(tb=tb, data_sync=SyncType.after)
                                src_rank.copy(input_buffer[dst_index: dst_index + chunks_per_step // 2], scratch_buffer[src_rank_id][src_index: src_index + chunks_per_step // 2], tb=tb)
                            else:
                                semaphores[gpu, tb - tb_offset].acquire(tb=tb)
                                channels[dst_rank_id, src_rank_id].wait(tb=tb, data_sync=SyncType.after)
                                src_rank.copy(input_buffer[dst_index: dst_index + chunks_per_step], scratch_buffer[src_rank_id][src_index: src_index + chunks_per_step], tb=tb)

        print(JSON())

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

alltoall_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)