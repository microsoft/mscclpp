# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.loop import *


def alltoall_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllToAll(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        instances=16,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels and Scratch Buffer
        channels = {}
        sync_channels = {}
        semaphores = {}
        scratch_buffer = {}
        for gpu in range(gpu_size):
            src_rank_id = gpu
            scratch_buffer[src_rank_id] = Buffer(src_rank_id, gpu_size - 1)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)
                    sync_channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)
                    semaphores[src_rank_id, dst_rank_id] = Semaphore(src_rank_id, initial_value=0)

        # Initial Synchronization
        for gpus in range(gpu_size):
            src_rank_id = gpus
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != peer:
                    sync_channels[dst_rank_id, src_rank_id].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    sync_channels[dst_rank_id, src_rank_id].wait(tb=0, relaxed=True, data_sync=SyncType.after)

        # Put Data in the Remote Rank
        for gpu in range(gpu_size):
            src_rank_id = gpu
            src_rank = Rank(src_rank_id)
            input_buffer = src_rank.get_input_buffer()
            for peer in range(1, gpu_size):
                dst_rank_id = (src_rank_id + peer) % gpu_size
                if dst_rank_id != src_rank_id:
                    remote_index = src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1
                    channels[dst_rank_id, src_rank_id].put(
                        scratch_buffer[dst_rank_id][remote_index : remote_index + 1],
                        input_buffer[dst_rank_id : dst_rank_id + 1],
                        tb=0,
                    )
                    channels[dst_rank_id, src_rank_id].signal(tb=0, data_sync=SyncType.before)
                    semaphores[src_rank_id, dst_rank_id].release(tb=0)

        # Copy Data From Scratch Buffer
        for gpu in range(gpu_size):
            src_rank_id = gpu
            src_rank = Rank(src_rank_id)
            input_buffer = src_rank.get_input_buffer()
            for peer in range(1, gpu_size):
                dst_rank_id = (src_rank_id - peer) % gpu_size
                if dst_rank_id != src_rank_id:
                    index = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                    semaphores[src_rank_id, dst_rank_id].acquire(tb=1)
                    channels[dst_rank_id, src_rank_id].wait(tb=1, data_sync=SyncType.after)
                    src_rank.copy(
                        input_buffer[dst_rank_id : dst_rank_id + 1],
                        scratch_buffer[src_rank_id][index : index + 1],
                        tb=1,
                    )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

alltoall_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
