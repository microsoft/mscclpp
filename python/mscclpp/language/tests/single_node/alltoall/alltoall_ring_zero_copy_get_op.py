# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def alltoall_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllToAll(gpu_size, chunksperloop, False)
    with MSCCLPPProgram(
        name,
        collective,
        gpu_size,
        instances=32,
        protocol="Simple",
        reuse_resources=False,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels and Scratch Buffer
        channels = {}
        for gpu in range(gpu_size):
            src_rank_id = gpu
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)

        # Initial Synchronization
        for gpus in range(gpu_size):
            src_rank_id = gpus
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != peer:
                    channels[dst_rank_id, src_rank_id].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id].wait(tb=0, data_sync=SyncType.after, relaxed=True)

        # Perform AlltoAll
        for gpu in range(gpu_size):
            src_rank_id = gpu
            src_rank = Rank(src_rank_id)
            output_buffer = src_rank.get_output_buffer()

            for peer in range(gpu_size):
                dst_rank_id = (peer + src_rank_id) % gpu_size
                input_buffer = Rank(dst_rank_id).get_input_buffer()
                local_index = dst_rank_id
                remote_index = src_rank_id
                
                if dst_rank_id != src_rank_id:
                    channels[dst_rank_id, src_rank_id].get(output_buffer[local_index: local_index + 1], input_buffer[remote_index: remote_index + 1], tb=0)
                else:
                    src_rank.copy(output_buffer[src_rank_id: src_rank_id + 1], input_buffer[src_rank_id: src_rank_id + 1], tb=0)

        # Final Synchronization
        for gpus in range(gpu_size):
            src_rank_id = gpus
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != peer:
                    channels[dst_rank_id, src_rank_id].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id].wait(tb=0, relaxed=True)

        print(JSON())

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

alltoall_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)