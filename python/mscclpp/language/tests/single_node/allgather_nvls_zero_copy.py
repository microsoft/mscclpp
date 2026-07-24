# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allgather_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size, instances):
    # Defaults instances=8, num_threads_per_block=256 are tuned for 64-GPU (4x GB200) MNNVL NVLS:
    # they give the best busbw across 1MB-1GB (instances saturate at 8; tpb=256 beats 512/1024).
    chunksperloop = 1
    collective = AllGather(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        instances=instances,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # NVLS multicast channel over the output buffer. For Allgather each
        # rank stores its own chunk to all ranks' output buffers via the switch.
        nvls_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.output)
        channels = {}
        for gpu in range(gpu_size):
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)] = MemoryChannel(peer, gpu)

        # Synchronization to ensure all the GPUs are ready
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True, data_sync=SyncType.after)

        # Broadcasting each rank's chunk to every rank via NVLS multimem store.
        # Rank `gpu` owns output chunk `gpu` (its input under in-place AllGather) and
        # stores it to offset `gpu` across all ranks in the switch group.
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            output_buffer = rank.get_output_buffer()
            nvls_chan.at_rank(gpu).broadcast(src_chunk=output_buffer[gpu : gpu + 1], buffer_offset=gpu, size=1, tb=0)

        # Synchronization to ensure the GPUs finished
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True, data_sync=SyncType.before)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=256, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")
parser.add_argument("--instances", type=int, default=8, help="number of instances (parallel threadblocks)")

args = parser.parse_args()

allgather_example(
    args.name,
    args.num_gpus,
    args.num_threads_per_block,
    args.min_message_size,
    args.max_message_size,
    args.instances,
)
