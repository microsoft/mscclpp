# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllReduce(gpu_size, chunksperloop, True)
    with MSCCLPPProgram(
        name,
        collective,
        gpu_size,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        for gpu in range(gpu_size):
            src_rank = gpu
            rank = Rank(src_rank)
            input_buffer = rank.get_input_buffer()
            scratch_buffer = Buffer(src_rank, gpu_size)

            dst_chunk = scratch_buffer[0:gpu_size]
            src_chunk = input_buffer[0:gpu_size]
            rank.copy(dst_chunk, src_chunk, tb=0)

            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    chan = Channel(dst_rank, src_rank)
                    chan.signal(tb=0, sync="before")
                    chan.wait(tb=0, sync="after")

        # do allreduce in scratch buffer
        buffer_offset = src_rank
        nvls_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.scratch)
        nvls_chan.group_load_reduce(buffer_offset, 1, input_buffer[gpu : gpu + 1], 0)
        nvls_chan.group_store(input_buffer[gpu : gpu + 1], buffer_offset, 1, tb=0)

        for gpu in range(gpu_size):
            src_rank = gpu
            rank = Rank(src_rank)
            input_buffer = rank.get_input_buffer()
            scratch_buffer = Buffer(src_rank, gpu_size)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    chan = Channel(dst_rank, src_rank)
                    chan.signal(tb=0, sync="before")
                    chan.wait(tb=0, sync="after")

            dst_chunk = input_buffer[0:gpu_size]
            src_chunk = scratch_buffer[0:gpu_size]
            rank.copy(dst_chunk, src_chunk, tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
