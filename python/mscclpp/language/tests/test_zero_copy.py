# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.internal.program import MSCCLPPProgram
from mscclpp.language.internal.collectives import AllGather


def allgather_example(name, num_threads_per_block, min_message_size, max_message_size):
    # Validating parameters
    gpus = 2

    size = gpus
    chunksperloop = 1
    collective = AllGather(size, chunksperloop, True)
    with MSCCLPPProgram(
        name,
        collective,
        size,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):

        size = gpus

        scratch_buffers = []
        for rank in range(size):
            scratch_buffers.append(Buffer(rank, 2))

        for src_rank in range(size):
            rank = Rank(src_rank)
            input_buffer = rank.get_output_buffer()
            rank.copy(scratch_buffers[src_rank][0:1], input_buffer[src_rank : src_rank + 1], tb=0)

            for dst_rank in range(size):
                if src_rank != dst_rank:
                    dst_scratch_buffer = Buffer(dst_rank, 1)
                    ch = Channel(dst_rank, src_rank, ChannelType.memory)
                    ch.signal(tb=0, sync=None)
                    ch.wait(tb=0, sync="after")
                    ch.put(scratch_buffers[dst_rank][1:2], scratch_buffers[src_rank][0:1], tb=0)
                    ch.signal(tb=0, sync="before")
                    ch.wait(tb=0, sync="after")
                    rank.copy( input_buffer[dst_rank : dst_rank + 1], scratch_buffers[src_rank][1:2], tb=0)


        print(JSON())

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(
    args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size
)
