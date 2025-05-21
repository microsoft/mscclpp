# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.channel_based_language.program import *
from mscclpp.channel_based_language.language_interface import *
from mscclpp.channel_based_language.collectives import AllGather

def allgather_example(name, gpus, num_threads_per_block, min_message_size, max_message_size):
    # Validating parameters
    if gpus <= 0:
        raise ValueError("Number of GPUs and instances must be a positive integer")

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

        size = 2

        for src_rank in range(size):
            r = rank(src_rank)
            src_input_buffer = r.get_output_buffer()
            src_chunk = src_input_buffer[src_rank:src_rank + 1] 
            for dst_rank in range(size):
                r = rank(dst_rank)
                dst_input_buffer = r.get_output_buffer()
                dst_chunk = dst_input_buffer[src_rank:src_rank + 1] 
                if src_rank != dst_rank:
                    ch = channel(dst_rank, src_rank, ChannelType.memory)
                    ch.signal(tb=0, sync=None)
                    ch.wait(tb=0, sync="after")
                    ch.put(dst_chunk, src_chunk, tb=0)
                    ch.signal(tb=0, sync="before")
                    ch.wait(tb=0, sync="after")

        # Generate JSON
        print(JSON())

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(
    args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size
)

