# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


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

        dst_chunk = Buffer(0, 1)

        ch = SwitchChannel(rank_list=[0, 1], buffer_type=BufferType.input)
        ch.group_load_reduce(buffer_offset=0, size=1, tb=0, dst_chunk=dst_chunk[0:1])
        ch.group_store(src_chunk=dst_chunk[0:1], buffer_offset=0, size=1, tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
