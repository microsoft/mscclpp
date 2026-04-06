# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from enum import Enum

def allreduce_example(name, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    gpu_size = 2
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Declaring Ranks, Channels, Buffers for 2 GPU allgather example
        first_rank = Rank(0)
        second_rank = Rank(1)
        first_ch = MemoryChannel(1, 0)
        second_ch = MemoryChannel(0, 1)
        first_input_buffer = first_rank.get_input_buffer()
        second_input_buffer = second_rank.get_input_buffer()
        first_scratch_buffer = Buffer(0, 3)
        second_scratch_buffer = Buffer(1, 3)

        # First rank puts packets in the remote scratch buffer of the second rank
        first_ch.put_packets(second_scratch_buffer[1: 2], first_input_buffer[1 : 2], tb=0)
        second_ch.put_packets(first_scratch_buffer[0 : 1], second_input_buffer[0 : 1], tb=0)

        # Second rank copy packets to scratch buffer and then read put packets to first rank output buffer
        first_rank.reduce(first_input_buffer[0 : 1], [first_scratch_buffer[0 : 1]], tb=1, packet=True)
        first_ch.put_packets(second_scratch_buffer[0 : 1], first_input_buffer[0 : 1], tb=1)

        # First rank copy packets to scratch buffer and then read put packets to second rank output buffer
        second_rank.reduce(second_input_buffer[1 : 2], [second_scratch_buffer[1 : 2]],  tb=1, packet=True)
        second_rank.copy_packets(second_scratch_buffer[2: 3], second_input_buffer[1 : 2], tb=1)
        second_ch.read_put_packets(first_scratch_buffer[1 : 2], second_scratch_buffer[2: 3], tb=1)

        # First rank copy packets to scratch buffer and then read put packets to second rank output buffer
        first_rank.unpack_packets(first_input_buffer[1 : 2], first_scratch_buffer[1 : 2], tb=2)
        second_rank.unpack_packets(second_input_buffer[0 : 1], second_scratch_buffer[0 : 1], tb=2)


        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
