# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Transfer Pack Test

This file tests the UNPACK_PACKETS, COPY_PACKETS, READ_PUT_PACKETS and
PUT_PACKETS operations. It implements a 2-GPU allgather with the LL
(low-latency) packet protocol.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def transfer_pack(name, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    gpu_size = 2
    collective = AllGather(gpu_size, chunksperloop, True)
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
        # Setup ranks, channels, output and scratch buffers for 2-GPU allgather
        first_rank = Rank(0)
        second_rank = Rank(1)
        first_ch = MemoryChannel(1, 0)
        second_ch = MemoryChannel(0, 1)
        first_output_buffer = first_rank.get_output_buffer()
        second_output_buffer = second_rank.get_output_buffer()
        first_scratch_buffer = Buffer(0, 2)
        second_scratch_buffer = Buffer(1, 2)

        # Rank 0 sends its output chunk as packets to rank 1's scratch buffer
        first_ch.put_packets(second_scratch_buffer[0:1], first_output_buffer[0:1], tb=0)

        # Rank 1 copies its output to scratch, then sends it as packets to rank 0's scratch buffer
        second_rank.copy_packets(second_scratch_buffer[1:2], second_output_buffer[1:2], tb=0)
        second_ch.read_put_packets(first_scratch_buffer[1:2], second_scratch_buffer[1:2], tb=1)

        # Both ranks unpack received packets from scratch into their output buffers
        first_rank.unpack_packets(first_output_buffer[1:2], first_scratch_buffer[1:2], tb=1)
        second_rank.unpack_packets(second_output_buffer[0:1], second_scratch_buffer[0:1], tb=2)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

transfer_pack(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
