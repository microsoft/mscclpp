# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce Pack Thread Block Group Test

This file tests the REDUCE_COPY_SEND_PACKETS and REDUCE_SEND_PACKETS
operations using thread block groups. It implements a 2-GPU allreduce
with the LL (low-latency) packet protocol, where multiple thread
blocks cooperate on each phase.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_pack_tbg(name, num_threads_per_block, min_message_size, max_message_size):
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
        # Setup ranks, channels, input and scratch buffers for 2-GPU allreduce
        first_rank = Rank(0)
        second_rank = Rank(1)
        first_ch = MemoryChannel(1, 0)
        second_ch = MemoryChannel(0, 1)
        first_input_buffer = first_rank.get_input_buffer()
        second_input_buffer = second_rank.get_input_buffer()
        first_scratch_buffer = Buffer(0, 3)
        second_scratch_buffer = Buffer(1, 3)
        tbg = []
        for i in range(3):
            tbg.append(ThreadBlockGroup(tb_list=[2 * i, 2 * i + 1]))

        # Each rank sends its input chunk as packets to the other rank's scratch buffer
        first_ch.put_packets(second_scratch_buffer[1:2], first_input_buffer[1:2], tb_group=tbg[0])
        second_ch.put_packets(first_scratch_buffer[0:1], second_input_buffer[0:1], tb_group=tbg[0])

        # Rank 0 reduces received scratch with its input, then sends the result to rank 1's scratch
        first_rank.reduce(first_input_buffer[0:1], [first_scratch_buffer[0:1]], tb_group=tbg[1], packet=True)
        first_ch.put_packets(second_scratch_buffer[0:1], first_input_buffer[0:1], tb_group=tbg[1])

        # Rank 1 reduces received scratch with its input, then sends the result back to rank 0's scratch
        second_rank.reduce(second_input_buffer[1:2], [second_scratch_buffer[1:2]], tb_group=tbg[1], packet=True)
        second_rank.copy_packets(second_scratch_buffer[2:3], second_input_buffer[1:2], tb_group=tbg[1])
        second_ch.read_put_packets(first_scratch_buffer[1:2], second_scratch_buffer[2:3], tb_group=tbg[1])

        # Both ranks unpack the final reduced packets from scratch into their output buffers
        first_rank.unpack_packets(first_input_buffer[1:2], first_scratch_buffer[1:2], tb_group=tbg[2])
        second_rank.unpack_packets(second_input_buffer[0:1], second_scratch_buffer[0:1], tb_group=tbg[2])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_pack_tbg(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
