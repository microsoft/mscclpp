# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce Thread Block Group Test

This file tests the PUT, GET, COPY, REDUCE_SEND and READ_REDUCE_SEND
operations using thread block groups. It implements a 2-GPU allreduce
with the Simple protocol and instruction fusion, where multiple thread
blocks cooperate on each operation.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_tbg(name, num_threads_per_block, min_message_size, max_message_size):
    collective = AllReduce(2, 2, True)
    with CollectiveProgram(
        name,
        collective,
        2,
        protocol="Simple",
        instr_fusion=True,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Setup ranks, memory channels, input buffers, and scratch buffers for 2-GPU AllReduce
        first_rank = Rank(0)
        second_rank = Rank(1)
        first_ch_tb0 = MemoryChannel(1, 0)
        first_ch_tb1 = MemoryChannel(1, 0)
        second_ch_tb0 = MemoryChannel(0, 1)
        second_ch_tb1 = MemoryChannel(0, 1)
        first_input_buffer = first_rank.get_input_buffer()
        second_input_buffer = second_rank.get_input_buffer()
        first_scratch_buffer = Buffer(0, 4)
        second_scratch_buffer = Buffer(1, 4)
        tbg = ThreadBlockGroup(tb_list=[0, 1])

        # Each rank copies its input chunks to scratch to prepare for remote access
        first_rank.copy(first_scratch_buffer[2:4], first_input_buffer[2:4], tb_group=tbg)
        second_rank.copy(second_scratch_buffer[0:2], second_input_buffer[0:2], tb_group=tbg)

        # Signal and wait on both TBs to ensure scratch data is visible to the remote rank
        first_ch_tb0.signal(tb=0)
        first_ch_tb1.signal(tb=1)
        second_ch_tb0.signal(tb=0)
        second_ch_tb1.signal(tb=1)

        first_ch_tb0.wait(tb=0)
        first_ch_tb1.wait(tb=1)
        second_ch_tb0.wait(tb=0)
        second_ch_tb1.wait(tb=1)

        # Rank 0 reduces chunk 0 from rank 1's scratch and writes result to both ranks
        first_ch_tb0.reduce(first_input_buffer[0:1], [second_scratch_buffer[0:1]], tb_group=tbg)
        first_ch_tb0.put(second_input_buffer[0:1], first_input_buffer[0:1], tb_group=tbg)

        # Rank 0 fetches chunk 1 from rank 1's scratch, reduces locally, and writes result to both ranks
        first_ch_tb0.get(first_scratch_buffer[1:2], second_scratch_buffer[1:2], tb_group=tbg)
        first_rank.reduce(first_input_buffer[1:2], [first_scratch_buffer[1:2]], tb_group=tbg)
        first_ch_tb0.put(second_input_buffer[1:2], first_input_buffer[1:2], tb_group=tbg)

        # Rank 1 reduces chunks 2-3 from rank 0's input, copies to scratch, and writes result to both ranks
        second_ch_tb0.reduce(second_input_buffer[2:4], [first_input_buffer[2:4]], tb_group=tbg)
        second_rank.copy(second_scratch_buffer[2:4], second_input_buffer[2:4], tb_group=tbg)
        second_ch_tb0.put(first_input_buffer[2:4], second_scratch_buffer[2:4], tb_group=tbg)

        # Final signal/wait on both TBs to ensure all reduced data is consistent across both ranks
        first_ch_tb0.signal(tb=0)
        first_ch_tb1.signal(tb=1)
        second_ch_tb0.signal(tb=0)
        second_ch_tb1.signal(tb=1)

        first_ch_tb0.wait(tb=0)
        first_ch_tb1.wait(tb=1)
        second_ch_tb0.wait(tb=0)
        second_ch_tb1.wait(tb=1)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_tbg(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
