# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce-Send-Packets Operation Test

This file demonstrates the use of reduce and send operations in MSCCL++ with
a focus on the packet format. The reduce-send-packets pattern combines local
reductions (where some chunks are already in packet format) with packet-based
remote data transfers, ensuring reliable communication and data integrity
between distributed GPUs through the packet format.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce-send-packets) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_send_packets_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for reduce-send-packets operations
    gpus = 2
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "reduce_send_packets_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create scratch buffers for each GPU (4 chunks per buffer)
        scratch_buffers = []
        for rank in range(gpus):
            scratch_buffers.append(Buffer(rank, 4))

        # Perform reduce-send-packets operations between all GPU pairs
        for src_rank in range(gpus):
            rank = Rank(src_rank)
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Establish memory channel for packet communication
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Perform packet-based reduce: combine input chunks [0:1] and [1:2] into scratch [2:3]
                    rank.reduce(
                        scratch_buffers[src_rank][0:1],
                        [scratch_buffers[src_rank][1:2]],
                        tb=0,
                        dst_chunk=scratch_buffers[src_rank][2:3],
                        packet=True,
                    )

                    # Send reduced result to destination GPU using packet format
                    ch.put_packets(scratch_buffers[dst_rank][3:4], scratch_buffers[src_rank][2:3], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_send_packets_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
