# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce-Copy-Send-Packets Operation Test

This file demonstrates the use of reduce, copy, and send operations in MSCCLPP with
packet format. The reduce-copy-send-packets pattern combines local reductions,
packet-based data copying, and remote data transfers, ensuring data integrity
during distributed GPU communication through the packet format.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce-copy-send-packets) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_copy_send_packets_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for reduce-copy-send-packets operations
    gpus = 2
    collective = TestCollective(gpus, 1, 0)

    with CollectiveProgram(
        "reduce_copy_send_packets_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        rank = Rank(0)
        input_buffer = rank.get_input_buffer()

        # Create scratch buffers for both GPUs
        scratch_buffers = [Buffer(0, 2), Buffer(1, 1)]

        # Establish memory channel for communication between GPUs
        ch = MemoryChannel(1, 0)

        # Perform packet-based reduce: combine input into local scratch buffer
        rank.reduce(input_buffer[0:1], [scratch_buffers[0][0:1]], tb=0, packet=True)

        # Copy reduced result to another buffer location using packet format
        rank.copy_packets(scratch_buffers[0][1:2], input_buffer[0:1], tb=0)

        # Send packet data to remote GPU
        ch.put_packets(scratch_buffers[1][0:1], input_buffer[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_copy_send_packets_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
