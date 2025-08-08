# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce Packets Fuse Operation Test

This file demonstrates the use of fused packet-based reduce operations in MSCCLPP.
The reduce packets fuse pattern combines multiple packet-based reduce operations
to efficiently aggregate data with packet format integrity while reducing
synchronization overhead.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce-packets-fuse) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_packets_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up single GPU for fused packet-based reduce operations
    gpus = 1
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "reduce_packets_test",
        collective,
        gpus,
        protocol="LL",  # Low Latency protocol required for packet operations
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        max_message_size=max_message_size,
        min_message_size=min_message_size,
    ):
        rank = Rank(0)
        scratch_buffer = Buffer(0, 4)

        # Perform fused packet-based reduce operations with data integrity
        # First reduce: combine scratch_buffer[0:1] and [1:2] into [3:4] with packet format
        rank.reduce(scratch_buffer[0:1], [scratch_buffer[1:2]], tb=0, packet=True, dst_chunk=scratch_buffer[3:4])
        # Second reduce: combine scratch_buffer[0:1] and [2:3] into [3:4] with packet format
        rank.reduce(scratch_buffer[0:1], [scratch_buffer[2:3]], tb=0, packet=True, dst_chunk=scratch_buffer[3:4])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_packets_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
