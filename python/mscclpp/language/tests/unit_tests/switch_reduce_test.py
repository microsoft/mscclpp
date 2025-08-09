# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Switch Reduce Operation Test

This file demonstrates the use of the switch reduce operation in MSCCLPP.
The switch reduce operation aggregates data from multiple ranks into a single rank
using a switch channel, which is useful for efficient many-to-one reduction
communication patterns in distributed GPU computations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (switch reduce) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def switch_reduce_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 3 GPUs
    gpus = 3
    collective = TestCollective(gpus, 1, 1)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "switch_reduce_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create a destination buffer for the reduce operation result
        dst_chunk = Buffer(0, 1)

        # Create a switch channel connecting ranks 0 and 1 with input buffer type
        ch = SwitchChannel(rank_list=[0, 1], buffer_type=BufferType.input)

        # Perform reduce operation at rank 0:
        # - Aggregates data from all connected ranks (many-to-one reduction)
        # - Uses buffer_offset=0, size=1, and threadblock 0
        # - Result is stored in dst_chunk[0:1]
        # - Switch channel enables efficient data aggregation from multiple sources
        ch.at_rank(0).reduce(buffer_offset=0, size=1, tb=0, dst_chunk=dst_chunk[0:1])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

switch_reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
