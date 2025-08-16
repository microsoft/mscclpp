# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Switch Broadcast Operation Test

This file demonstrates the use of the switch broadcast operation in MSCCLPP.
The switch broadcast operation sends data from one rank to multiple ranks
using a switch channel, which is useful for efficient one-to-many
communication patterns in distributed GPU computations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (switch broadcast) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def switch_broadcast_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 3 GPUs
    gpus = 3
    collective = TestCollective(gpus, 1, 1)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "group_store_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create a destination buffer for the broadcast operation
        dst_chunk = Buffer(0, 1)

        # Create a switch channel connecting ranks 0 and 1 with input buffer type
        ch = SwitchChannel(rank_list=[0, 1], buffer_type=BufferType.input)

        # Perform broadcast operation from rank 0:
        # - Broadcasts data from dst_chunk[0:1] to all connected ranks
        # - Uses buffer_offset=0, size=1, and threadblock 0
        # - Switch channel enables efficient one-to-many communication
        ch.at_rank(0).broadcast(src_chunk=dst_chunk[0:1], buffer_offset=0, size=1, tb=0)

        # Output the generated program in JSON format
        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

switch_broadcast_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
