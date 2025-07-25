# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Copy-Packet Operation Test

This file demonstrates the use of the copy_packet operation in MSCCLPP.
The copy_packet operation copies data from a source buffer to a destination buffer
in packet format, which ensures efficient local memory transfers while maintaining
data integrity for subsequent packet-based operations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (copy_packet) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def copy_packet_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 1 GPU
    gpus = 1
    collective = TestCollective(gpus, 1, 1)

    # Initialize MSCCLPP program context with Simple protocol
    with MSCCLPPProgram(
        "copy_packet_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        max_message_size=max_message_size,
        min_message_size=min_message_size,
    ):
        # Get rank 0 for the copy operation
        rank = Rank(0)

        # Get the input buffer from the rank
        input_buffer = rank.get_input_buffer()

        # Create a scratch buffer for the destination
        scratch_buffer = Buffer(0, 1)

        # Perform copy_packet operation:
        # - Copies data from input_buffer[0:1] to scratch_buffer[0:1]
        # - Uses threadblock 0 for the operation\
        rank.copy_packet(scratch_buffer[0:1], input_buffer[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

copy_packet_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
