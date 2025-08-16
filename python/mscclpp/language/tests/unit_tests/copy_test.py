# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Copy Operation Test

This file demonstrates the use of the copy operation in MSCCLPP.
The copy operation transfers data from a source buffer to a destination buffer
using standard memory copy semantics. This is a fundamental operation for
local data movement within a GPU's memory space.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (copy) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def copy_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 1 GPU
    gpus = 1
    collective = TestCollective(gpus, 1, 1)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "copy_test",
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

        # Get input and output buffers from the rank
        input_buffer = rank.get_input_buffer()
        output_buffer = rank.get_output_buffer()

        # Perform copy operation:
        # - Copies data from input_buffer[0:1] to output_buffer[0:1]
        # - Uses threadblock 0 for the operation
        rank.copy(output_buffer[0:1], input_buffer[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

copy_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
