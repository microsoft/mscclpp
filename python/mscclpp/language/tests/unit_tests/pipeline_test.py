# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Pipeline Copy Operation Test

The pipeline test performs a simple copy operation from an input buffer to an output buffer
within a LoopIterationContext, which executes multiple iterations and processes 1MB (2^20 bytes)
chunks of data in each iteration. This showcases how MSCCLPP handles chunked data movement
operations across pipeline iterations in the language framework.

WARNING: This algorithm is designed solely for demonstrating a single operation
(copy) and is NOT intended for production use. This test may not work correctly
in the MSCCLPP executor as it is meant for educational and demonstration purposes only.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.loop import *


def pipeline_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 1 GPU for the copy operation
    gpus = 1
    collective = TestCollective(gpus, 1, 1)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "barrier_test",
        collective,
        gpus,
        protocol="Simple",
        instances=1,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):

        # Get rank 0 for the copy operation (single GPU setup)
        rank = Rank(0)

        # Get input and output buffers from the rank for data movement
        input_buffer = rank.get_input_buffer()
        output_buffer = rank.get_output_buffer()

        # Define the pipeline loop context with specific data chunking parameters
        with LoopIterationContext(unit=2**20, num_chunks=1):
            # Copy data from input_buffer[0:1] to output_buffer[0:1] using threadblock 0
            rank.copy(output_buffer[0:1], input_buffer[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

pipeline_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
