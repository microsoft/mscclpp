# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Barrier Operation Test

This file demonstrates the use of the barrier operation in MSCCLPP.
The barrier operation synchronizes execution across multiple thread blocks, ensuring
that all participating thread blocks reach the barrier point before any of them
can proceed. This is essential for coordinating distributed computations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (barrier) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def barrier_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 1 GPU
    gpus = 1
    collective = TestCollective(gpus, 0, 0)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "barrier_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Get rank 0 for barrier synchronization
        rank = Rank(0)

        # Perform barrier operation to synchronize thread blocks 0 and 1
        # This ensures both thread blocks reach this point before proceeding
        rank.barrier(tb_list=[0, 1])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

barrier_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
