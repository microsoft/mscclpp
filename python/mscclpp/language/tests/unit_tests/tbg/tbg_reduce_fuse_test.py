# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce Fuse Operation Test

This file demonstrates the use of fused reduce operations in MSCCLPP.
The reduce fuse pattern combines multiple reduce operations to efficiently
aggregate data chunks with reduced overhead, optimizing local data
reduction patterns within a GPU's memory space.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce-fuse) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up single GPU for fused reduce operations
    gpus = 1
    collective = TestCollective(gpus, 3, 2)

    with CollectiveProgram(
        "reduce_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        max_message_size=max_message_size,
        min_message_size=min_message_size,
    ):
        rank = Rank(0)
        input_buffer = rank.get_input_buffer()
        output_buffer = rank.get_output_buffer()

        # Declare Thread Block Group
        tbg = ThreadBlockGroup(tb_list=[0, 1, 2, 3])

        # Perform fused reduce operations: multiple reductions with reduced overhead
        # First reduce: combine input_buffer[0:1] and [1:2] into output_buffer[0:1]
        rank.reduce(input_buffer[0:1], [input_buffer[1:2]], tb_group=tbg, dst_chunk=output_buffer[0:1])
        # Second reduce: combine input_buffer[0:1] and [2:3] into output_buffer[0:1]
        rank.reduce(input_buffer[0:1], [input_buffer[2:3]], tb_group=tbg, dst_chunk=output_buffer[0:1])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
