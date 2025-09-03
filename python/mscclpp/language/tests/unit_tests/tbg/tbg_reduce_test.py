# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce Operation Test

This file demonstrates the use of the reduce operation in MSCCLPP.
The reduce operation combines multiple data chunks using a reduction
function (such as sum or max), enabling efficient data aggregation
within a single GPU's memory space.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up single GPU for local reduce operation
    gpus = 1
    collective = TestCollective(gpus, 2, 1)

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

        # Perform local reduce: combine input_buffer[0:1] and input_buffer[1:2] into output_buffer[0:1]
        rank.reduce(input_buffer[0:1], [input_buffer[1:2]], tb_group=tbg, dst_chunk=output_buffer[0:1])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
