# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Switch Channel + Thread Block Group Test

This file demonstrates the use of the switch reduce and broadcast operations
together with a ThreadBlockGroup in MSCCLPP. The same collective operation is
executed cooperatively by a group of thread blocks (tbg) instead of a single
thread block.

WARNING: This algorithm is designed solely for demonstrating the use of the
switch reduce/broadcast operations with a ThreadBlockGroup and is NOT intended
for production use. This test may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def tbg_switch_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 4 GPUs
    gpus = 4
    collective = TestCollective(gpus, 1, 1)

    with CollectiveProgram(
        "tbg_switch_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Destination buffer for the reduce result on rank 0
        dst_chunk = Buffer(0, 1)
        # Source buffer for the broadcast on rank 0
        src_chunk = Buffer(0, 1)

        # Create a switch channel connecting all 4 ranks with input buffer type
        ch = SwitchChannel(rank_list=[0, 1, 2, 3], buffer_type=BufferType.input)

        # Declare a Thread Block Group of 4 thread blocks
        tbg = ThreadBlockGroup(tb_list=[0, 1, 2, 3])

        # Switch reduce at rank 0 executed cooperatively by the whole thread block group
        ch.at_rank(0).reduce(buffer_offset=0, size=1, dst_chunk=dst_chunk[0:1], tb_group=tbg)

        # Switch broadcast from rank 0 executed cooperatively by the whole thread block group
        ch.at_rank(0).broadcast(src_chunk=src_chunk[0:1], buffer_offset=0, size=1, tb_group=tbg)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

tbg_switch_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
