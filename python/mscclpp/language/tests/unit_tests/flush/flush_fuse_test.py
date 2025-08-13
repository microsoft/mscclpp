# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Flush Fuse Operation Test

This file demonstrates the use of fused flush operations in MSCCLPP.
The flush-fuse pattern merges multiple consecutive flush operations
into a single flush for optimization, as performing more than one
flush in sequence is unnecessary.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (flush-fuse) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def flush_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 1 GPUs for fused flush operations
    gpus = 2
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "flush_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # First flush operation with post-synchronization
        ch = PortChannel(1, 0)
        ch.flush(tb=0, data_sync=SyncType.after)

        # Second flush operation with pre-synchronization (fused pattern)
        ch = PortChannel(1, 0)
        ch.flush(tb=0, data_sync=SyncType.before)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

flush_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
