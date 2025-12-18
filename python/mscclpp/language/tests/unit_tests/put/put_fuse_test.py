# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Put Fuse Operation Test

This file demonstrates the use of fused put operations in MSCCLPP.
The put fuse pattern combines multiple put operations to efficiently
write data to remote GPUs with reduced synchronization overhead,
optimizing remote memory write patterns.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (put-fuse) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def put_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for fused put operations
    gpus = 2
    collective = TestCollective(gpus, 2, 0)

    with CollectiveProgram(
        "put_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Perform fused put operations between all GPU pairs
        for src_rank in range(gpus):
            rank = Rank(src_rank)
            src_buff = rank.get_input_buffer()
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    rank = Rank(dst_rank)
                    dst_buff = rank.get_input_buffer()

                    # Establish memory channel for remote memory write
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Synchronize before fused put operations
                    ch.signal(tb=0, relaxed=True)
                    ch.wait(tb=0, data_sync=SyncType.after, relaxed=True)

                    # Perform first put operation: write src_buff[0:1] to dst_buff[1:2]
                    ch.put(dst_buff[1:2], src_buff[0:1], tb=0)

                    # Second put operation with new channel (fused pattern)
                    ch = MemoryChannel(dst_rank, src_rank)
                    ch.put(dst_buff[0:1], src_buff[1:2], tb=0)

                    # Synchronize after fused operations
                    ch.signal(tb=0, data_sync=SyncType.before)
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

put_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
