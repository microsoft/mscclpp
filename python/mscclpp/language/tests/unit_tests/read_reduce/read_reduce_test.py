# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Read-Reduce Operation Test

This file demonstrates the use of read-reduce operations in MSCCLPP.
The read-reduce pattern combines remote memory read with local reduction.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (read-reduce) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def read_reduce_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for read-reduce operations
    gpus = 2
    collective = TestCollective(gpus, 2, 2)

    with CollectiveProgram(
        "read_reduce_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Iterate through all GPU pairs to perform read-reduce operations
        for src_rank in range(gpus):
            # Get the current rank and its input/output buffers
            rank = Rank(src_rank)
            input_buff = rank.get_input_buffer()
            output_buff = rank.get_output_buffer()
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Get the peer rank and its input buffer for remote read
                    peer_rank = Rank(dst_rank)
                    peer_input_buff = peer_rank.get_input_buffer()

                    # Establish memory channel for read-reduce communication
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Initial synchronization with relaxed semantics for better performance
                    ch.signal(tb=0, relaxed=True)
                    ch.wait(tb=0, data_sync=SyncType.after, relaxed=True)

                    # Perform read-reduce operation
                    ch.reduce(input_buff[0:1], [peer_input_buff[1:2]], tb=0, local_dst_chunk=output_buff[0:1])

                    # Final synchronization to ensure the operation completes
                    ch.signal(tb=0, data_sync=SyncType.before)
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

read_reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
