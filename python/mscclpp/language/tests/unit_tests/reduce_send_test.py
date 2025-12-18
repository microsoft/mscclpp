# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reduce-Send Operation Test

This file demonstrates the use of reduce and send operations in MSCCLPP.
The reduce-send pattern combines local reduction with remote data transfer,
using memory channels with synchronization to coordinate distributed
GPU computations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (reduce-send) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_send_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for reduce-send operations
    gpus = 2
    collective = TestCollective(gpus, 2, 2)

    with CollectiveProgram(
        "reduce_send_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Iterate through all source ranks
        for src_rank in range(gpus):
            rank = Rank(src_rank)
            input_buff = rank.get_input_buffer()
            output_buff = rank.get_output_buffer()
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    peer_rank = Rank(dst_rank)
                    peer_output_buff = peer_rank.get_output_buffer()

                    # Establish memory channel for communication
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Synchronize before operation
                    ch.signal(tb=0, relaxed=True)
                    ch.wait(tb=0, data_sync=SyncType.after, relaxed=True)

                    # Perform local reduce: combine input_buff[0:1] and input_buff[1:2]
                    rank.reduce(input_buff[0:1], [input_buff[1:2]], tb=0, dst_chunk=output_buff[0:1])

                    # Send reduced result to peer GPU
                    ch.put(peer_output_buff[1:2], output_buff[0:1], tb=0)

                    # Synchronize after operation
                    ch.signal(tb=0, data_sync=SyncType.before)
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_send_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
