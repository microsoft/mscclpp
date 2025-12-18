# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Put-With-Signal-And-Flush Operation Test

This file demonstrates the use of put-with-signal-and-flush operations in MSCCLPP.
The put-with-signal-and-flush operation combines data transfer, signaling, and
flushing to ensure data consistencys.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (put-with-signal-and-flush) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def put_with_signal_and_flush_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for put-with-signal-and-flush operations
    gpus = 2
    collective = TestCollective(gpus, 2, 0)

    with CollectiveProgram(
        "put_with_signal_and_flush_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Iterate through all GPU pairs to perform put-with-signal-and-flush operations
        for src_rank in range(gpus):
            # Get the source rank and its input buffer
            rank = Rank(src_rank)
            src_buff = rank.get_input_buffer()
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Get the destination rank and its input buffer
                    rank = Rank(dst_rank)
                    dst_buff = rank.get_input_buffer()

                    # Establish port channel for put-with-signal-and-flush communication
                    ch = PortChannel(dst_rank, src_rank)

                    # Initial synchronization: send signal and wait for completion
                    ch.signal(tb=0)
                    ch.wait(tb=0, data_sync=SyncType.after)

                    # Perform put_with_signal_and_flush operation
                    ch.put_with_signal_and_flush(dst_buff[1:2], src_buff[0:1], tb=0)

                    # Wait for the put-with-signal-and-flush operation to complete
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

put_with_signal_and_flush_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
