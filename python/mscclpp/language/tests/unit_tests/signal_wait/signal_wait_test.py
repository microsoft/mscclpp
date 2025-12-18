# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Signal-Wait Operation Test

This file demonstrates the use of signal and wait operations in MSCCLPP.
The signal-wait pattern provides synchronization between GPUs using
memory channels, enabling coordinated execution and data consistency
in distributed GPU communications.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (signal-wait) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def signal_wait_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for signal-wait synchronization
    gpus = 2
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "signal_wait_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Perform signal-wait synchronization between all GPU pairs
        for src_rank in range(gpus):
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Establish memory channel for synchronization
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Send signal before data operations
                    ch.signal(tb=0, data_sync=SyncType.before)

                    # Wait for signal completion after data operations
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

signal_wait_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
