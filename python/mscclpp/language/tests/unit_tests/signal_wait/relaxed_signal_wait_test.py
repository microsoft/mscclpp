# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Relaxed Signal-Wait Operation Test

This file demonstrates the use of relaxed signal and wait operations in MSCCLPP.
The relaxed signal-wait pattern provides looser synchronization between GPUs,
allowing for better performance when strict ordering is not required
in distributed GPU communications.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (relaxed-signal-wait) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def relaxed_signal_wait_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for relaxed signal-wait synchronization
    gpus = 2
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "relaxed_signal_wait_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Perform relaxed signal-wait operations between all GPU pairs
        for src_rank in range(gpus):
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Establish memory channel for relaxed synchronization
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Send relaxed signal (allows reordering for better performance)
                    ch.signal(tb=0, relaxed=True)

                    # Wait with relaxed semantics (looser synchronization guarantees)
                    ch.wait(tb=0, relaxed=True)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

relaxed_signal_wait_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
