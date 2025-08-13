# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Relaxed Signal-Wait Fuse Operation Test

This file demonstrates the fusion of relaxed signal-wait operations in MSCCLPP.
The relaxed signal-wait fuse pattern combines multiple relaxed signal-wait operations
to optimize performance by reducing overhead.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (relaxed-signal-wait-fuse) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def relaxed_signal_wait_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 2 GPUs for fused relaxed signal-wait operations
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
        # Perform fused relaxed signal-wait operations between all GPU pairs
        for src_rank in range(gpus):
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Perform fused relaxed signal operation
                    ch = MemoryChannel(dst_rank, src_rank)
                    ch.signal(tb=0, relaxed=True)
                    ch = MemoryChannel(dst_rank, src_rank)
                    ch.signal(tb=0, relaxed=True)

                    # Perform fused relaxed wait operation
                    ch = MemoryChannel(dst_rank, src_rank)
                    ch.wait(tb=0, relaxed=True)
                    ch = MemoryChannel(dst_rank, src_rank)
                    ch.wait(tb=0, relaxed=True)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

relaxed_signal_wait_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
