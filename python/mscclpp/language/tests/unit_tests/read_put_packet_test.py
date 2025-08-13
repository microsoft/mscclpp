# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Read-Put-Packets Operation Test

This file demonstrates the use of the read_put_packets operation in MSCCLPP.
The read_put_packets operation combines a local read with a remote write in a single
operation. It reads data from the source in packet format to ensure the data
is ready, then transfers it in packet format to the destination, which is useful for
certain communication patterns.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (read_put_packets) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def read_put_packets_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 0, 0)

    # Initialize MSCCLPP program context with LL (Low Latency) protocol
    with CollectiveProgram(
        "read_put_packets_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create scratch buffers for each GPU rank
        scratch_buffers = []
        for rank in range(gpus):
            scratch_buffers.append(Buffer(rank, 2))

        # Perform read_put_packets operations
        for src_rank in range(gpus):
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Perform read_put_packets operation:
                    # - Reads from src_rank's buffer[0:1]
                    # - Writes to dst_rank's buffer[1:2]
                    # - Uses threadblock 0 for the operation
                    # Note: Both source and destination chunks must use scratch buffers
                    # because the data is in LL (Low Latency) format
                    ch.read_put_packets(scratch_buffers[dst_rank][1:2], scratch_buffers[src_rank][0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

read_put_packets_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
