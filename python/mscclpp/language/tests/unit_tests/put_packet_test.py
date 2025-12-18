# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Put-Packets Operation Test

This file demonstrates the use of the put_packets operation in MSCCLPP.
The put_packets operation writes data from a source buffer to a destination buffer
in packet format, which is useful for efficient data transfer in distributed
GPU communication patterns.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (put_packets) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def put_packets_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 1, 0)

    # Initialize MSCCLPP program context with LL (Low Latency) protocol
    with CollectiveProgram(
        "put_packets_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        for src_rank in range(gpus):
            rank = Rank(src_rank)
            src_buff = rank.get_input_buffer()
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    # Create a destination buffer on the target GPU
                    dst_buff = Buffer(dst_rank, 1)

                    # Establish a memory channel from source to destination GPU
                    ch = MemoryChannel(dst_rank, src_rank)

                    # Perform put_packets operation:
                    # - Transfers data from src_buff[0:1] to dst_buff[0:1]
                    # - Uses threadblock 0 for the operation
                    # - Data is transferred in packet format for efficient communication
                    ch.put_packets(dst_buff[0:1], src_buff[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

put_packets_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
