# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unpack-Copy-Packet Operation Test

This file demonstrates the use of the unpack_copy_packet operation in MSCCL++.
The unpack-copy-packet pattern converts data from packet format back to the
standard format and then copies it to the target buffer.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (unpack-copy-packet) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def unpack_copy_packet_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up single GPU for unpack-copy-packet operations
    gpus = 1
    collective = TestCollective(gpus, 1, 1)

    with MSCCLPPProgram(
        "unpack_copy_packet_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        max_message_size=max_message_size,
        min_message_size=min_message_size,
    ):
        rank = Rank(0)
        input_buffer = rank.get_input_buffer()
        output_buffer = rank.get_output_buffer()
        scratch_buffer = Buffer(0, 1)

        # Step 1: Copy data from input to scratch buffer in packet format
        rank.copy_packet(scratch_buffer[0:1], input_buffer[0:1], tb=0)

        # Step 2: Unpack packet data from scratch buffer to output buffer
        rank.unpack_copy_packet(output_buffer[0:1], scratch_buffer[0:1], tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

unpack_copy_packet_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
