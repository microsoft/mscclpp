# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Switch Broadcast Operation Test

This file demonstrates the use of the switch broadcast operation in MSCCLPP.
The switch broadcast operation sends data from one rank to multiple ranks
using a switch channel, which is useful for efficient one-to-many
communication patterns in distributed GPU computations.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (switch broadcast) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def switch_broadcast_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up a test environment with 3 GPUs
    gpus = 2
    collective = AllGather(gpus, 1, True)

    # Initialize MSCCLPP program context with Simple protocol
    with CollectiveProgram(
        "group_store_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        scratch_buffers = [Buffer(i, gpus) for i in range(gpus)]
        ch = SwitchChannel(rank_list=[i for i in range(gpus)], buffer_type=BufferType.scratch)

        for gpu_id in range(gpus):
            rank = Rank(gpu_id)
            output_buffer = rank.get_output_buffer()
            rank.copy_packets(
                dst_chunk=scratch_buffers[gpu_id][gpu_id: gpu_id + 1],
                src_chunk=output_buffer[gpu_id: gpu_id + 1],
                tb=0,
            )
            ch.at_rank(gpu_id).broadcast(src_chunk=scratch_buffers[gpu_id][gpu_id: gpu_id + 1], buffer_offset=gpu_id, size=1, tb=0)
            for peer in range(gpus):
                if peer != gpu_id:
                    rank.unpack_packets(
                        dst_chunk=output_buffer[peer: peer + 1],
                        src_chunk=scratch_buffers[gpu_id][peer: peer + 1],
                        tb=0,
                    )

        # Output the generated program in JSON format
        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

switch_broadcast_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
