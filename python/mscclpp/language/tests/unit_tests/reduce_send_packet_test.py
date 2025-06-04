# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_send_packet_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 0, 0)
    with MSCCLPPProgram(
        "reduce_send_packet_test",
        collective,
        gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        scratch_buffers = []
        for rank in range(gpus):
            scratch_buffers.append(Buffer(rank, 4))

        for src_rank in range(gpus):
            rank = Rank(src_rank)
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    ch = Channel(dst_rank, src_rank, ChannelType.memory)
                    ch.signal(tb=0, relaxed=True)
                    ch.wait(tb=0, sync=SyncType.after, relaxed=True)
                    rank.reduce(
                        scratch_buffers[src_rank][0:1],
                        [scratch_buffers[src_rank][1:2]],
                        tb=0,
                        dst_chunk=scratch_buffers[src_rank][2:3],
                    )
                    ch.put(scratch_buffers[dst_rank][3:4], scratch_buffers[src_rank][2:3], tb=0)
                    ch.signal(tb=0, sync=SyncType.before)
                    ch.wait(tb=0, sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_send_packet_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
