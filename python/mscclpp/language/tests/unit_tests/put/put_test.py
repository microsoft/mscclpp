# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def put_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 2, 0)
    with MSCCLPPProgram(
        "put_test",
        collective,
        gpus,
        protocol="Simple",
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
                    rank = Rank(dst_rank)
                    dst_buff = rank.get_input_buffer()
                    output_buff = rank.get_output_buffer()
                    ch = Channel(dst_rank, src_rank)
                    ch.signal(tb=0, relaxed=True)
                    ch.wait(tb=0, sync=SyncType.after, relaxed=True)
                    ch.put(dst_buff[1:2], src_buff[0:1], tb=0)
                    ch.signal(tb=0, sync=SyncType.before)
                    ch.wait(tb=0, sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

put_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
