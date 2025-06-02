# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def relaxed_signal_wait_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 0, 0)
    with MSCCLPPProgram(
        "relaxed_signal_wait_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        for src_rank in range(gpus):
            for dst_rank in range(gpus):
                if src_rank != dst_rank:
                    ch = Channel(dst_rank, src_rank, ChannelType.memory)
                    ch.relaxed_signal(tb=0)
                    ch.relaxed_wait(tb=0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

relaxed_signal_wait_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
