# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def group_load_reduce_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 1, 0)
    with MSCCLPPProgram(
        "group_load_reduce_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        dst_chunk = Buffer(0, 1)

        ch = SwitchChannel(rank_list=[0, 1], buffer_type=BufferType.input)
        ch.group_load_reduce(
            buffer_offset=0,
            size=1,
            tb=0,
            dst_chunk=dst_chunk[0:1]
        )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

group_load_reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
