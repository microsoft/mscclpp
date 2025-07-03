# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def reduce_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 1
    collective = TestCollective(gpus, 2, 1)
    with MSCCLPPProgram(
        "reduce_test",
        collective,
        gpus,
        instances=1,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        max_message_size=max_message_size,
        min_message_size=min_message_size,
    ):
        rank = Rank(0)
        input_buffer = rank.get_input_buffer()
        output_buffer = rank.get_output_buffer()

        rank.reduce(input_buffer[0:1], [input_buffer[1:2]], tb=0, dst_chunk=output_buffer[0:1])

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

reduce_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
