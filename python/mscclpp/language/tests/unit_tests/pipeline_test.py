# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.pipeline import *


def barrier_test(num_threads_per_block, min_message_size, max_message_size):
    gpus = 2
    collective = TestCollective(gpus, 2, 0)
    with MSCCLPPProgram(
        "barrier_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):

        rank = Rank(0)
        buffer = rank.get_input_buffer()
        rank.copy(buffer[0:1], buffer[1:2], 0)
        with LoopIterationContext(unit=2**20, num_chunks=1):
            rank.copy(buffer[1:2], buffer[0:1], 0)
        rank.copy(buffer[0:1], buffer[1:2], 0)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

barrier_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
