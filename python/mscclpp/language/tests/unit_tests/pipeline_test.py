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
        instances=2,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):

        rank = Rank(0)
        rank1 = Rank(1)
        buffer = rank.get_input_buffer()
        peer_buffer = rank1.get_input_buffer()
        ch = MemoryChannel(1, 0)
        rank.barrier([0, 1])
        with LoopIterationContext(unit=2**20, num_chunks=1):
            ch.reduce(buffer[0:1], [peer_buffer[0:1]], tb=0, local_dst_chunk=buffer[0:1])
            ch.reduce(buffer[0:1], [peer_buffer[1:2]], tb=0, local_dst_chunk=buffer[0:1])
        rank.barrier([0, 1])
        # ch.signal(0, data_sync=SyncType.both)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

barrier_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
