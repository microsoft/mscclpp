# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allgather_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllGather(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Scratch Buffers
        scratch_buffer = []
        for gpu in range(gpu_size):
            scratch_buffer.append(Buffer(gpu, 2 * gpu_size))

        # Copying it to scratch buffer
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            scratch_offset = gpu_size
            input_buffer = rank.get_input_buffer()
            rank.copy_packets(
                scratch_buffer[gpu][scratch_offset + gpu : scratch_offset + gpu + 1], input_buffer[0:1], tb=0
            )

        # Putting packets in the remote scratch buffer
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            output_buffer = rank.get_output_buffer()
            for peer in range(1, gpu_size):
                dst_rank = (gpu + peer) % gpu_size
                ch = MemoryChannel(dst_rank, gpu)
                tb = 0
                ch.read_put_packets(
                    scratch_buffer[dst_rank][gpu : gpu + 1],
                    scratch_buffer[gpu][scratch_offset + gpu : scratch_offset + gpu + 1],
                    tb,
                )

        # Copying packets from local scratch buffer to local buffer
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            output_buffer = rank.get_output_buffer()
            for peer in range(1, gpu_size):
                dst_rank = (gpu + peer) % gpu_size
                rank.unpack_packets(
                    output_buffer[dst_rank : dst_rank + 1],
                    scratch_buffer[gpu][dst_rank : dst_rank + 1],
                    tb=0,
                )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
