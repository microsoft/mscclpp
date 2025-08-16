# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = gpu_size
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="LL",
        instr_fusion=True,
        auto_sync=False,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Scratch Buffers
        scratch_buffer = []
        for gpu in range(gpu_size):
            scratch_buffer.append(Buffer(gpu, 2 * (gpu_size - 1) * gpu_size))

        # Creating Channels
        channels = {}
        for gpu in range(gpu_size):
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)] = MemoryChannel(peer, gpu)

        # Each rank sends the nth chunk to the nth rank into scratch space
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for peer in range(gpu_size):
                if peer != gpu:
                    scratch_index = gpu if gpu < peer else gpu - 1
                    scratch_index *= gpu_size
                    input_index = peer * gpu_size
                    tb = peer if peer < gpu else peer - 1
                    channels[(peer, gpu)].put_packets(
                        scratch_buffer[peer][scratch_index : scratch_index + gpu_size],
                        input_buffer[input_index : input_index + gpu_size],
                        tb,
                    )

        # Each rank performs a local reduction on the nth chunk
        for gpu in range(gpu_size):
            for index in range(gpu_size):
                chunks = []
                for peer in range(gpu_size):
                    if peer != gpu:
                        scratch_index = peer if peer < gpu else peer - 1
                        scratch_index = scratch_index * gpu_size + index
                        chunks.append(scratch_buffer[gpu][scratch_index : scratch_index + 1])
                rank = Rank(gpu)
                input_buffer = rank.get_input_buffer()
                input_index = gpu * gpu_size + index
                rank.reduce(input_buffer[input_index : input_index + 1], chunks, index, packet=True)

                for peer in range(gpu_size):
                    if peer != gpu:
                        scratch_index = gpu if gpu < peer else gpu - 1
                        scratch_index = gpu_size * (gpu_size - 1) + scratch_index * gpu_size + index
                        channels[(peer, gpu)].put_packets(
                            scratch_buffer[peer][scratch_index : scratch_index + 1],
                            input_buffer[input_index : input_index + 1],
                            index,
                        )

        # Each rank get final result from scratch space
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for peer in range(gpu_size):
                if peer != gpu:
                    input_index = peer * gpu_size
                    scratch_index = peer if peer < gpu else peer - 1
                    scratch_index = gpu_size * (gpu_size - 1) + scratch_index * gpu_size
                    rank.unpack_packets(
                        input_buffer[input_index : input_index + gpu_size],
                        scratch_buffer[gpu][scratch_index : scratch_index + gpu_size],
                        peer,
                    )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
