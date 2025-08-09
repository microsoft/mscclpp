# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    num_tb = 8
    collective = AllReduce(gpu_size, num_tb, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="Simple",
        instr_fusion=True,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels
        channels = {}
        for gpu in range(gpu_size):
            for tb in range(num_tb):
                for peer in range(gpu_size):
                    if peer != gpu:
                        channels[(peer, gpu, tb)] = MemoryChannel(peer, gpu)

        # Ensuring the data is ready on the remote side
        for gpu in range(gpu_size):
            for tb in range(num_tb):
                for peer in range(gpu_size):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].signal(tb, relaxed=True)

        for gpu in range(gpu_size):
            for tb in range(num_tb):
                for peer in range(gpu_size):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].wait(tb, data_sync=SyncType.after, relaxed=True)

        # Main AllReduce Logic
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for tb in range(num_tb):
                index = gpu * num_tb + tb
                src_chunk = input_buffer[index : index + 1]

                chunks = []
                for peer in range(gpu_size):
                    if gpu != peer:
                        peer_rank = Rank(peer)
                        peer_input_buffer = peer_rank.get_input_buffer()
                        chunks.append(peer_input_buffer[index : index + 1])
                        channels[(peer, gpu, tb)].reduce(src_chunk, [peer_input_buffer[index : index + 1]], tb)

                for peer in range(gpu_size):
                    if gpu != peer:
                        peer_rank = Rank(peer)
                        peer_input_buffer = peer_rank.get_input_buffer()
                        channels[(peer, gpu, tb)].put(peer_input_buffer[index : index + 1], src_chunk, tb)

        # Synchronization and Finalization
        for gpu in range(gpu_size):
            for tb in range(num_tb):
                for peer in range(gpu_size):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].signal(tb, data_sync=SyncType.before)

        for gpu in range(gpu_size):
            for tb in range(num_tb):
                for peer in range(gpu_size):
                    if gpu != peer:
                        channels[(peer, gpu, tb)].wait(tb)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
