# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allgather_example(name, gpus_per_node, num_threads_per_block, min_message_size, max_message_size):
    nodes = 2
    gpu_size = nodes * gpus_per_node
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
            scratch_buffer.append(Buffer(gpu, gpu_size))

        # Copying data to the scratch buffer
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_output_buffer()
            rank.copy_packets(scratch_buffer[gpu][gpu : gpu + 1], input_buffer[gpu : gpu + 1], tb=gpu)

        # Intra node put pkt
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank = peer + gpus_per_node * node
                        ch = MemoryChannel(dst_rank, src_rank)
                        ch.read_put_packets(
                            scratch_buffer[dst_rank][src_rank : src_rank + 1],
                            scratch_buffer[src_rank][src_rank : src_rank + 1],
                            tb=peer,
                        )

        # Inter node put pkt
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank = gpu + gpus_per_node * node
                dst_rank = gpu + gpus_per_node * ((node + 1) % nodes)
                ch = PortChannel(dst_rank, src_rank)
                ch.read_put_packets(
                    scratch_buffer[dst_rank][src_rank : src_rank + 1],
                    scratch_buffer[src_rank][src_rank : src_rank + 1],
                    tb=gpu,
                )

        # Intra node put pkt
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank = gpu + gpus_per_node * node
                src_offset = gpu + gpus_per_node * ((node + 1) % nodes)
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank = peer + gpus_per_node * node
                        ch = MemoryChannel(dst_rank, src_rank)
                        ch.read_put_packets(
                            scratch_buffer[dst_rank][src_offset : src_offset + 1],
                            scratch_buffer[src_rank][src_offset : src_offset + 1],
                            tb=peer,
                        )

        # Copying packet from local scratch buffer to local buffer
        for gpu in range(gpu_size):
            for peer in range(1, gpu_size):
                dst_rank = (gpu + peer) % gpu_size
                rank = Rank(gpu)
                input_buffer = rank.get_output_buffer()
                rank.unpack_packets(
                    input_buffer[dst_rank : dst_rank + 1], scratch_buffer[gpu][dst_rank : dst_rank + 1], tb=peer
                )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(
    args.name, args.gpus_per_node, args.num_threads_per_block, args.min_message_size, args.max_message_size
)
