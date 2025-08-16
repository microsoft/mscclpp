# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_example(name, gpus_per_node, num_threads_per_block, min_message_size, max_message_size):
    nodes = 2
    gpu_size = nodes * gpus_per_node
    chunksperloop = 1
    collective = ReduceScatter(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="LL",
        instr_fusion=True,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Scratch Buffers
        scratch_buffer = []
        for gpu in range(gpu_size):
            scratch_buffer.append(Buffer(gpu, gpu_size))

        # Creating Channels
        memory_channels = {}
        port_channels = {}
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                next_src_rank_id = (gpu + gpus_per_node * (node + 1)) % gpu_size
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)] = MemoryChannel(dst_rank_id, src_rank_id)
                port_channels[src_rank_id] = PortChannel(next_src_rank_id, src_rank_id)

        # Sharing Data Intra Node
        inter_node_data_offset = int(gpu_size / 2)
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + node * gpus_per_node
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_input_buffer()
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + node * gpus_per_node
                        next_dst_rank_id = peer + gpus_per_node * ((node + 1) % nodes)
                        dst_offset = gpu
                        r_dst_offset = gpu if gpu < peer else gpu - 1
                        r_dst_offset += inter_node_data_offset
                        memory_channels[(dst_rank_id, src_rank_id)].put_packets(
                            scratch_buffer[dst_rank_id][dst_offset : dst_offset + 1],
                            src_buffer[dst_rank_id : dst_rank_id + 1],
                            tb=0,
                        )
                        memory_channels[(dst_rank_id, src_rank_id)].put_packets(
                            scratch_buffer[dst_rank_id][r_dst_offset : r_dst_offset + 1],
                            src_buffer[next_dst_rank_id : next_dst_rank_id + 1],
                            tb=0,
                        )

        # Reducing Internode Data
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + node * gpus_per_node
                next_src_rank_id = gpu + gpus_per_node * ((node + 1) % nodes)
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_input_buffer()

                chunks = []
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        base_offset = peer if peer < gpu else peer - 1
                        offset = base_offset + inter_node_data_offset
                        chunks.append(scratch_buffer[src_rank_id][offset : offset + 1])

                if len(chunks) > 0:
                    src_rank.reduce(src_buffer[next_src_rank_id : next_src_rank_id + 1], chunks, tb=0, packet=True)
                src_rank.copy_packets(
                    scratch_buffer[src_rank_id][gpu : gpu + 1],
                    src_buffer[next_src_rank_id : next_src_rank_id + 1],
                    tb=0,
                )
                port_channels[src_rank_id].read_put_packets(
                    scratch_buffer[next_src_rank_id][gpu_size - 1 : gpu_size],
                    scratch_buffer[src_rank_id][gpu : gpu + 1],
                    tb=0,
                )

        # Final Reducing
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + node * gpus_per_node
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_input_buffer()

                chunks = [scratch_buffer[src_rank_id][gpu_size - 1 : gpu_size]]
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        offset = peer
                        chunks.append(scratch_buffer[src_rank_id][offset : offset + 1])

                src_rank.reduce(src_buffer[src_rank_id : src_rank_id + 1], chunks, tb=0, packet=True)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(
    args.name, args.gpus_per_node, args.num_threads_per_block, args.min_message_size, args.max_message_size
)
