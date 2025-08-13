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
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
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

        # Ensuring the others ranks are ready to receive data
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)].signal(tb=0, relaxed=True)

        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)].wait(tb=0, data_sync=SyncType.after, relaxed=True)

        # Intranode AllGather
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_output_buffer()
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        dst_rank = Rank(dst_rank_id)
                        dst_buffer = dst_rank.get_output_buffer()
                        memory_channels[(dst_rank_id, src_rank_id)].put(
                            dst_buffer[src_rank_id : src_rank_id + 1], src_buffer[src_rank_id : src_rank_id + 1], tb=0
                        )

        # Internode AllGather
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_output_buffer()
                dst_rank_id = (gpu + gpus_per_node * (node + 1)) % gpu_size
                dst_rank = Rank(dst_rank_id)
                dst_buffer = dst_rank.get_output_buffer()
                port_channels[src_rank_id].signal(tb=0)
                port_channels[src_rank_id].wait(tb=0, data_sync=SyncType.after)
                port_channels[src_rank_id].put_with_signal_and_flush(
                    dst_buffer[src_rank_id : src_rank_id + 1], src_buffer[src_rank_id : src_rank_id + 1], tb=0
                )

        # Intranode AllGather
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                src_offset = gpu + gpus_per_node * ((node + 1) % nodes)
                src_rank = Rank(src_rank_id)
                src_buffer = src_rank.get_output_buffer()
                port_channels[src_rank_id].wait(tb=0, data_sync=SyncType.after)
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        dst_rank = Rank(dst_rank_id)
                        dst_buffer = dst_rank.get_output_buffer()
                        memory_channels[(dst_rank_id, src_rank_id)].put(
                            dst_buffer[src_offset : src_offset + 1], src_buffer[src_offset : src_offset + 1], tb=0
                        )

        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)].signal(tb=0, data_sync=SyncType.before)

        # Ensure all the communication finished
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                src_offset = gpu + gpus_per_node * ((node + 1) % nodes)
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)].wait(tb=0)

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
