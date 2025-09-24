# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_example(name, gpus_per_node, tbg_size, num_threads_per_block, min_message_size, max_message_size):
    nodes = 2
    num_gpus = nodes * gpus_per_node
    chunksperloop = 1
    collective = AllReduce(num_gpus, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        num_gpus,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        reuse_resources=False,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels
        memory_channels = {}
        port_channels = {}
        scratch_buffer = []
        tb_offset = 1
        total_tb = tbg_size + tb_offset
        tbg = ThreadBlockGroup(tb_list=[i for i in range(tb_offset, tb_offset + tbg_size)])

        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                next_rank_id = (gpu + gpus_per_node * (node + 1)) % num_gpus
                scratch_buffer.append(Buffer(src_rank_id, num_gpus * num_gpus))
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        memory_channels[(dst_rank_id, src_rank_id)] = MemoryChannel(dst_rank_id, src_rank_id)
                port_channels[src_rank_id] = PortChannel(next_rank_id, src_rank_id)

        # Transfer Data to Remote ScratchBuffer
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                src_rank = Rank(src_rank_id)
                src_input_buffer = src_rank.get_input_buffer()
                next_rank_id = (gpu + gpus_per_node * (node + 1)) % num_gpus
                index = src_rank_id * num_gpus
                next_index = next_rank_id * num_gpus
                for peer in range(gpus_per_node):
                    dst_rank_id = peer + gpus_per_node * node
                    if peer == gpu:
                        src_rank.copy_packets(
                            scratch_buffer[dst_rank_id][index : index + num_gpus],
                            src_input_buffer[0:num_gpus],
                            tb_group=tbg,
                        )
                    else:
                        memory_channels[(dst_rank_id, src_rank_id)].put_packets(
                            scratch_buffer[dst_rank_id][index : index + num_gpus],
                            src_input_buffer[0:num_gpus],
                            tb_group=tbg,
                        )
                port_channels[src_rank_id].read_put_packets(
                    scratch_buffer[next_rank_id][index : index + num_gpus],
                    scratch_buffer[src_rank_id][index : index + num_gpus],
                    tb=0,
                )
                for peer in range(gpus_per_node):
                    dst_rank_id = peer + gpus_per_node * node
                    if peer == gpu:
                        continue
                    memory_channels[(dst_rank_id, src_rank_id)].read_put_packets(
                        scratch_buffer[dst_rank_id][next_index : next_index + num_gpus],
                        scratch_buffer[src_rank_id][next_index : next_index + num_gpus],
                        tb_group=tbg,
                    )
                src_rank.reduce(
                    src_input_buffer[0:num_gpus],
                    [
                        scratch_buffer[src_rank_id][i * num_gpus : i * num_gpus + num_gpus]
                        for i in range(0, num_gpus)
                        if i != src_rank_id
                    ],
                    tb_group=tbg,
                    packet=True,
                )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
parser.add_argument("--tbg_size", type=int, help="number of thread blocks in the thread block group")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(
    args.name,
    args.gpus_per_node,
    args.tbg_size,
    args.num_threads_per_block,
    args.min_message_size,
    args.max_message_size,
)
