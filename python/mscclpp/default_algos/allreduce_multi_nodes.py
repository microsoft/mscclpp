# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Multi-node AllReduce implementation using packet-based communication.
This implements a hierarchical AllReduce: intra-node allreduce followed by
inter-node exchange and final intra-node allreduce.
"""

import argparse
from mscclpp.language.utils import AlgoSpec
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_multi_nodes(spec: AlgoSpec, thread_block_group_size: int) -> CollectiveProgram:
    """
    Implements a multi-node AllReduce using a hierarchical approach:
    1. Intra-node allreduce
    2. Inter-node exchange (exchange reduced data between nodes)
    3. Intra-node allreduce
    """
    # Configuration constants
    num_nodes = spec.world_size // spec.nranks_per_node
    gpus_per_node = spec.nranks_per_node
    total_gpus = num_nodes * gpus_per_node
    packets_per_gpu = num_nodes

    with CollectiveProgram.from_spec(spec) as prog:
        # Initialize communication channels and buffers
        intra_node_memory_channels = {}
        inter_node_port_channels = {}
        scratch_buffers = []
        thread_block_offset = 1
        thread_block_groups = []
        global_intra_node_tbg = ThreadBlockGroup(
            tb_list=[
                i
                for i in range(thread_block_offset, thread_block_offset + (gpus_per_node - 1) * thread_block_group_size)
            ]
        )
        for i in range(gpus_per_node - 1):
            thread_block_groups.append(
                ThreadBlockGroup(
                    tb_list=[
                        i
                        for i in range(
                            thread_block_offset + i * thread_block_group_size,
                            thread_block_offset + (i + 1) * thread_block_group_size,
                        )
                    ]
                )
            )

        scratch_buffer_size = 2 * total_gpus + packets_per_gpu * num_nodes
        for node_id in range(num_nodes):
            for local_gpu_id in range(gpus_per_node):
                current_rank_id = local_gpu_id + gpus_per_node * node_id
                scratch_buffers.append(Buffer(current_rank_id, scratch_buffer_size))
                for peer_gpu_id in range(gpus_per_node):
                    if peer_gpu_id != local_gpu_id:
                        peer_rank_id = peer_gpu_id + gpus_per_node * node_id
                        intra_node_memory_channels[(peer_rank_id, current_rank_id)] = MemoryChannel(
                            peer_rank_id, current_rank_id
                        )
                for peer_node_id in range(num_nodes):
                    if peer_node_id != node_id:
                        peer_node_rank_id = (local_gpu_id + gpus_per_node * peer_node_id) % total_gpus
                        inter_node_port_channels[(current_rank_id, peer_node_rank_id)] = PortChannel(peer_node_rank_id, current_rank_id)

        # AllReduce
        for node_id in range(num_nodes):
            for local_gpu_id in range(gpus_per_node):
                current_rank_id = local_gpu_id + gpus_per_node * node_id
                current_rank = Rank(current_rank_id)
                input_buffer = current_rank.get_input_buffer()

                # Intra Node Exchange Data
                for peer_gpu_id in range(gpus_per_node):
                    peer_rank_id = peer_gpu_id + gpus_per_node * node_id
                    peer_data_offset = peer_gpu_id * packets_per_gpu
                    tbg_id = peer_gpu_id if peer_gpu_id < local_gpu_id else peer_gpu_id - 1
                    if peer_gpu_id != local_gpu_id:
                        intra_node_memory_channels[(peer_rank_id, current_rank_id)].put_packets(
                            scratch_buffers[peer_rank_id][
                                local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu
                            ],
                            input_buffer[peer_data_offset : peer_data_offset + packets_per_gpu],
                            tb_group=thread_block_groups[tbg_id],
                        )

                # Intra Node Reduce
                other_gpu_data = [
                    scratch_buffers[current_rank_id][
                        packets_per_gpu * gpu_idx : packets_per_gpu * gpu_idx + packets_per_gpu
                    ]
                    for gpu_idx in range(gpus_per_node)
                    if gpu_idx != local_gpu_id
                ]
                current_rank.reduce(
                    input_buffer[local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu],
                    other_gpu_data,
                    tb_group=global_intra_node_tbg,
                    packet=True,
                )

                current_rank.copy_packets(
                    scratch_buffers[current_rank_id][
                        local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu
                    ],
                    input_buffer[local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu],
                    tb_group=global_intra_node_tbg,
                )

                current_rank.barrier(
                    tb_list=[i for i in range(thread_block_offset + (gpus_per_node - 1) * thread_block_group_size)]
                )

                inter_node_offset = total_gpus
                for peer_node_id in range(num_nodes):
                    if peer_node_id != node_id:
                        peer_node_rank_id = (local_gpu_id + gpus_per_node * peer_node_id) % total_gpus
                        inter_node_port_channels[(current_rank_id, peer_node_rank_id)].put_packets(
                            scratch_buffers[peer_node_rank_id][
                                inter_node_offset
                                + node_id * packets_per_gpu : inter_node_offset
                                + node_id * packets_per_gpu
                                + packets_per_gpu
                            ],
                            scratch_buffers[current_rank_id][
                                local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu
                            ],
                            tb=0,
                        )

                # Reduce Received Data from Remote Node
                inter_node_data = [
                    scratch_buffers[current_rank_id][
                        inter_node_offset
                        + peer_node_id * packets_per_gpu : inter_node_offset
                        + peer_node_id * packets_per_gpu
                        + packets_per_gpu
                    ]
                    for peer_node_id in range(num_nodes)
                    if peer_node_id != node_id
                ]
                current_rank.reduce(
                    input_buffer[local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu],
                    inter_node_data,
                    tb_group=global_intra_node_tbg,
                    packet=True,
                )

                current_rank.copy_packets(
                    scratch_buffers[current_rank_id][
                        inter_node_offset
                        + node_id * packets_per_gpu : inter_node_offset
                        + node_id * packets_per_gpu
                        + packets_per_gpu],
                    input_buffer[local_gpu_id * packets_per_gpu : local_gpu_id * packets_per_gpu + packets_per_gpu],
                    tb_group=global_intra_node_tbg,
                )

                # Broadcast Reduced Data
                broadcast_offset = total_gpus + packets_per_gpu * num_nodes
                for peer_gpu_id in range(gpus_per_node):
                    peer_rank_id = peer_gpu_id + gpus_per_node * node_id

                    if peer_gpu_id != local_gpu_id:
                        tbg_id = peer_gpu_id if peer_gpu_id < local_gpu_id else peer_gpu_id - 1
                        intra_node_memory_channels[(peer_rank_id, current_rank_id)].read_put_packets(
                            scratch_buffers[peer_rank_id][
                                broadcast_offset
                                + local_gpu_id * packets_per_gpu : broadcast_offset
                                + local_gpu_id * packets_per_gpu
                                + packets_per_gpu
                            ],
                            scratch_buffers[current_rank_id][
                                inter_node_offset
                                + node_id * packets_per_gpu : inter_node_offset
                                + node_id * packets_per_gpu
                                + packets_per_gpu
                            ],
                            tb_group=thread_block_groups[tbg_id],
                        )

                # Unpack Data Received from other GPUs in the same node
                for peer_gpu_id in range(gpus_per_node):
                    if peer_gpu_id != local_gpu_id:
                        tbg_id = peer_gpu_id if peer_gpu_id < local_gpu_id else peer_gpu_id - 1
                        current_rank.unpack_packets(
                            input_buffer[
                                peer_gpu_id * packets_per_gpu : peer_gpu_id * packets_per_gpu + packets_per_gpu
                            ],
                            scratch_buffers[current_rank_id][
                                broadcast_offset
                                + peer_gpu_id * packets_per_gpu : broadcast_offset
                                + peer_gpu_id * packets_per_gpu
                                + packets_per_gpu
                            ],
                            tb_group=thread_block_groups[tbg_id],
                        )

    return prog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of the program")
    parser.add_argument("--num_gpus", type=int, help="total number of gpus")
    parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
    parser.add_argument("--tbg", type=int, default=1, help="thread block group size")
    parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
    parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
    parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

    args = parser.parse_args()

    spec = AlgoSpec(
        name=args.name,
        collective=AllReduce(args.num_gpus, 1, True),
        nranks_per_node=args.gpus_per_node,
        world_size=args.num_gpus,
        in_place=True,
        instances=1,
        protocol="LL",
        auto_sync=False,
        num_threads_per_block=args.num_threads_per_block,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        min_message_size=args.min_message_size,
        max_message_size=args.max_message_size,
    )

    prog = allreduce_multi_nodes(spec, args.tbg)
    print(prog.to_json())
