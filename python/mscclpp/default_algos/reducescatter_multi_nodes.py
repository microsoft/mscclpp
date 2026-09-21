# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hierarchical multi-node ReduceScatter for the low-latency packet protocol."""

from mscclpp.language.channel import MemoryChannel, PortChannel
from mscclpp.language.collectives import ReduceScatter
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Buffer, Rank
from mscclpp.language.thread_block_group import ThreadBlockGroup
from mscclpp.language.utils import AlgoSpec


def reducescatter_multi_nodes(
    spec: AlgoSpec,
    thread_block_group_size: int = 1,
) -> CollectiveProgram:
    """Build a hierarchical ReduceScatter across nodes and local GPUs."""
    if not isinstance(spec.collective, ReduceScatter):
        raise ValueError("reducescatter_multi_nodes requires a ReduceScatter collective")
    if spec.protocol != "LL":
        raise ValueError("reducescatter_multi_nodes requires protocol='LL'")
    if spec.world_size % spec.nranks_per_node != 0:
        raise ValueError("world_size must be divisible by nranks_per_node")
    if spec.collective.chunk_factor != 1:
        raise ValueError("reducescatter_multi_nodes requires chunk_factor=1")
    if not spec.in_place or not spec.collective.inplace:
        raise ValueError("reducescatter_multi_nodes requires in-place buffers")
    if thread_block_group_size <= 0:
        raise ValueError("thread_block_group_size must be positive")

    num_nodes = spec.world_size // spec.nranks_per_node
    gpus_per_node = spec.nranks_per_node
    total_gpus = spec.world_size

    with CollectiveProgram.from_spec(spec) as prog:
        local_receive_slots = (gpus_per_node - 1) * num_nodes
        local_send_offset = local_receive_slots
        remote_receive_offset = local_send_offset + num_nodes
        scratch_slots = remote_receive_offset + num_nodes - 1
        scratch_buffers = [Buffer(rank, scratch_slots) for rank in range(total_gpus)]
        logical_thread_blocks = (gpus_per_node - 1) + num_nodes
        thread_block_groups = [
            ThreadBlockGroup(
                tb_list=[
                    logical_block * thread_block_group_size + group_offset
                    for group_offset in range(thread_block_group_size)
                ]
            )
            for logical_block in range(logical_thread_blocks)
        ]
        global_thread_block_group = ThreadBlockGroup(
            tb_list=[
                thread_block
                for thread_block_group in thread_block_groups
                for thread_block in thread_block_group.tb_list
            ]
        )

        intra_node_channels: dict[tuple[int, int], MemoryChannel] = {}
        for node_id in range(num_nodes):
            for src_local_rank in range(gpus_per_node):
                for dst_local_rank in range(gpus_per_node):
                    if src_local_rank == dst_local_rank:
                        continue
                    src_rank = src_local_rank + node_id * gpus_per_node
                    dst_rank = dst_local_rank + node_id * gpus_per_node
                    intra_node_channels[(dst_rank, src_rank)] = MemoryChannel(
                        dst_rank,
                        src_rank,
                    )

        inter_node_channels: dict[tuple[int, int], PortChannel] = {}
        for reducer_local_rank in range(gpus_per_node):
            for owner_node_id in range(num_nodes):
                owner_rank = reducer_local_rank + owner_node_id * gpus_per_node
                for src_node_id in range(num_nodes):
                    if src_node_id == owner_node_id:
                        continue
                    src_rank = reducer_local_rank + src_node_id * gpus_per_node
                    inter_node_channels[(owner_rank, src_rank)] = PortChannel(
                        owner_rank,
                        src_rank,
                    )

        # Local GPU r reduces the chunks owned by local rank r on every node.
        for node_id in range(num_nodes):
            for src_local_rank in range(gpus_per_node):
                src_rank = src_local_rank + node_id * gpus_per_node
                src_input = Rank(src_rank).get_input_buffer()
                for dst_local_rank in range(gpus_per_node):
                    if src_local_rank == dst_local_rank:
                        continue
                    dst_rank = dst_local_rank + node_id * gpus_per_node
                    local_peer_slot = src_local_rank if src_local_rank < dst_local_rank else src_local_rank - 1
                    dst_peer_slot = dst_local_rank - 1 if src_local_rank < dst_local_rank else dst_local_rank
                    peer_thread_blocks = thread_block_groups[dst_peer_slot].tb_list
                    for owner_node_id in range(num_nodes):
                        chunk_index = dst_local_rank + owner_node_id * gpus_per_node
                        scratch_slot = local_peer_slot * num_nodes + owner_node_id
                        intra_node_channels[(dst_rank, src_rank)].put_packets(
                            scratch_buffers[dst_rank][scratch_slot : scratch_slot + 1],
                            src_input[chunk_index : chunk_index + 1],
                            tb=peer_thread_blocks[owner_node_id % len(peer_thread_blocks)],
                        )

        # Reduce each same-local-rank shard locally, then send every remote-node
        # partial directly to the same-local-rank owner.
        local_reduce_offset = gpus_per_node - 1
        for src_node_id in range(num_nodes):
            for reducer_local_rank in range(gpus_per_node):
                src_rank = reducer_local_rank + src_node_id * gpus_per_node
                rank = Rank(src_rank)
                input_buffer = rank.get_input_buffer()

                for owner_node_id in range(num_nodes):
                    thread_block_group = thread_block_groups[local_reduce_offset + owner_node_id]
                    chunk_index = reducer_local_rank + owner_node_id * gpus_per_node
                    owner_rank = chunk_index
                    local_packets = []
                    for peer_local_rank in range(gpus_per_node):
                        if peer_local_rank == reducer_local_rank:
                            continue
                        local_peer_slot = (
                            peer_local_rank if peer_local_rank < reducer_local_rank else peer_local_rank - 1
                        )
                        scratch_slot = local_peer_slot * num_nodes + owner_node_id
                        local_packets.append(scratch_buffers[src_rank][scratch_slot : scratch_slot + 1])

                    local_reduced_chunk = input_buffer[chunk_index : chunk_index + 1]
                    if local_packets:
                        rank.reduce(
                            local_reduced_chunk,
                            local_packets,
                            tb_group=thread_block_group,
                            packet=True,
                        )

                    if num_nodes == 1:
                        continue

                    local_packet_slot = local_send_offset + owner_node_id
                    rank.copy_packets(
                        scratch_buffers[src_rank][local_packet_slot : local_packet_slot + 1],
                        local_reduced_chunk,
                        tb_group=thread_block_group,
                    )
                    if src_node_id == owner_node_id:
                        continue

                    remote_node_slot = src_node_id if src_node_id < owner_node_id else src_node_id - 1
                    inter_node_channels[(owner_rank, src_rank)].read_put_packets(
                        scratch_buffers[owner_rank][
                            remote_receive_offset + remote_node_slot : remote_receive_offset + remote_node_slot + 1
                        ],
                        scratch_buffers[src_rank][local_packet_slot : local_packet_slot + 1],
                        tb_group=thread_block_group,
                    )

        if num_nodes == 1:
            return prog

        # Every owner consumes its local partial and remote-node partials as packets.
        for owner_rank in range(total_gpus):
            owner = Rank(owner_rank)
            owner_input = owner.get_input_buffer()
            owner_node_id = owner_rank // gpus_per_node
            thread_block_group = thread_block_groups[local_reduce_offset + owner_node_id]
            owner_chunk = owner_input[owner_rank : owner_rank + 1]

            owner.unpack_packets(
                owner_chunk,
                scratch_buffers[owner_rank][local_send_offset + owner_node_id : local_send_offset + owner_node_id + 1],
                tb_group=global_thread_block_group,
            )

            remote_packets = [
                scratch_buffers[owner_rank][
                    remote_receive_offset + remote_node_slot : remote_receive_offset + remote_node_slot + 1
                ]
                for remote_node_slot in range(num_nodes - 1)
            ]
            owner.reduce(
                owner_chunk,
                remote_packets,
                tb_group=global_thread_block_group,
                packet=True,
            )

    return prog


if __name__ == "__main__":
    import argparse

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
        collective=ReduceScatter(args.num_gpus, 1, True),
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

    prog = reducescatter_multi_nodes(spec, args.tbg)
    print(prog.to_json())
