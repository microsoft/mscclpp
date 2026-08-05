# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hierarchical multi-node AllGather for the low-latency packet protocol."""

from mscclpp.language.channel import MemoryChannel, PortChannel
from mscclpp.language.collectives import AllGather
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Buffer, Rank
from mscclpp.language.utils import AlgoSpec


def allgather_multi_nodes(spec: AlgoSpec) -> CollectiveProgram:
    """Build a hierarchical AllGather across nodes and local GPUs."""
    if not isinstance(spec.collective, AllGather):
        raise ValueError("allgather_multi_nodes requires an AllGather collective")
    if spec.protocol != "LL":
        raise ValueError("allgather_multi_nodes requires protocol='LL'")
    if spec.world_size % spec.nranks_per_node != 0:
        raise ValueError("world_size must be divisible by nranks_per_node")
    if spec.collective.chunk_factor != 1:
        raise ValueError("allgather_multi_nodes requires chunk_factor=1")
    if spec.in_place != spec.collective.inplace:
        raise ValueError("spec.in_place must match spec.collective.inplace")
    num_nodes = spec.world_size // spec.nranks_per_node
    gpus_per_node = spec.nranks_per_node
    total_gpus = spec.world_size

    with CollectiveProgram.from_spec(spec) as prog:
        scratch_slots = (gpus_per_node - 1) + 2 * (num_nodes - 1) + (gpus_per_node - 1) * (num_nodes - 1)
        scratch_buffers = [Buffer(rank, scratch_slots) for rank in range(total_gpus)]

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
        for local_rank in range(gpus_per_node):
            for src_node_id in range(num_nodes):
                for dst_node_id in range(num_nodes):
                    if src_node_id == dst_node_id:
                        continue
                    src_rank = local_rank + src_node_id * gpus_per_node
                    dst_rank = local_rank + dst_node_id * gpus_per_node
                    inter_node_channels[(dst_rank, src_rank)] = PortChannel(
                        dst_rank,
                        src_rank,
                    )

        thread_block_offset = 0
        local_sources = []
        for rank_id in range(total_gpus):
            rank = Rank(rank_id)
            output_chunk = rank.get_output_buffer()[rank_id : rank_id + 1]
            if spec.collective.inplace:
                local_sources.append(output_chunk)
            else:
                input_chunk = rank.get_input_buffer()[0:1]
                rank.copy(output_chunk, input_chunk, tb=0)
                local_sources.append(input_chunk)

        # Phase 0: exchange contributions within each node.
        phase_0_send_offset = thread_block_offset
        for node_id in range(num_nodes):
            for src_local_rank in range(gpus_per_node):
                for dst_local_rank in range(gpus_per_node):
                    if src_local_rank == dst_local_rank:
                        continue
                    src_rank = src_local_rank + node_id * gpus_per_node
                    dst_rank = dst_local_rank + node_id * gpus_per_node
                    scratch_slot = src_local_rank if src_local_rank < dst_local_rank else src_local_rank - 1
                    thread_block = dst_local_rank - 1 if src_local_rank < dst_local_rank else dst_local_rank
                    intra_node_channels[(dst_rank, src_rank)].put_packets(
                        scratch_buffers[dst_rank][scratch_slot : scratch_slot + 1],
                        local_sources[src_rank],
                        tb=phase_0_send_offset + thread_block,
                    )

        phase_0_unpack_offset = phase_0_send_offset + gpus_per_node - 1
        for node_id in range(num_nodes):
            for dst_local_rank in range(gpus_per_node):
                dst_rank = dst_local_rank + node_id * gpus_per_node
                rank = Rank(dst_rank)
                for src_local_rank in range(gpus_per_node):
                    if src_local_rank == dst_local_rank:
                        continue
                    src_rank = src_local_rank + node_id * gpus_per_node
                    scratch_slot = src_local_rank - 1 if dst_local_rank < src_local_rank else src_local_rank
                    rank.unpack_packets(
                        rank.get_output_buffer()[src_rank : src_rank + 1],
                        scratch_buffers[dst_rank][scratch_slot : scratch_slot + 1],
                        tb=phase_0_unpack_offset + scratch_slot,
                    )

        # Phase 1: exchange same-local-rank contributions across nodes.
        phase_1_send_offset = phase_0_unpack_offset + gpus_per_node - 1
        remote_receive_offset = gpus_per_node - 1
        local_packet_offset = remote_receive_offset + num_nodes - 1
        for local_rank in range(gpus_per_node):
            for src_node_id in range(num_nodes):
                src_rank = local_rank + src_node_id * gpus_per_node
                rank = Rank(src_rank)
                for dst_node_id in range(num_nodes):
                    if src_node_id == dst_node_id:
                        continue
                    thread_block = dst_node_id - 1 if src_node_id < dst_node_id else dst_node_id
                    local_packet_slot = local_packet_offset + thread_block
                    rank.copy_packets(
                        scratch_buffers[src_rank][local_packet_slot : local_packet_slot + 1],
                        local_sources[src_rank],
                        tb=phase_1_send_offset + thread_block,
                    )

                    dst_rank = local_rank + dst_node_id * gpus_per_node
                    remote_scratch_slot = remote_receive_offset + (
                        src_node_id if src_node_id < dst_node_id else src_node_id - 1
                    )
                    inter_node_channels[(dst_rank, src_rank)].read_put_packets(
                        scratch_buffers[dst_rank][remote_scratch_slot : remote_scratch_slot + 1],
                        scratch_buffers[src_rank][local_packet_slot : local_packet_slot + 1],
                        tb=phase_1_send_offset + thread_block,
                    )

        phase_1_unpack_offset = phase_1_send_offset + num_nodes - 1
        for local_rank in range(gpus_per_node):
            for dst_node_id in range(num_nodes):
                dst_rank = local_rank + dst_node_id * gpus_per_node
                rank = Rank(dst_rank)
                for src_node_id in range(num_nodes):
                    if src_node_id == dst_node_id:
                        continue
                    src_rank = local_rank + src_node_id * gpus_per_node
                    remote_node_slot = src_node_id - 1 if dst_node_id < src_node_id else src_node_id
                    scratch_slot = remote_receive_offset + remote_node_slot
                    rank.unpack_packets(
                        rank.get_output_buffer()[src_rank : src_rank + 1],
                        scratch_buffers[dst_rank][scratch_slot : scratch_slot + 1],
                        tb=phase_1_unpack_offset + remote_node_slot,
                    )

        # Phase 2: fan out remote-node contributions within each node.
        phase_2_send_offset = phase_1_unpack_offset + num_nodes - 1
        local_fanout_offset = local_packet_offset + num_nodes - 1
        for dst_node_id in range(num_nodes):
            for src_node_id in range(num_nodes):
                if src_node_id == dst_node_id:
                    continue
                remote_node_slot = src_node_id - 1 if dst_node_id < src_node_id else src_node_id
                for src_local_rank in range(gpus_per_node):
                    src_rank = src_local_rank + dst_node_id * gpus_per_node
                    remote_scratch_slot = remote_receive_offset + remote_node_slot
                    for dst_local_rank in range(gpus_per_node):
                        if src_local_rank == dst_local_rank:
                            continue
                        dst_rank = dst_local_rank + dst_node_id * gpus_per_node
                        local_peer_slot = src_local_rank if src_local_rank < dst_local_rank else src_local_rank - 1
                        fanout_slot = local_fanout_offset + local_peer_slot + (gpus_per_node - 1) * remote_node_slot
                        thread_block = (dst_local_rank - 1 if src_local_rank < dst_local_rank else dst_local_rank) + (
                            gpus_per_node - 1
                        ) * remote_node_slot
                        intra_node_channels[(dst_rank, src_rank)].read_put_packets(
                            scratch_buffers[dst_rank][fanout_slot : fanout_slot + 1],
                            scratch_buffers[src_rank][remote_scratch_slot : remote_scratch_slot + 1],
                            tb=phase_2_send_offset + thread_block,
                        )

        phase_2_unpack_offset = phase_2_send_offset + (num_nodes - 1) * (gpus_per_node - 1)
        for dst_node_id in range(num_nodes):
            for src_node_id in range(num_nodes):
                if src_node_id == dst_node_id:
                    continue
                remote_node_slot = src_node_id - 1 if dst_node_id < src_node_id else src_node_id
                for dst_local_rank in range(gpus_per_node):
                    dst_rank = dst_local_rank + dst_node_id * gpus_per_node
                    rank = Rank(dst_rank)
                    for src_local_rank in range(gpus_per_node):
                        if src_local_rank == dst_local_rank:
                            continue
                        src_rank = src_local_rank + src_node_id * gpus_per_node
                        local_peer_slot = src_local_rank - 1 if dst_local_rank < src_local_rank else src_local_rank
                        fanout_slot = local_fanout_offset + local_peer_slot + (gpus_per_node - 1) * remote_node_slot
                        thread_block = local_peer_slot + (gpus_per_node - 1) * remote_node_slot
                        rank.unpack_packets(
                            rank.get_output_buffer()[src_rank : src_rank + 1],
                            scratch_buffers[dst_rank][fanout_slot : fanout_slot + 1],
                            tb=phase_2_unpack_offset + thread_block,
                        )

    return prog


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, required=True)
    parser.add_argument("--num_threads_per_block", type=int, default=1024)
    parser.add_argument("--min_message_size", type=int, default=1 << 10)
    parser.add_argument("--max_message_size", type=int, default=8 << 20)
    parser.add_argument(
        "--in_place",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    algo_spec = AlgoSpec(
        name=args.name,
        collective=AllGather(args.num_gpus, 1, args.in_place),
        nranks_per_node=args.gpus_per_node,
        world_size=args.num_gpus,
        in_place=args.in_place,
        instances=1,
        protocol="LL",
        auto_sync=False,
        num_threads_per_block=args.num_threads_per_block,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        min_message_size=args.min_message_size,
        max_message_size=args.max_message_size,
        tags={"default": 1},
    )
    program = allgather_multi_nodes(algo_spec)
    print(program.to_json())
