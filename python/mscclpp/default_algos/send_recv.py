# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def send_recv_test(name, nnodes, gpus_per_node, split_mask):
    gpu_size = nnodes * gpus_per_node
    collective = SendRecv(gpu_size, 1, False)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="Simple",
        num_threads_per_block=1024,
        use_double_scratch_buffer=False,
        min_message_size=0,
        max_message_size=2**64 - 1,
        instances=4
    ):
        # Creating separate port channels for next and prev directions.
        # When prev and next are the same peer (e.g., 2-node ring), both channels go to the same peer
        # and get distinct tags. To ensure cross-rank tag matching (rank A's prev_channel signal
        # arrives at rank B's next_channel wait), we create channels in opposite order for the
        # "higher" rank so that tags cross-match:
        #   Lower rank:  [next(tag0), prev(tag1)]
        #   Higher rank:  [prev(tag0), next(tag1)]
        # Then lower.prev(tag1) == higher.next(tag1) ✓ and higher.prev(tag0) == lower.next(tag0) ✓
        # When prev != next (3+ nodes), each channel targets a different peer so each gets tag 0
        # and this ordering doesn't matter.
        group_size = split_mask + 1
        num_groups = gpu_size // group_size
        next_channels = {}  # channel for sending to next rank
        prev_channels = {}  # channel for receiving from prev rank
        prev_next_ids = {}
        for node in range(nnodes):
            for gpu in range(gpus_per_node):
                global_rank_id = gpu + gpus_per_node * node
                position_in_group = global_rank_id & split_mask
                group_id = global_rank_id // group_size
                next_group_id = (group_id + 1) % num_groups
                next_global_rank_id = next_group_id * group_size + position_in_group
                prev_group_id = (group_id - 1 + num_groups) % num_groups
                prev_global_rank_id = prev_group_id * group_size + position_in_group
                if prev_global_rank_id == next_global_rank_id and global_rank_id > prev_global_rank_id:
                    # Higher rank: create prev first, then next (swapped order)
                    prev_channels[global_rank_id] = PortChannel(prev_global_rank_id, global_rank_id)
                    next_channels[global_rank_id] = PortChannel(next_global_rank_id, global_rank_id)
                else:
                    # Lower rank or different peers: create next first, then prev
                    next_channels[global_rank_id] = PortChannel(next_global_rank_id, global_rank_id)
                    prev_channels[global_rank_id] = PortChannel(prev_global_rank_id, global_rank_id)
                prev_next_ids[global_rank_id] = (prev_global_rank_id, next_global_rank_id)

        # sync with the next rank and the previous rank in the group
        for node in range(nnodes):
            for gpu in range(gpus_per_node):
                global_rank_id = gpu + gpus_per_node * node
                prev_global_rank_id, next_global_rank_id = prev_next_ids[global_rank_id]
                prev_channels[global_rank_id].signal(tb=0, data_sync=SyncType.none)
                next_channels[global_rank_id].wait(tb=0, data_sync=SyncType.after)

                src_rank = Rank(global_rank_id)
                src_buffer = src_rank.get_input_buffer()
                dst_rank = Rank(next_global_rank_id)
                dst_buffer = dst_rank.get_output_buffer()

                next_channels[global_rank_id].put_with_signal(dst_buffer[:], src_buffer[:], tb=0)
                prev_channels[global_rank_id].wait(tb=0, data_sync=SyncType.none)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--nnodes", type=int, default=1, help="number of nodes")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
parser.add_argument("--split_mask", type=lambda x: int(x, 0), default=0x3, help="split mask (e.g. 0x3)")

args = parser.parse_args()

send_recv_test(
    args.name, args.nnodes, args.gpus_per_node, args.split_mask
)
