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
    collective = TestCollective(gpu_size, 1, 1)

    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="Simple",
        num_threads_per_block=1024,
        use_double_scratch_buffer=False,
        min_message_size=0,
        max_message_size=2**64 - 1,
        instances=1,   # ✅ correctness-first
    ):

        # Ring grouping
        group_size = split_mask + 1
        num_groups = gpu_size // group_size

        next_channels = {}
        prev_channels = {}
        prev_next_ids = {}

        # ------------------------------------------------------------------
        # Channel creation (parity-based for deterministic tag matching)
        # ------------------------------------------------------------------
        for node in range(nnodes):
            for gpu in range(gpus_per_node):
                rank = gpu + gpus_per_node * node

                pos = rank & split_mask
                group = rank // group_size

                next_group = (group + 1) % num_groups
                prev_group = (group - 1 + num_groups) % num_groups

                next_rank = next_group * group_size + pos
                prev_rank = prev_group * group_size + pos

                # ✅ parity-based creation order
                if (rank & 1) == 0:
                    next_channels[rank] = PortChannel(next_rank, rank)
                    prev_channels[rank] = PortChannel(prev_rank, rank)
                else:
                    prev_channels[rank] = PortChannel(prev_rank, rank)
                    next_channels[rank] = PortChannel(next_rank, rank)

                prev_next_ids[rank] = (prev_rank, next_rank)

        # ------------------------------------------------------------------
        # Ring send/recv (deadlock-free)
        # ------------------------------------------------------------------
        for node in range(nnodes):
            for gpu in range(gpus_per_node):
                rank = gpu + gpus_per_node * node
                prev_rank, next_rank = prev_next_ids[rank]

                ch_from_prev = prev_channels[rank]
                ch_to_next = next_channels[rank]

                src_rank = Rank(rank)
                src_buf = src_rank.get_input_buffer()
                src_chunk = src_buf[0:src_buf.size]

                dst_rank = Rank(next_rank)
                dst_buf = dst_rank.get_output_buffer()
                dst_chunk = dst_buf[0:dst_buf.size]

                if rank == 0:
                    # ✅ starter sends first
                    ch_to_next.put_with_signal_and_flush(
                        dst_chunk,
                        src_chunk,
                        tb=0,
                    )
                    # then receive from prev
                    ch_from_prev.wait(tb=0, data_sync=SyncType.after)
                else:
                    # ✅ everyone else receives first
                    ch_from_prev.wait(tb=0, data_sync=SyncType.after)
                    ch_to_next.put_with_signal_and_flush(
                        dst_chunk,
                        src_chunk,
                        tb=0,
                    )

        print(JSON())


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--nnodes", type=int, default=1, help="number of nodes")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
parser.add_argument(
    "--split_mask",
    type=lambda x: int(x, 0),
    default=0x3,
    help="split mask (e.g. 0x3)",
)

args = parser.parse_args()

send_recv_test(
    args.name,
    args.nnodes,
    args.gpus_per_node,
    args.split_mask,
)
