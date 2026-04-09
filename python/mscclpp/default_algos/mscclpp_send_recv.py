# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def send_recv_test_ring_even_ranks(name, nnodes, gpus_per_node):
    nranks = nnodes * gpus_per_node

    if nranks < 2:
        raise ValueError("This test requires at least 2 ranks")
    if nranks % 2 != 0:
        raise ValueError(
            f"This odd/even ring schedule requires an even number of ranks, got {nranks}"
        )

    collective = TestCollective(nranks, 1, 1)

    with CollectiveProgram(
        name,
        collective,
        nranks,
        protocol="Simple",
        num_threads_per_block=1024,
        use_double_scratch_buffer=False,
        min_message_size=0,
        max_message_size=2**64 - 1,
        instances=2,
    ):
        next_channels = {}
        prev_channels = {}

        # --------------------------------------------------------------
        # Classic ring across all ranks:
        #   prev = (rank - 1 + nranks) % nranks
        #   next = (rank + 1) % nranks
        # --------------------------------------------------------------
        for rank in range(nranks):
            prev_rank = (rank - 1 + nranks) % nranks
            next_rank = (rank + 1) % nranks

            # Deterministic channel creation order
            if (rank & 1) == 0:
                next_channels[rank] = PortChannel(next_rank, rank)
                prev_channels[rank] = PortChannel(prev_rank, rank)
            else:
                prev_channels[rank] = PortChannel(prev_rank, rank)
                next_channels[rank] = PortChannel(next_rank, rank)

                # --------------------------------------------------------------
        # --------------------------------------------------------------
        # Ring send/recv with explicit ACK
        #
        # Data path:
        #   sender:   put_with_signal() to next
        #   receiver: wait() from prev
        #
        # ACK path:
        #   receiver: signal() back to prev after data is available
        #   sender:   wait() for ACK from next before proceeding
        #
        # Even ranks: send first, then recv, then ACK prev, then wait ACK
        # Odd ranks : recv first, then ACK prev, then send, then wait ACK
        # --------------------------------------------------------------
        for rank in range(nranks):
            prev_rank = (rank - 1 + nranks) % nranks
            next_rank = (rank + 1) % nranks

            src_rank = Rank(rank)
            next_rank_obj = Rank(next_rank)

            src_buf = src_rank.get_input_buffer()
            next_out_buf = next_rank_obj.get_output_buffer()

            src_chunk = src_buf[0:src_buf.size]
            dst_chunk = next_out_buf[0:next_out_buf.size]

            ch_to_next = next_channels[rank]
            ch_from_prev = prev_channels[rank]

            if (rank & 1) == 0:
                # Send data to next and signal arrival
                ch_to_next.put_with_signal(
                    dst_chunk,
                    src_chunk,
                    tb=0,
                )

                # Wait for data from prev to become visible locally
                ch_from_prev.wait(
                    tb=0,
                    data_sync=SyncType.after,
                )

                # Ack back to prev that this rank has observed/consumed input
                ch_from_prev.signal(
                    tb=0,
                )

                # Wait for next rank to ack our outgoing transfer
                ch_to_next.wait(
                    tb=0,
                )

            else:
                # Wait for data from prev first
                ch_from_prev.wait(
                    tb=0,
                    data_sync=SyncType.after,
                )

                # Ack back to prev that this rank has observed/consumed input
                ch_from_prev.signal(
                    tb=0,
                )

                # Then send data to next
                ch_to_next.put_with_signal(
                    dst_chunk,
                    src_chunk,
                    tb=0,
                )

                # Wait for next rank to ack our outgoing transfer
                ch_to_next.wait(
                    tb=0,
                )
        # --------------------------------------------------------------
        # Ring send/recv
        #
        # Even ranks: send first, then wait
        # Odd ranks : wait first, then send
        #
        # This is safe for an even-sized ring and avoids the
        # single-rank-starter wave.
        # --------------------------------------------------------------
        '''
        for rank in range(nranks):
            prev_rank = (rank - 1 + nranks) % nranks
            next_rank = (rank + 1) % nranks

            src_rank = Rank(rank)
            next_rank_obj = Rank(next_rank)

            src_buf = src_rank.get_input_buffer()
            next_out_buf = next_rank_obj.get_output_buffer()

            src_chunk = src_buf[0:src_buf.size]
            dst_chunk = next_out_buf[0:next_out_buf.size]

            ch_to_next = next_channels[rank]
            ch_from_prev = prev_channels[rank]

            if (rank & 1) == 0:
                ch_to_next.put_with_signal_and_flush(
                    dst_chunk,
                    src_chunk,
                    tb=0,
                )
                ch_from_prev.wait(
                    tb=0,
                    data_sync=SyncType.after,
                )
            else:
                ch_from_prev.wait(
                    tb=0,
                    data_sync=SyncType.after,
                )
                ch_to_next.put_with_signal_and_flush(
                    dst_chunk,
                    src_chunk,
                    tb=0,
                )

        '''
        print(JSON())


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="name of the program")
parser.add_argument("--nnodes", type=int, default=1, help="number of nodes")
parser.add_argument("--gpus_per_node", type=int, required=True, help="number of GPUs per node")

args = parser.parse_args()

send_recv_test_ring_even_ranks(
    args.name,
    args.nnodes,
    args.gpus_per_node,
)
