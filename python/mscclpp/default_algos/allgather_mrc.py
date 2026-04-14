# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.loop import LoopIterationContext
from mscclpp.language.utils import AlgoSpec


def allgather_mrc(spec: AlgoSpec) -> CollectiveProgram:
   size = spec.world_size
   
   with CollectiveProgram.from_spec(spec) as prog:
        # Port channels for inter-node communication
        port_channels = {}
        for n in range(size):
            src_rank = n
            next_rank = (n + 1) % size
            prev_rank = (n - 1) % size
            if src_rank != next_rank:
                port_channels[next_rank, src_rank] = PortChannel(next_rank, src_rank)
            if src_rank != prev_rank:
                port_channels[prev_rank, src_rank] = PortChannel(prev_rank, src_rank)

        # ===== Inter-node ring send + Intra-node AllGather =====
        for step in range(size - 1):
            for n in range(size):
                src_rank = n
                next_rank = (n + 1) % size
                prev_rank = (n - 1) % size
                offset = (n - step) % size
                recv_offset = (n - 1 - step) % size

                # Sending to the next node
                ch_to_next = port_channels[next_rank, src_rank]
                ch_from_prev = port_channels[prev_rank, src_rank]

                src_chunk = Rank(src_rank).get_output_buffer()[offset:offset + 1]
                dst_chunk_next = Rank(next_rank).get_output_buffer()[offset:offset + 1]

                if step == 0:
                    # Signal prev_rank that data is ready for it to read
                    ch_from_prev.signal(tb=0)
                    # Wait for signal from next_rank
                    ch_to_next.wait(tb=0)

                ch_to_next.put(dst_chunk_next, src_chunk, tb=0)
                ch_to_next.signal(tb=0)

                # Receiving from the previous node and sharing the data inside the node
                recv_src_chunk = Rank(src_rank).get_output_buffer()[recv_offset:recv_offset + 1]
                ch_from_prev.wait(tb=0)

        return prog
