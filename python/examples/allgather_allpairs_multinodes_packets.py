# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllGather
from mscclpp.language.buffer import Buffer
from mscclpp.language.types import ChannelType, ReplicationPolicy


def allgather_multinodes_allpair(gpus, gpus_per_node, instances):
    """
    Implements a multi-node allgather collective using an allpairs algorithm with MSCCL++ DSL.
    Steps:
    1. Each rank sends a chunk to all other ranks' scratch buffers using packet format.
    2. Copy the chunk from the scratch buffer to the output buffer using packet format.
    """
    collective = AllGather(gpus, 1, True)
    with MSCCLPPProgram(
        "allgather_multinodes_allpair",
        collective,
        gpus,
        instances,
        protocol="LL",
        replication_policy=ReplicationPolicy.interleaved,
        num_threads_per_block=1024,
    ):
        for g in range(gpus):
            src_rank = g
            c = chunk(src_rank, Buffer.input, 0, 1)
            for peer in range(1, gpus):
                dst_rank = (src_rank + peer) % gpus
                tb = dst_rank if dst_rank < src_rank else dst_rank - 1
                if src_rank // gpus_per_node == dst_rank // gpus_per_node:
                    c.put_packet(dst_rank, Buffer.scratch, index=src_rank, sendtb=tb)
                else:
                    c.put_packet(
                        dst_rank,
                        Buffer.scratch,
                        index=src_rank,
                        sendtb=tb,
                        chan_type=ChannelType.port,
                        temp_buffer=Buffer.scratch,
                        temp_buffer_index=src_rank,
                    )

        # Copying packet from local scratch buffer to local buffer
        for g in range(gpus):
            src_rank = g
            src_offset = src_rank
            for peer in range(1, gpus):
                dst_rank = (g + peer) % gpus
                tb = src_offset if src_offset < dst_rank else src_offset - 1
                c = chunk(dst_rank, Buffer.scratch, src_offset, 1)
                c.copy_packet(dst_rank, Buffer.output, src_offset, sendtb=tb + gpus - 1)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("gpus_per_node", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

allgather_multinodes_allpair(
    args.num_gpus,
    args.gpus_per_node,
    args.instances,
)
