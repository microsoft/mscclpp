# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllGather
from mscclpp.language.buffer import Buffer
from mscclpp.language.types import ChannelType, ReplicationPolicy


def allgather_allpair(name, gpus, gpus_per_node, instances, num_threads_per_block, min_message_size, max_message_size):
    collective = AllGather(gpus, 1, True)
    with MSCCLPPProgram(
        name,
        collective,
        gpus,
        instances,
        protocol="LL",
        replication_policy=ReplicationPolicy.interleaved,
        num_threads_per_block=num_threads_per_block,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
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
                        chan_type=ChannelType.proxy,
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
parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus")
parser.add_argument("--instances", type=int, help="number of instances")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_allpair(
    args.name,
    args.num_gpus,
    args.gpus_per_node,
    args.instances,
    args.num_threads_per_block,
    args.min_message_size,
    args.max_message_size,
)
