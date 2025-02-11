# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllGather
from mscclpp.language.buffer import Buffer
from mscclpp.language.types import DataFormat, ChannelType, ReplicationPolicy


def allgather_hierarchical(name, gpus, gpus_per_node, instances, num_threads_per_block, min_message_size, max_message_size):
    nodes = int(gpus/gpus_per_node)
    collective = AllGather(gpus_per_node * nodes, 1, True)
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
        # scratch buffer partition:
        # intra-node broadcast: (gpus_per_node - 1) chunks
        # inter-node broadcast: (nodes - 1) chunks + (nodes - 1) tmp chunks
        # intra-node broadcast: (gpus_per_node - 1) chunks

        # Intra-node broadcast, send from output to scratch
        tb_offset_0_0 = 0
        scratch_offset_0 = 0
        for n in range(nodes):
            for r1 in range(gpus_per_node):
                for r2 in range(gpus_per_node):
                    if r1 == r2:
                        continue
                    self_rank = r1 + n * gpus_per_node
                    peer_rank = r2 + n * gpus_per_node
                    output_idx = self_rank
                    scratch_idx = r1 if r1 < r2 else r1 - 1
                    tb = r2 - 1 if r1 < r2 else r2
                    c = chunk(self_rank, Buffer.output, index=output_idx)
                    c.put_packet(peer_rank, Buffer.scratch, index=scratch_idx + scratch_offset_0, sendtb=tb + tb_offset_0_0)

        # Intra-node broadcast, recv from scratch to output
        tb_offset_0_1 = tb_offset_0_0 + gpus_per_node - 1
        for n in range(nodes):
            for r1 in range(gpus_per_node):
                for r2 in range(gpus_per_node):
                    if r1 == r2:
                        continue
                    self_rank = r1 + n * gpus_per_node
                    peer_rank = r2 + n * gpus_per_node
                    output_idx = peer_rank
                    scratch_idx = r2 - 1 if r1 < r2 else r2
                    tb = scratch_idx
                    c = chunk(self_rank, Buffer.scratch, index=scratch_idx + scratch_offset_0)
                    c.copy_packet(self_rank, Buffer.output, index=output_idx, sendtb=tb + tb_offset_0_1)

        # Inter-node same-rank broadcast, send from output to scratch
        tb_offset_1_0 = tb_offset_0_1 + gpus_per_node - 1
        scratch_offset_1_0 = scratch_offset_0 + gpus_per_node - 1
        scratch_offset_1_1 = scratch_offset_1_0 + nodes - 1
        for r in range(gpus_per_node):
            for n1 in range(nodes):
                for n2 in range(nodes):
                    if n1 == n2:
                        continue
                    self_rank = r + n1 * gpus_per_node
                    peer_rank = r + n2 * gpus_per_node
                    output_idx = self_rank
                    scratch_idx = n1 if n1 < n2 else n1 - 1
                    tb = n2 - 1 if n1 < n2 else n2
                    c = chunk(self_rank, Buffer.output, index=output_idx)
                    c.put_packet(peer_rank, Buffer.scratch, index=scratch_idx + scratch_offset_1_0, sendtb=tb + tb_offset_1_0,
                                 chan_type=ChannelType.proxy, temp_buffer=Buffer.scratch,
                                 temp_buffer_index=tb + scratch_offset_1_1)

        # Inter-node same-rank broadcast, recv from scratch to output
        tb_offset_1_1 = tb_offset_1_0 + nodes - 1
        for r in range(gpus_per_node):
            for n1 in range(nodes):
                for n2 in range(nodes):
                    if n1 == n2:
                        continue
                    self_rank = r + n1 * gpus_per_node
                    peer_rank = r + n2 * gpus_per_node
                    output_idx = peer_rank
                    scratch_idx = n2 - 1 if n1 < n2 else n2
                    tb = scratch_idx
                    c = chunk(self_rank, Buffer.scratch, index=scratch_idx + scratch_offset_1_0)
                    c.copy_packet(self_rank, Buffer.output, index=output_idx, sendtb=tb + tb_offset_1_1)

        # Intra-node broadcast, send from scratch to scratch
        tb_offset_2_0 = tb_offset_1_1 + nodes - 1
        scratch_offset_2_0 = scratch_offset_1_0
        scratch_offset_2_1 = scratch_offset_1_1 + nodes - 1
        for n1 in range(nodes):
            for n2 in range(nodes):
                if n1 == n2:
                    continue
                for r1 in range(gpus_per_node):
                    for r2 in range(gpus_per_node):
                        if r1 == r2:
                            continue
                        self_rank = r1 + n1 * gpus_per_node
                        peer_rank = r2 + n1 * gpus_per_node
                        self_scratch_idx = n2 - 1 if n1 < n2 else n2
                        peer_scratch_idx = (r1 if r1 < r2 else r1 - 1) + (gpus_per_node - 1) * (n2 - 1 if n1 < n2 else n2)
                        tb = (r2 - 1 if r1 < r2 else r2) + (gpus_per_node - 1) * (n2 - 1 if n1 < n2 else n2)
                        c = chunk(self_rank, Buffer.scratch, index=self_scratch_idx + scratch_offset_2_0)
                        c.put_packet(peer_rank, Buffer.scratch, index=peer_scratch_idx + scratch_offset_2_1, sendtb=tb + tb_offset_2_0, src_format=DataFormat.packet)

        # Intra-node broadcast, recv from scratch to output
        tb_offset_2_1 = tb_offset_2_0 + (nodes - 1) * (gpus_per_node - 1)
        for n1 in range(nodes):
            for n2 in range(nodes):
                if n1 == n2:
                    continue
                for r1 in range(gpus_per_node):
                    for r2 in range(gpus_per_node):
                        if r1 == r2:
                            continue
                        self_rank = r1 + n1 * gpus_per_node
                        output_idx = r2 + n2 * gpus_per_node
                        scratch_idx = (r2 - 1 if r1 < r2 else r2) + (gpus_per_node - 1) * (n2 - 1 if n1 < n2 else n2)
                        tb = scratch_idx
                        c = chunk(self_rank, Buffer.scratch, index=scratch_idx + scratch_offset_2_1)
                        c.copy_packet(self_rank, Buffer.output, index=output_idx, sendtb=tb + tb_offset_2_1)

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

allgather_hierarchical(
    args.name, args.num_gpus, args.gpus_per_node, args.instances, args.num_threads_per_block, args.min_message_size, args.max_message_size
)