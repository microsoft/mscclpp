# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allgather_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size, instances):
    # Packet (LL protocol) NVLS AllGather, tuned for small-message latency.
    #
    # Unlike allgather_nvls_zero_copy.py (Simple protocol + full-mesh barriers around
    # an NVLS multimem store), this variant carries an LL flag inside every packet, so
    # the broadcast is self-synchronizing and NO signal/wait barriers are needed. Each
    # rank packs its own chunk into scratch, multicasts those packets to every rank's
    # scratch via the switch (gstorepkt / MULTI_STORE_PKT), and unpacks locally.
    chunksperloop = 1
    collective = AllGather(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        instances=instances,
        protocol="LL",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=True,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Scratch holds packet-formatted chunks: gpu_size slots per rank, one per source rank.
        scratch_buffer = []
        for gpu in range(gpu_size):
            scratch_buffer.append(Buffer(gpu, gpu_size))

        # NVLS multicast channel bound to the scratch buffer (the packet staging area).
        nvls_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.scratch)

        # Pack each rank's own chunk into its scratch slot `gpu`, then multicast those
        # packets to slot `gpu` of every rank's scratch via the switch. copy + broadcast
        # share tb=0 so the packed packets are produced before they are read.
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            output_buffer = rank.get_output_buffer()
            rank.copy_packets(scratch_buffer[gpu][gpu : gpu + 1], output_buffer[gpu : gpu + 1], tb=0)
            nvls_chan.at_rank(gpu).broadcast_packets(
                src_chunk=scratch_buffer[gpu][gpu : gpu + 1], buffer_offset=gpu, size=1, tb=0
            )

        # Unpack every slot from local scratch into the output buffer. Each unpack waits
        # on the packet flag delivered by the owning rank's multicast (no barrier needed).
        # Slot j is unpacked on tb=j to parallelize across thread blocks.
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            output_buffer = rank.get_output_buffer()
            for j in range(gpu_size):
                rank.unpack_packets(output_buffer[j : j + 1], scratch_buffer[gpu][j : j + 1], tb=j)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")
parser.add_argument("--instances", type=int, default=1, help="number of instances (parallel threadblocks)")

args = parser.parse_args()

allgather_example(
    args.name,
    args.num_gpus,
    args.num_threads_per_block,
    args.min_message_size,
    args.max_message_size,
    args.instances,
)
