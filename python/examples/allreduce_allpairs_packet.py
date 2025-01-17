# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllReduce
from mscclpp.language.buffer import Buffer


def allreduce_allpairs(gpus, instances):
    """
    AllReduce with all pairs algorithm using packets format.
    Steps:
    1. Each rank sends the nth chunk to the nth rank into scratch space.
    2. Each rank performs a local reduction on the nth chunk. Then sends the reduced data to all other ranks.
    3. Each rank retrieves the final result from scratch space.
    """
    size = gpus
    chunksperloop = gpus * gpus
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLPPProgram(
        "allreduce_packets",
        collective,
        size,
        instances,
        protocol="LL",
        use_double_scratch_buffer=True,
    ):
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for tb in range(size):
                if tb == r1:
                    continue
                remote_rank = tb
                index = remote_rank * size
                c = chunk(r1, Buffer.input, index, size)
                c.put_packet(remote_rank, "scratch", index=r1 * size, sendtb=tb)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(size):
                c = chunk(r, Buffer.input, r * size + index)
                for peer in range(size):
                    if peer != r:
                        c.reduce_packet(chunk(r, "scratch", peer * size + index), recvtb=index)
                for peer in range(size):
                    if peer != r:
                        c.put_packet(peer, "scratch", (size * size) + r * size + index, sendtb=index)

        # Each rank get final result from scratch space
        for r in range(size):
            for peer in range(size):
                if peer != r:
                    c = chunk(r, "scratch", size * size + peer * size, size)
                    c.copy_packet(r, Buffer.input, peer * size, sendtb=peer)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances)
