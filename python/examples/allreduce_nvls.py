# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllReduce
from mscclpp.language.buffer import Buffer


def allreduce_nvls(gpus, instances):
    """
    Allreduce via NVLS channel
    Steps:
    1. Sync all the ranks to make sure the data is ready.
    2. Call group_load_reduce to reduce the data.
    3. Call group_store to propagate the data to all the ranks.
    """
    size = gpus
    chunksperloop = gpus
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLPPProgram(
        "allreduce_nvls",
        collective,
        size,
        instances,
    ):
        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            index = rank
            c = chunk(rank, Buffer.input, index)
            reduce_chunks = []
            # make sure the data is ready
            for nghr in range(size):
                if rank != nghr:
                    c_peer = chunk(nghr, Buffer.input, index)
                    reduce_chunks.append(c_peer)
                    c.signal(nghr, Buffer.input, index, sendtb=0)
            for nghr in range(size):
                if rank != nghr:
                    c.wait(nghr, Buffer.input, index, recvtb=0)
            c = c.group_load_reduce(reduce_chunks, recvtb=0)
            ngbrs = [nghr for nghr in range(size) if nghr != rank]
            c.group_store(ngbrs, sendtb=0)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

allreduce_nvls(args.num_gpus, args.instances)
