# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllReduce
from mscclpp.language.buffer import Buffer


def allreduce_allpairs(gpus, instances):
    size = gpus
    chunksperloop = gpus * gpus
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLPPProgram(
        "allreduce_pairs",
        collective,
        size,
        instances,
        protocol="Simple",
    ):

        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            for tb in range(size):
                index = rank * size
                c = chunk(rank, Buffer.input, index + tb)
                # make sure the data is ready
                for nghr in range(size):
                    peer_index = nghr * size
                    if rank != nghr:
                        c_peer = chunk(rank, Buffer.input, peer_index + tb)
                        c_peer.signal(nghr, Buffer.input, peer_index + tb, sendtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.wait(nghr, Buffer.input, index + tb, recvtb=tb)
                # reduce the chunks
                for i in range(size):
                    nghr = (rank + i) % size
                    if rank != nghr:
                        c.reduce(chunk(nghr, Buffer.input, index + tb), recvtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.signal(nghr, Buffer.input, index + tb, sendtb=tb)

        # wait for all the chunks is ready, then get the chunks
        for rank in range(size):
            for tb in range(size):
                for nghr in range(size):
                    if rank != nghr:
                        index = nghr * size
                        c = chunk(rank, Buffer.input, index + tb)
                        c.wait(nghr, Buffer.input, index + tb, recvtb=tb)
                for i in range(size):
                    nghr = (rank + i) % size
                    index = nghr * size
                    if rank != nghr:
                        c = chunk(rank, Buffer.input, index + tb)
                        c.get(nghr, Buffer.input, index + tb, recvtb=tb)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances)
