# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


def allreduce(gpus, instances, protocol):
    size = gpus
    chunksperloop = gpus * (gpus - 1)
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLPPProgram(
        "allreduce_mi300",
        topology,
        collective,
        instances,
        protocol=protocol,
    ):
        for rank in range(size):
            for tb in range(size - 1):
                for i in range(size):
                    nghr = (rank + tb + 1 + i) % size
                    if rank == nghr:
                        continue
                    c = chunk(rank, Buffer.input, nghr * (size - 1) + tb)
                    c.put(nghr, "scratch", rank * (size - 1) + tb, sendtb=tb)
        for rank in range(size):
            for tb in range(size - 1):
                for i in range(size):
                    nghr = (rank + tb + 1 + i) % size
                    if rank == nghr:
                        continue
                    c = chunk(rank, Buffer.input, nghr * (size - 1) + tb)
                    c.signal(nghr, "scratch", rank * (size - 1) + tb, sendtb=tb)
                for i in range(size):
                    nghr = (rank + tb + 1 + i) % size
                    if rank == nghr:
                        continue
                    c = chunk(rank, "scratch", nghr * (size - 1) + tb)
                    c.wait(nghr, Buffer.input, rank * (size - 1) + tb, recvtb=tb)

        for rank in range(size):
            for tb in range(size - 1):
                c = chunk(rank, Buffer.input, rank * (size - 1) + tb)
                for nghr in range(size):
                    if rank != nghr:
                        index = nghr * (size - 1)
                        c.reduce(chunk(rank, "scratch", index + tb), recvtb=tb)
                for i in range(size):
                    nghr = (rank + i) % size
                    index = rank * (size-1)
                    if rank != nghr:
                        c = chunk(rank, Buffer.input, index + tb)
                        c.put(nghr, Buffer.input, index + tb, sendtb=tb)

        for rank in range(size):
            for tb in range(size - 1):
                for i in range(size):
                    nghr = (rank + i) % size
                    index = rank * (size - 1)
                    if rank == nghr:
                        continue
                    c = chunk(rank, Buffer.input, index + tb)
                    c.signal(nghr, Buffer.input, index + tb, sendtb=tb)
                for i in range(size):
                    nghr = (rank + i) % size
                    if rank == nghr:
                        continue
                    c = chunk(rank, Buffer.input, nghr * (size - 1) + tb)
                    c.wait(nghr, Buffer.input, nghr * (size - 1) + tb, recvtb=tb)

        Json()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")
parser.add_argument("--protocol", type=str, default="Simple", choices=["Simple"], help="Protocol")

args = parser.parse_args()

allreduce(args.num_gpus, args.instances, args.protocol)
