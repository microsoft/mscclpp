# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.buffer import Buffer
from mscclpp.language.collectives import SendRecv
from mscclpp.language.types import ChannelType


def send_recv(instances):
    """
    Send and receive data between two ranks using port channels.
    steps:
    1. Each rank sends a chunk to the other rank's scratch buffer and signals the other rank that the data has been sent.
    2. Wait for the data to be received then copy it to the output buffer.
    """
    size = 2
    chunksperloop = 1
    collective = SendRecv(size, chunksperloop, False)
    with MSCCLPPProgram(
        "send_recv",
        collective,
        size,
        instances,
    ):
        for r in range(size):
            for nghr in range(size):
                if nghr == r:
                    continue
                c = chunk(r, Buffer.input, 0)
                c.put(
                    nghr,
                    Buffer.scratch,
                    1,
                    sendtb=0,
                    chan_type=ChannelType.port,
                )
                c.signal(nghr, Buffer.scratch, 1, sendtb=0, chan_type=ChannelType.port)
                c.flush(nghr, Buffer.scratch, 1, sendtb=0, chan_type=ChannelType.port)

        for r in range(size):
            c = chunk(r, Buffer.scratch, 1)
            c.wait(1 - r, Buffer.input, 0, recvtb=0, chan_type=ChannelType.port)
            c.copy(r, Buffer.output, 0, sendtb=0)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

send_recv(args.instances)
