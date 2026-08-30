# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Multi-Node Transfer Test

This file tests the SIGNAL, WAIT, PUT, PUT_WITH_SIGNAL and
PUT_WITH_SIGNAL_AND_FLUSH operations on PortChannels in a multi-node
environment. It implements a 2-GPU allgather using the Simple protocol,
exercising the different port-channel synchronization primitives.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def multi_node_transfer(name, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 2
    gpu_size = 2
    collective = AllGather(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Setup ranks, channels, output and scratch buffers for 2-GPU allgather
        first_rank = Rank(0)
        second_rank = Rank(1)
        first_ch1 = PortChannel(1, 0)
        second_ch1 = PortChannel(0, 1)
        first_ch2 = PortChannel(1, 0)
        second_ch2 = PortChannel(0, 1)
        first_output_buffer = first_rank.get_output_buffer()
        second_output_buffer = second_rank.get_output_buffer()

        # Initial handshake on both port channels: peers exchange SIGNAL/WAIT to
        # ensure remote buffers are ready before any data transfer begins.
        first_ch1.signal(tb=0)
        second_ch1.signal(tb=0)
        first_ch1.wait(tb=0)
        second_ch1.wait(tb=0)
        first_ch2.signal(tb=1)
        second_ch2.signal(tb=1)
        first_ch2.wait(tb=1)
        second_ch2.wait(tb=1)

        # Rank 0 -> rank 1 via ch1: PUT followed by an explicit SIGNAL and FLUSH
        first_ch1.put(second_output_buffer[0:1], first_output_buffer[0:1], tb=0)
        first_ch1.signal(tb=0)
        first_ch1.flush(tb=0)
        # Rank 0 -> rank 1 via ch2: PUT_WITH_SIGNAL fuses the data transfer with
        # the completion signal, followed by a separate FLUSH
        first_ch2.put_with_signal(second_output_buffer[1:2], first_output_buffer[1:2], tb=1)
        first_ch2.flush(tb=1)
        # Rank 1 -> rank 0 via ch1: PUT_WITH_SIGNAL_AND_FLUSH fuses PUT, SIGNAL
        # and FLUSH into a single operation
        second_ch1.put_with_signal_and_flush(first_output_buffer[2:4], second_output_buffer[2:4], tb=0)

        # Final WAITs ensure all incoming transfers have completed on each rank
        first_ch1.wait(tb=0)
        second_ch1.wait(tb=0)
        second_ch2.wait(tb=1)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

multi_node_transfer(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
