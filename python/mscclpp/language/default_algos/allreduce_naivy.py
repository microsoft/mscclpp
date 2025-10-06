# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from mscclpp.language.internal.types import AlgoSpec
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allreduce_naivy(spec: AlgoSpec) -> CollectiveProgram:
    chunksperloop = 1
    gpu_size = spec.nranks_per_node
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        spec.name,
        collective,
        gpu_size,
        protocol="LL",
        num_threads_per_block=spec.num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=spec.min_message_size,
        max_message_size=spec.max_message_size,
    ) as prog:
        # Creating Scratch Buffers
        scratch_buffer = []
        for gpu in range(gpu_size):
            scratch_buffer.append(Buffer(gpu, 2 * gpu_size))

        # Creating Channels
        channels = {}
        for gpu in range(gpu_size):
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)] = MemoryChannel(peer, gpu)

        # Each rank sends the nth chunk to the nth rank into scratch space
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)].put_packets(
                        scratch_buffer[peer][gpu : gpu + 1], input_buffer[peer : peer + 1], 0
                    )

        # Each rank performs a local reduction on the nth chunk
        for gpu in range(gpu_size):
            chunks = []
            for peer in range(gpu_size):
                if peer != gpu:
                    chunks.append(scratch_buffer[gpu][peer : peer + 1])
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            rank.reduce(input_buffer[gpu : gpu + 1], chunks, 0, packet=True)
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)].put_packets(
                        scratch_buffer[peer][gpu_size + gpu : gpu_size + gpu + 1], input_buffer[gpu : gpu + 1], 0
                    )

        # Each rank get final result from scratch space
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            for peer in range(gpu_size):
                if peer != gpu:
                    rank.unpack_packets(
                        input_buffer[peer : peer + 1], scratch_buffer[gpu][gpu_size + peer : gpu_size + peer + 1], 0
                    )

    return prog
