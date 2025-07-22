# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.pipeline import *


def allreduce_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    name = "allreduce"
    gpu_size = 2
    collective = AllReduce(gpu_size, 1, True)
    with MSCCLPPProgram(
        name,
        collective,
        gpu_size,
        instances=2,
        protocol="Simple",
        instr_fusion=True,
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        nvls_chan = SwitchChannel(rank_list=[0, 1], buffer_type=BufferType.scratch)
        nranks = 2
        for i in range(nranks):
            src_rank = i
            dst_rank = (i + 1) % nranks
            chan = MemoryChannel(dst_rank, src_rank)
            chan1 = MemoryChannel(dst_rank, src_rank)
            rank = Rank(i)
            sem0 = Semaphore(rank=i, initial_value=0)
            sem1 = Semaphore(rank=i, initial_value=0)
            input_buffer = rank.get_input_buffer()
            output_buffer = rank.get_output_buffer()
            scratch_buffer = Buffer(i, nranks)
            with LoopIterationContext(unit=2**20, num_chunks=1):
                # copy data to scratch buffer
                for offset in range(nranks):
                    dst_chunk = scratch_buffer[offset : offset + 1]
                    src_chunk = input_buffer[offset : offset + 1]
                    rank.copy(dst_chunk, src_chunk, tb=0)
                chan.signal(tb=0, data_sync=SyncType.before)
                chan.wait(tb=0, data_sync=SyncType.after)
                sem0.release(tb=0)

                # do allreduce in scratch buffer
                sem0.acquire(tb=1, data_sync=SyncType.after)
                nvls_chan.at_rank(src_rank).reduce(buffer_offset=i, size=1, dst_chunk=scratch_buffer[i : i + 1], tb=1)
                nvls_chan.at_rank(src_rank).broadcast(
                    src_chunk=scratch_buffer[i : i + 1], buffer_offset=i, size=1, tb=1
                )
                chan1.signal(tb=1, data_sync=SyncType.before)
                sem1.release(tb=1)

                # copy data back to output buffer
                sem1.acquire(tb=2)
                chan1.wait(tb=2, data_sync=SyncType.after)
                for index in range(nranks):
                    dst_chunk = output_buffer[index : index + 1]
                    src_chunk = scratch_buffer[index : index + 1]
                    rank.copy(dst_chunk, src_chunk, tb=2)

            print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allreduce_example(args.name, args.num_gpus, args.num_threads_per_block, args.min_message_size, args.max_message_size)
