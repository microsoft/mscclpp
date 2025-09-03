# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def allgather_example(name, num_threads_per_block, min_message_size, max_message_size):
    gpu_size = 2
    chunksperloop = 1
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
        # Loop over each source GPU rank
        for src_rank_id in range(gpu_size):
            # Create a Rank object for the source GPU
            src_rank = Rank(src_rank_id)
            # Get the the src buffer where the data is stored
            src_buffer = src_rank.get_output_buffer()
            # Take a slice corresponding to data
            src_chunk = src_buffer[src_rank_id : src_rank_id + 1]

            # Loop over each destination GPU rank
            for dst_rank_id in range(gpu_size):
                # Create a Rank object for the destination GPU
                dst_rank = Rank(dst_rank_id)
                # Get the the dst buffer where the data will be send
                dst_input_buffer = dst_rank.get_output_buffer()
                # Take a slice corresponding where the data will be send
                dst_chunk = dst_input_buffer[src_rank_id : src_rank_id + 1]

                # Skip sending from a rank to itself
                if src_rank_id != dst_rank_id:
                    # Define a channel from src_rank → dst_rank using memory channel
                    ch = MemoryChannel(dst_rank_id, src_rank_id)
                    # Define Thread Block Group
                    tbg = ThreadBlockGroup(tb_list=[i for i in range(4)])
                    # Step 1: source signals to indicate it is ready to receive data
                    ch.signal(tb=0, relaxed=True)
                    # Step 2: wait for the destination rank to be ready
                    ch.wait(tb=0, data_sync=SyncType.after, relaxed=True)
                    # Step 3: Synchronize thread blocks and perform data transfer
                    # Barrier ensures all thread blocks in the group are synchronized before data transfer
                    src_rank.barrier(tb_list=tbg.tb_list)
                    # Transfer data from source chunk to destination chunk using the memory channel
                    # The ThreadBlockGroup (tbg) coordinates which thread blocks participate in the transfer
                    ch.put(dst_chunk, src_chunk, tb_group=tbg)
                    # Post-transfer barrier ensures all thread blocks complete the put operation
                    # before proceeding to the next step
                    src_rank.barrier(tb_list=tbg.tb_list)
                    # Step 4: source signals to indicate put is done
                    ch.signal(tb=0, data_sync=SyncType.before)
                    # Step 5: wait for receive data from destination rank
                    ch.wait(tb=0, data_sync=SyncType.after)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(args.name, args.num_threads_per_block, args.min_message_size, args.max_message_size)
