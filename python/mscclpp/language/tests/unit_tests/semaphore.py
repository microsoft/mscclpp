# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Semaphore Operation Test

This file demonstrates the use of semaphore operations in MSCCLPP.
The semaphore operations provide asynchronus synchronization mechanisms between GPUs.

WARNING: This algorithm is designed solely for demonstrating the use of a single
operation (semaphore) and is NOT intended for production use. This test
may not work correctly in the MSCCLPP executor.
"""

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def semaphore_test(num_threads_per_block, min_message_size, max_message_size):
    # Set up 1 GPUs for semaphore synchronization
    gpus = 1
    collective = TestCollective(gpus, 0, 0)

    with CollectiveProgram(
        "semaphore_test",
        collective,
        gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create semaphore for inter-GPU synchronization
        sm = Semaphore(rank=0, initial_value=0)

        # Acquire semaphore (blocks until available)
        sm.acquire(tb=0, data_sync=SyncType.after)

        # Release semaphore (allows other GPUs to proceed)
        sm.release(tb=1, data_sync=SyncType.before)

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

semaphore_test(args.num_threads_per_block, args.min_message_size, args.max_message_size)
