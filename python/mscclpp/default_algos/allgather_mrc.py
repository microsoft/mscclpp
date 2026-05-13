# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
from mscclpp.language.loop import LoopIterationContext
from mscclpp.language.utils import AlgoSpec


def allgather_mrc(spec: AlgoSpec) -> CollectiveProgram:
   size = spec.world_size
   
   with CollectiveProgram.from_spec(spec) as prog:
        # Port channels for inter-node communication
        port_channels = {}
        for n in range(size):
            src_rank = n
            next_rank = (n + 1) % size
            prev_rank = (n - 1) % size
            if src_rank != next_rank:
                port_channels[next_rank, src_rank] = PortChannel(next_rank, src_rank)
            if src_rank != prev_rank:
                port_channels[prev_rank, src_rank] = PortChannel(prev_rank, src_rank)

        # ===== Inter-node ring send + Intra-node AllGather =====
        for step in range(size - 1):
            for n in range(size):
                src_rank = n
                next_rank = (n + 1) % size
                prev_rank = (n - 1) % size
                offset = (n - step) % size
                recv_offset = (n - 1 - step) % size

                # Sending to the next node
                ch_to_next = port_channels[next_rank, src_rank]
                ch_from_prev = port_channels[prev_rank, src_rank]

                src_chunk = Rank(src_rank).get_output_buffer()[offset:offset + 1]
                dst_chunk_next = Rank(next_rank).get_output_buffer()[offset:offset + 1]

                if step == 0:
                    # Signal prev_rank that data is ready for it to read
                    ch_from_prev.signal(tb=0)
                    # Wait for signal from next_rank
                    ch_to_next.wait(tb=0)

                ch_to_next.put(dst_chunk_next, src_chunk, tb=0)
                ch_to_next.signal(tb=0)

                # Receiving from the previous node and sharing the data inside the node
                recv_src_chunk = Rank(src_rank).get_output_buffer()[recv_offset:recv_offset + 1]
                ch_from_prev.wait(tb=0)

        return prog

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of the program")
    parser.add_argument("--num_gpus", type=int, help="total number of gpus")
    parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
    parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
    parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
    parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

    args = parser.parse_args()

    spec = AlgoSpec(
        name=args.name,
        collective=AllGather(args.num_gpus, 1, True),
        nranks_per_node=args.gpus_per_node,
        world_size=args.num_gpus,
        in_place=True,
        instances=1,
        protocol="Simple",
        auto_sync=True,
        num_threads_per_block=args.num_threads_per_block,
        reuse_resources=False,
        use_double_scratch_buffer=False,
        min_message_size=args.min_message_size,
        max_message_size=args.max_message_size,
    )

    prog = allgather_mrc(spec)
    print(prog.to_json())