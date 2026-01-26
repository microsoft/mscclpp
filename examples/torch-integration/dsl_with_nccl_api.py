# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# LD_PRELOAD=<MSCCLPP_REPO>/build/lib/nccl/libmscclpp_nccl.so  torchrun --nnodes=1 --nproc_per_node=8 dsl_with_nccl_api.py

import os
from typing import Any, Dict
import torch, torch.distributed as dist
import mscclpp
from mscclpp.language.collectives import AllReduce
from mscclpp.language.channel import SwitchChannel, MemoryChannel, BufferType, SyncType
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Rank


def allreduce_nvls(spec: mscclpp.AlgoSpec) -> CollectiveProgram:
    gpu_size = spec.world_size
    with CollectiveProgram.from_spec(spec) as program:
        # Creating Channels
        nvls_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.input)
        channels = {}
        for gpu in range(gpu_size):
            for peer in range(gpu_size):
                if peer != gpu:
                    channels[(peer, gpu)] = MemoryChannel(peer, gpu)

        # Synchronization to Ensure all the Gpus are Ready
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True, data_sync=SyncType.after)

        # Reducing and Storing the data
        for gpu in range(gpu_size):
            buffer_offset = gpu
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            nvls_chan.at_rank(gpu).reduce(
                buffer_offset=buffer_offset, size=1, dst_chunk=input_buffer[gpu : gpu + 1], tb=0
            )
            nvls_chan.at_rank(gpu).broadcast(
                src_chunk=input_buffer[gpu : gpu + 1], buffer_offset=buffer_offset, size=1, tb=0
            )

        # Synchronization to Ensure the Gpus finished
        for gpu in range(gpu_size):
            src_rank = gpu
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].signal(tb=0, relaxed=True, data_sync=SyncType.before)
            for peer in range(gpu_size):
                if peer != src_rank:
                    dst_rank = peer
                    channels[(dst_rank, src_rank)].wait(tb=0, relaxed=True)

    return program


def setup_plan(algo_collection_builder: mscclpp.AlgorithmCollectionBuilder, rank: int, world_size: int):
    spec = mscclpp.AlgoSpec(
        name="allreduce_nvls",
        collective=AllReduce(8, 1, True),
        nranks_per_node=8,
        world_size=world_size,
        in_place=True,
        instances=2,
        protocol="Simple",
        num_threads_per_block=1024,
        min_message_size=1 << 20,
        max_message_size=48 << 30,
        tags={"nvls": 1},
    )

    algo = mscclpp.compile(algo=allreduce_nvls, algo_spec=spec, rank=rank)
    algo_collection_builder.add_algorithm_builder(algo)


def selector(algorithms: Dict[str, Any], req):
    if req.collective != "allreduce":
        return None
    if req.message_size < 1 << 20:
        return None
    return algorithms["allreduce"]["allreduce_nvls"]


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    algorithm_collection_builder = mscclpp.AlgorithmCollectionBuilder()
    setup_plan(algorithm_collection_builder, rank, world)
    algorithm_collection_builder.set_algorithm_selector(selector)
    dist.init_process_group(backend="nccl", device_id=local)
    return rank, world, local


def main():
    _, _, local = init_dist()
    torch.cuda.set_device(local)
    buffer = mscclpp.RawGpuBuffer(24 << 20)
    dlpack = buffer.to_dlpack(data_type=str(torch.bfloat16))
    x = torch.utils.dlpack.from_dlpack(dlpack)
    x.normal_()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {local} allreduce completed successfully.")


if __name__ == "__main__":
    main()
