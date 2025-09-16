# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# LD_PRELOAD=<MSCCLPP_REPO>/build/apps/nccl/libmscclpp_nccl.so  torchrun --nnodes=1 --nproc_per_node=8 dsl-torch-integration/dsl_torch_integration.py

import os
import torch, torch.distributed as dist
from mscclpp import jit, RawGpuBuffer
from mscclpp.jit import AlgoSpec
from mscclpp.language.collectives import AllReduce
from mscclpp.language.channel import SwitchChannel, MemoryChannel, BufferType, SyncType
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Rank
from mscclpp.plans import ExecutionPlanRegistry


def allreduce_nvls(spec: AlgoSpec) -> CollectiveProgram:
    chunksperloop = 1
    gpu_size = spec.world_size
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram(
        spec.name,
        collective,
        gpu_size,
        instances=8,
        protocol=spec.protocol,
        num_threads_per_block=spec.num_threads_per_block,
        min_message_size=spec.min_message_size,
        max_message_size=spec.max_message_size,
    ) as program:
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


def setup_plan(registry: ExecutionPlanRegistry, rank: int, world_size: int):
    plan_handle = jit.compile(
        algo=allreduce_nvls,
        name="allreduce_nvls",
        collective="allreduce",
        rank=rank,
        nranks_per_node=8,
        world_size=world_size,
        instances=2,
        protocol="Simple",
        num_threads_per_block=1024,
        min_message_size=1 << 20,
        max_message_size=48 << 30,
        tags={"nvls"},
    )
    registry.register_plan(plan_handle)


def selector(plans, req):
    if req.collective != "allreduce":
        return None
    if req.message_size < 1 << 20:
        return None
    nvls = [p for p in plans if "nvls" in p.tags]
    return nvls[0] if nvls else plans[0]


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    registry = ExecutionPlanRegistry()
    setup_plan(registry, rank, world)
    registry.set_selector(selector)
    dist.init_process_group(backend="nccl")
    return rank, world, local


def main():
    _, _, local = init_dist()
    torch.cuda.set_device(local)
    buffer = RawGpuBuffer(24 << 20)
    dlpack = buffer.to_dlpack(dataType=str(torch.bfloat16))
    x = torch.utils.dlpack.from_dlpack(dlpack)
    x.normal_()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
