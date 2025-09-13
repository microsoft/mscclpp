# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch, torch.distributed as dist
from mscclpp import jit
from mscclpp.language.collectives import AllReduce
from mscclpp.language.channel import SwitchChannel, MemoryChannel, BufferType, SyncType
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Rank
from mscclpp.plan import Registry, Request
import mscclpp.plan as plan

def allreduce_nvls(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    chunksperloop = 1
    collective = AllReduce(gpu_size, chunksperloop, True)
    with CollectiveProgram (
        name,
        collective,
        gpu_size,
        instances=8,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
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

def setup_plan(rank: int, world_size: int):
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
        min_msg_size=1<<20,
        max_msg_size=48<<30,
        tags={"nvls"},
    )
    Registry.register(plan_handle)

def selector(plans: dict, req: Request):
    collective_plans = plans.get(req.collective)
    nvls = [p for p in collective_plans if "nvls" in p.tags]
    return nvls[0] if nvls else collective_plans[0]

def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    setup_plan(rank, world)
    plan.set_selector(selector)
    dist.init_process_group(backend="nccl")
    return rank, world, local

def main():
    _, _, local = init_dist()
    torch.device("cuda", local)
    x = torch.randn(12<<20, dtype=torch.float16, device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
