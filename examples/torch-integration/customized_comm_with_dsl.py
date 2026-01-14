# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm_with_dsl.py

import os
import torch
import mscclpp.comm as mscclpp_comm
import mscclpp
from mscclpp.language.collectives import AllReduce
from mscclpp.language.channel import SwitchChannel, MemoryChannel, BufferType, SyncType
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Rank
import netifaces as ni
import ipaddress


def allreduce_nvls(spec: mscclpp.AlgoSpec) -> CollectiveProgram:
    gpu_size = spec.world_size
    with CollectiveProgram(
        spec.name,
        spec.collective,
        gpu_size,
        instances=spec.instances,
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


def setup_plan(rank: int, world_size: int, nranks_per_node: int):
    spec = mscclpp.AlgoSpec(
        name="allreduce_nvls",
        collective=AllReduce(world_size, 1, True),
        nranks_per_node=nranks_per_node,
        world_size=world_size,
        in_place=True,
        instances=nranks_per_node,
        protocol="Simple",
        num_threads_per_block=1024,
        min_message_size=1 << 20,
        max_message_size=48 << 30,
        tags={"nvls": 1},
    )

    algorithms = []
    algo = mscclpp.compile(algo=allreduce_nvls, algo_spec=spec, rank=rank)
    algorithms.append(algo)
    return algorithms


def interfaces_for_ip_netifaces(ip: str):
    target = ipaddress.ip_address(ip)
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if "addr" in link:
                    addr = ipaddress.ip_address(link["addr"])
                    if addr == target:
                        return interface
    return None


def dtype_to_mscclpp_dtype(dtype: torch.dtype) -> mscclpp.DataType:
    if dtype == torch.float16:
        return mscclpp.DataType.float16
    elif dtype == torch.float32:
        return mscclpp.DataType.float32
    elif dtype == torch.int32:
        return mscclpp.DataType.int32
    elif dtype == torch.bfloat16:
        return mscclpp.DataType.bfloat16
    else:
        raise ValueError(f"Unknown data type: {dtype}")


class CustomizedComm:
    def __init__(self, comm: mscclpp.CommGroup, algorithms=[]):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        self.executor = mscclpp.Executor(comm.communicator)
        self.algorithms = algorithms

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream: torch.cuda.Stream = None):
        assert op == torch.distributed.ReduceOp.SUM
        algo: mscclpp.Algorithm = self.algorithms[0]
        algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=dtype_to_mscclpp_dtype(tensor.dtype),
            stream=stream.cuda_stream if stream is not None else 0,
        )

    def barrier_cpu(self):
        self.comm.barrier()

    def destroy(self):
        self.algorithms = None
        self.executor = None
        self.comm = None


def init_dist() -> CustomizedComm:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    nranks_per_node = int(torch.cuda.device_count())
    algorithms = setup_plan(rank, world, nranks_per_node)
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    mscclpp_group = mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world)
    return CustomizedComm(mscclpp_group, algorithms)


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    comm = init_dist()
    comm.barrier_cpu()
    buffer = mscclpp.RawGpuBuffer(24 << 20)
    dlpack = buffer.to_dlpack(data_type=str(torch.bfloat16))
    x = torch.utils.dlpack.from_dlpack(dlpack)
    x.normal_()
    comm.all_reduce(x, op=torch.distributed.ReduceOp.SUM, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    comm.barrier_cpu()
    print(f"Rank {comm.rank} allreduce completed successfully.")
    comm.destroy()


if __name__ == "__main__":
    main()
