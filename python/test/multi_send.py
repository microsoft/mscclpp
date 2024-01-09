import cupy as cp
from mscclpp import (
    ProxyService,
)
import mscclpp.comm as mscclpp_comm
from .mscclpp_mpi import MpiGroup
from mscclpp.utils import KernelBuilder, pack
import struct
import os
from typing import List

def create_proxy_channels(proxy_service: ProxyService, group: mscclpp_comm.CommGroup, 
                          nchannels: int, memory: List[cp.ndarray]):
    remote = 0 if group.my_rank == 1 else 1
    assert len(memory) == nchannels
    tran = group.my_ib_device(group.my_rank % 8)
    connections = group.make_connection([remote], tran)
    channels = []
    for channel in range(nchannels):
        channels.append(group.make_proxy_channels(proxy_service, memory[channel], connections)[remote])
    return channels

def main(group: mscclpp_comm.CommGroup):
    nelem = 4
    nchannels = 256

    memory = [cp.zeros(nelem, dtype=cp.int32) for _ in range(nchannels)]
    memory_expected = [cp.zeros_like(memory[i]) for i in range(nchannels)]
    nelemPerRank = nelem // group.nranks
    sizePerRank = nelemPerRank * memory[0].itemsize
    offset = sizePerRank * group.my_rank

    for channel in range(nchannels):
        memory[channel][(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = nchannels * group.my_rank + channel + 1
    for rank in range(group.nranks):
        for channel in range(nchannels):
            memory_expected[channel][(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = nchannels * rank + channel + 1
    group.barrier()

    proxy_service = ProxyService()
    channels = create_proxy_channels(proxy_service, group, nchannels, memory)
    handles = [ch.device_handle().raw for ch in channels]
    channel_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

    # params = b"" + pack(channel_mem, offset, sizePerRank, len(handles), group.my_rank)
    params = b"" + struct.pack("P", channel_mem.data.ptr) + struct.pack("i", len(handles))

    file_dir = os.path.dirname(os.path.abspath(__file__))
    send_kernel = KernelBuilder(
        file="nw_out_test.cu",
        kernel_name="nw_out_put_kernel",
        file_dir=file_dir,
    ).get_compiled_kernel()
    recv_kernel = KernelBuilder(
        file="nw_in_test.cu",
        kernel_name="nw_in_wait_kernel",
        file_dir=file_dir,
    ).get_compiled_kernel()

    proxy_service.start_proxy()
    group.barrier()

    nblocks = 1
    nthreads = 1
    if group.my_rank == 0:
        send_kernel.launch_kernel(params, nblocks, nthreads, 0, None)
    else:
        recv_kernel.launch_kernel(params, nblocks, nthreads, 0, None)
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()
    proxy_service.stop_proxy()
    for c in range(nchannels):
        if group.my_rank == 1:
            assert cp.array_equal(memory[c], memory_expected[c])

if __name__ == "__main__":
    mpi_group = MpiGroup([0, 1])
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    # os.environ['MSCCLPP_HCA_DEVICES'] = 'mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8'

    main(group)

    del group