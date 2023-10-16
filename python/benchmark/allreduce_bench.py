# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from test.mscclpp_group import MscclppGroup
import cupy as cp
from test.mscclpp_mpi import MpiGroup
from test.utils import KernelBuilder, pack
from mscclpp import Transport
from mpi4py import MPI
from prettytable import PrettyTable


def human_readable_size(size, decimal_places=1):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def benchmark(table: PrettyTable, niter: int, nelem: int):
    mpi_group = MpiGroup()
    group = MscclppGroup(mpi_group)
    group.barrier()

    remote_nghrs = list(range(mpi_group.comm.size))
    remote_nghrs.remove(mpi_group.comm.rank)

    # create a connection for each remote neighbor
    connections = group.make_connection(remote_nghrs, Transport.CudaIpc)
    memory = cp.zeros(2*nelem, dtype=cp.int32)
    memory[:] = group.my_rank+1
    cp.cuda.runtime.deviceSynchronize()

    # create a sm_channel for each remote neighbor
    sm_channels = group.make_sm_channels(memory, connections)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(file="allreduce1.cu", kernel_name="allreduce1", file_dir=file_dir).get_compiled_kernel()
    params = b""
    device_handles = []
    for rank in range(group.nranks):
        if rank != group.my_rank:
            device_handles.append(sm_channels[rank].device_handle().raw)
    params += pack(cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8), memory, group.my_rank, group.nranks, nelem)


    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            kernel.launch_kernel(params, 24, 1024, 0, stream.ptr)
        graph = stream.end_capture()


    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    group.barrier()
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    time_per_iter = cp.cuda.get_elapsed_time(start, end) / niter * 1000.0
    memory_nbytes = memory.nbytes // 2
    algBw = memory_nbytes / time_per_iter / 1e3
    if group.my_rank == 0:
        table.add_row([human_readable_size(memory_nbytes), "{:.2f}".format(time_per_iter), "{:.2f}".format(algBw)])
    # memory[:] = group.my_rank+1
    # for i in range(1):
    #     kernel.launch_kernel(params, 24, 1024, 0, None)
    # print(cp.nonzero(memory[0:nelem]-36))


if __name__ == "__main__":

    # Create a table
    table = PrettyTable()

    if MPI.COMM_WORLD.rank == 0:
        # Set table headers
        table.field_names = ["Size", "Time (us)", "AlgBW (GB/s)"]

    for i in range(10,28):
        benchmark(table, 1000, 2**i)

    if MPI.COMM_WORLD.rank == 0:
        print(table)
