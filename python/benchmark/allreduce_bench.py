# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import cupy as cp
from test.mscclpp_group import MscclppGroup
from test.mscclpp_mpi import MpiGroup
from test.utils import KernelBuilder, pack
from mscclpp import Transport
from mpi4py import MPI
from prettytable import PrettyTable
import cupy.cuda.nccl as nccl


def human_readable_size(size, decimal_places=1):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream.ptr)
        graph = stream.end_capture()


    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niter * 1000.0



def benchmark(table: PrettyTable, niter: int, nelem: int):
    mpi_group = MpiGroup()
    group = MscclppGroup(mpi_group)
    group.barrier()

    remote_nghrs = list(range(mpi_group.comm.size))
    remote_nghrs.remove(mpi_group.comm.rank)

    # create a connection for each remote neighbor
    connections = group.make_connection(remote_nghrs, Transport.CudaIpc)
    memory = cp.zeros(nelem, dtype=cp.float32)
    type_str = ""
    if memory.dtype == cp.float16:
        type_str = "__half"
    elif memory.dtype == cp.float32:
        type_str = "float"
    elif memory.dtype == cp.int32:
        type_str = "int"
    else:
        raise RuntimeError("Unknown data type")
    memory[:] = group.my_rank+1
    cp.cuda.runtime.deviceSynchronize()

    # create a sm_channel for each remote neighbor
    sm_channels = group.make_sm_channels(memory, connections)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(file="allreduce1.cu", kernel_name="allreduce1", file_dir=file_dir, macro_dict={"TYPE": type_str}).get_compiled_kernel()
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
    memory_nbytes = memory.nbytes
    algBw = memory_nbytes / time_per_iter / 1e3
    if group.my_rank == 0:
        table.add_row([human_readable_size(memory_nbytes), "{:.2f}".format(time_per_iter), "{:.2f}".format(algBw)])
    memory[:] = group.my_rank+1
    kernel.launch_kernel(params, 24, 1024, 0, None)
    expected = 0
    for i in range(group.nranks):
        expected += (i+1)
    assert(cp.allclose(memory[0:nelem], expected))
    if group.my_rank == 0:
        print(".", end="", flush=True)
    # print(cp.nonzero(memory[0:nelem]-expected), memory[0:8])


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Create a NCCL unique ID and communicator
if rank == 0:
    uid = nccl.get_unique_id()
else:
    uid = None
uid = comm.bcast(uid, root=0)
nccl_comm = nccl.NcclCommunicator(size, uid, rank)
def nccl_benchmark(table: PrettyTable, niter: int, nelem: int):

    memory = cp.zeros(nelem, dtype=cp.float32)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            nccl_comm.allReduce(memory.data.ptr, memory.data.ptr, memory.size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, stream.ptr)
        graph = stream.end_capture()


    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    MPI.COMM_WORLD.barrier()
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    time_per_iter = cp.cuda.get_elapsed_time(start, end) / niter * 1000.0
    memory_nbytes = memory.nbytes
    algBw = memory_nbytes / time_per_iter / 1e3
    if rank == 0:
        table.add_row([human_readable_size(memory_nbytes), "{:.2f}".format(time_per_iter), "{:.2f}".format(algBw)])
        print(".", end="", flush=True)



if __name__ == "__main__":

    # Create a table
    table = PrettyTable()

    if MPI.COMM_WORLD.rank == 0:
        # Set table headers
        table.field_names = ["Size", "Time (us)", "AlgBW (GB/s)"]

    for i in range(10,28):
        # nccl_benchmark(table, 1000, 2**i)
        benchmark(table, 1000, 2**i)

    if MPI.COMM_WORLD.rank == 0:
        print()
        print(table)
