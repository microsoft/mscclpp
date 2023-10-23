# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import cupy as cp
from .mscclpp_op import MscclppOp
from .nccl_op import NcclOp
from mpi4py import MPI
from prettytable import PrettyTable


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



def run_benchmark(mscclpp_op: MscclppOp, nccl_op: NcclOp, table: PrettyTable, niter: int, nelem: int):
    memory = cp.zeros(nelem, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    mscclpp_call = mscclpp_op.make_callback1(memory)
    nccl_call = nccl_op.make_callback(memory)

    memory_nbytes = memory.nbytes
    mscclpp_time = bench_time(niter, mscclpp_call)
    mscclpp_algBw = memory_nbytes / mscclpp_time / 1e3
    nccl_time = bench_time(niter, nccl_call)
    nccl_algBw = memory_nbytes / nccl_time / 1e3
    if MPI.COMM_WORLD.rank == 0:
        table.add_row([human_readable_size(memory_nbytes), "{:.2f}".format(mscclpp_time), "{:.2f}".format(mscclpp_algBw), "{:.2f}".format(nccl_time), "{:.2f}".format(nccl_algBw), "{:.2f}".format(nccl_time / mscclpp_time)])
    # memory[:] = group.my_rank+1
    # kernel.launch_kernel(params, 24, 1024, 0, None)
    # expected = 0
    # for i in range(group.nranks):
    #     expected += (i+1)
    # assert(cp.allclose(memory[0:nelem], expected))
    if MPI.COMM_WORLD.rank == 0:
        print(".", end="", flush=True)
    # print(cp.nonzero(memory[0:nelem]-expected), memory[0:8])

if __name__ == "__main__":
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    N_GPUS_PER_NODE = shm_comm.size
    shm_comm.Free()
    cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

    # create a NcclOp and MscclppOp
    mscclpp_op = MscclppOp()
    nccl_op = NcclOp()

    table = None
    if MPI.COMM_WORLD.rank == 0:
        # Set table headers
        table = PrettyTable()
        table.field_names = ["Size", "Time (us)", "AlgBW (GB/s)", "NCCL Time (us)", "NCCL AlgBW (GB/s)", "Speed Up"]

    for i in range(10,28):
        run_benchmark(mscclpp_op, nccl_op, table, 100, 2**i)

    if MPI.COMM_WORLD.rank == 0:
        print()
        print(table)
    mscclpp_op = None
    nccl_op = None
