# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import cupy as cp
from .mscclpp_op import MscclppOp
from .nccl_op import NcclOp
from mpi4py import MPI
from prettytable import PrettyTable

data_type = cp.float32

def human_readable_size(size, decimal_places=1):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def check_correctness(memory, func):
    rand_gen = cp.random.default_rng(seed=MPI.COMM_WORLD.rank)
    memory[:] = rand_gen.random(memory.shape)
    cp.cuda.runtime.deviceSynchronize()
    output_memory = func(0)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.zeros_like(memory)
    for i in range(MPI.COMM_WORLD.size):
        rand_gen = cp.random.default_rng(seed=i)
        expected += rand_gen.random(memory.shape)
    return cp.allclose(output_memory, expected)

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
    memory = cp.zeros(nelem, dtype=data_type)
    cp.cuda.runtime.deviceSynchronize()

    if memory.nbytes < 2**21:
        mscclpp_call = mscclpp_op.make_callback2(memory)
    else:
        mscclpp_call = mscclpp_op.make_callback1(memory)
    nccl_call = nccl_op.make_callback(memory)

    memory_nbytes = memory.nbytes
    mscclpp_time = bench_time(niter, mscclpp_call)
    mscclpp_algBw = memory_nbytes / mscclpp_time / 1e3
    mscclpp_check = "PASS" if check_correctness(memory, mscclpp_call) else "FAIL"

    nccl_time = bench_time(niter, nccl_call)
    nccl_algBw = memory_nbytes / nccl_time / 1e3
    nccl_check = "PASS" if check_correctness(memory, nccl_call) else "FAIL"

    if MPI.COMM_WORLD.rank == 0:
        table.add_row([human_readable_size(memory_nbytes), "{:.2f}".format(mscclpp_time), "{:.2f}".format(mscclpp_algBw), mscclpp_check, "{:.2f}".format(nccl_time), "{:.2f}".format(nccl_algBw), nccl_check, "{:.2f}".format(nccl_time / mscclpp_time)])
    if MPI.COMM_WORLD.rank == 0:
        print(".", end="", flush=True)

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
        table.field_names = ["Size", "Time (us)", "AlgBW (GB/s)", "Correctness", "NCCL Time (us)", "NCCL AlgBW (GB/s)", "NCCL Correctness", "Speed Up"]

    for i in range(10,30):
        run_benchmark(mscclpp_op, nccl_op, table, 10, 3*2**i)

    if MPI.COMM_WORLD.rank == 0:
        print()
        print(table)
    mscclpp_op = None
    nccl_op = None
