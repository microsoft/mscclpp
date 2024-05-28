# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from os import path
from mscclpp import (
    DataType,
    Executor,
    ExecutionPlan,
)
import mscclpp.comm as mscclpp_comm

import cupy as cp
from mpi4py import MPI

MSCCLPP_ROOT_PATH = "/root/mscclpp"


def bench_time(niters: int, ngraphIters: int, func):
    # capture cuda graph for niters of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niters):
            func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(ngraphIters):
        graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niters * 1000.0 / ngraphIters


if __name__ == "__main__":
    mscclpp_group = mscclpp_comm.CommGroup(MPI.COMM_WORLD)
    cp.cuda.Device(MPI.COMM_WORLD.rank % mscclpp_group.nranks_per_node).use()
    executor = Executor(mscclpp_group.communicator)
    execution_plan = ExecutionPlan(
        "allreduce_mi300", path.join(MSCCLPP_ROOT_PATH, "test", "execution-files", "allreduce5.json")
    )

    nelems = 1024 * 1024 * 200
    cp.random.seed(42)
    buffer = cp.random.random(nelems).astype(cp.float16)
    sub_arrays = cp.split(buffer, MPI.COMM_WORLD.size)
    sendbuf = sub_arrays[MPI.COMM_WORLD.rank]
    expected = cp.zeros_like(sendbuf)
    for i in range(mscclpp_group.nranks):
        expected += sub_arrays[i]
    mscclpp_group.barrier()

    executor_func = lambda stream: executor.execute(
        MPI.COMM_WORLD.rank,
        sendbuf.data.ptr,
        sendbuf.data.ptr,
        sendbuf.nbytes,
        sendbuf.nbytes,
        DataType.float16,
        512,
        execution_plan,
        stream.ptr,
    )
    # check correctness
    stream = cp.cuda.Stream(non_blocking=True)
    executor_func(stream)
    stream.synchronize()
    assert cp.allclose(sendbuf, expected, atol=1e-2 * mscclpp_group.nranks)

    execution_time = bench_time(
        100,
        10,
        executor_func
    )
    print(f"Rank: {MPI.COMM_WORLD.rank} Execution time: {execution_time} us, data size: {sendbuf.nbytes} bytes")
    executor = None
    mscclpp_group = None
