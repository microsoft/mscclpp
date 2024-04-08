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


def bench_time(niters: int, func):
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
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niters * 1000.0


if __name__ == "__main__":
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    N_GPUS_PER_NODE = shm_comm.size
    shm_comm.Free()

    cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()
    mscclpp_group = mscclpp_comm.CommGroup(MPI.COMM_WORLD)
    executor = Executor(mscclpp_group.communicator, N_GPUS_PER_NODE)
    execution_plan = ExecutionPlan(
        "allreduce_pairs", path.join(MSCCLPP_ROOT_PATH, "test", "execution-files", "allreduce.json")
    )

    nelems = 1024 * 1024
    cp.random.seed(42)
    buffer = cp.random.random(nelems).astype(cp.float16)
    sub_arrays = cp.split(buffer, MPI.COMM_WORLD.size)
    sendbuf = sub_arrays[MPI.COMM_WORLD.rank]

    expected = cp.zeros_like(sendbuf)
    for i in range(MPI.COMM_WORLD.size):
        expected += sub_arrays[i]

    stream = cp.cuda.Stream(non_blocking=True)
    executor.execute(
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
    stream.synchronize()
    assert cp.allclose(sendbuf, expected, atol=1e-3 * MPI.COMM_WORLD.size)

    execution_time = bench_time(
        1000,
        lambda stream: executor.execute(
            MPI.COMM_WORLD.rank,
            sendbuf.data.ptr,
            sendbuf.data.ptr,
            sendbuf.nbytes,
            sendbuf.nbytes,
            DataType.float16,
            512,
            execution_plan,
            stream.ptr,
        ),
    )
    print(f"Execution time: {execution_time} us, data size: {sendbuf.nbytes} bytes")
    executor = None
    mscclpp_group = None
