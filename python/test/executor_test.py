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

if __name__ == "__main__":
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    N_GPUS_PER_NODE = shm_comm.size
    shm_comm.Free()

    cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()
    mscclpp_group = mscclpp_comm.CommGroup(MPI.COMM_WORLD)
    executor = Executor(mscclpp_group.communicator, N_GPUS_PER_NODE)
    execution_plan = ExecutionPlan(path.join(MSCCLPP_ROOT_PATH, "test", "execution-files", "allreduce.json"))

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
    executor = None
    mscclpp_group = None
