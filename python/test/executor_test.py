# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp import (
    DataType,
    Executor,
    ExecutionPlan,
    PacketType,
    npkit,
)
import mscclpp.comm as mscclpp_comm
import os

import cupy as cp
from mpi4py import MPI


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


def parse_size(size_str):
    """Convert a human-readable buffer size string to an integer."""
    size_str = size_str.strip()
    if not size_str:
        raise ValueError("Size string can not be empty")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if size_str[-1].upper() in units:
        return int(size_str[:-1]) * units[size_str[-1].upper()]
    else:
        return int(size_str)


def parse_dtype(dtype_str):
    """Convert a human-readable data type string to a numpy data type."""
    dtype_str = dtype_str.strip().lower()
    if dtype_str == "float16":
        return cp.float16
    elif dtype_str == "float32":
        return cp.float32
    elif dtype_str == "int32":
        return cp.int32
    else:
        raise ValueError(f"Unknown data type: {dtype_str}")


def dtype_to_mscclpp_dtype(dtype):
    if dtype == cp.float16:
        return DataType.float16
    elif dtype == cp.float32:
        return DataType.float32
    elif dtype == cp.int32:
        return DataType.int32
    else:
        raise ValueError(f"Unknown data type: {dtype}")


def main(
    execution_plan_name: str,
    execution_plan_path: str,
    size: int,
    nthreads_per_block: int,
    in_place: bool = True,
    dtype: cp.dtype = cp.float16,
    packet_type: PacketType = PacketType.LL16,
    seed: int = 42,
):
    mscclpp_group = mscclpp_comm.CommGroup(MPI.COMM_WORLD)
    cp.cuda.Device(mscclpp_group.my_rank % mscclpp_group.nranks_per_node).use()
    executor = Executor(mscclpp_group.communicator)
    npkit_dump_dir = os.getenv("NPKIT_DUMP_DIR")
    if npkit_dump_dir is not None:
        npkit.init(mscclpp_group.my_rank)
    execution_plan = ExecutionPlan(execution_plan_name, execution_plan_path)

    if "allgather" in execution_plan_name:
        cp.random.seed(seed)
        nelems = size // cp.dtype(dtype).itemsize
        buffer = cp.empty(nelems * mscclpp_group.nranks, dtype=dtype)
        buffer[:] = cp.random.random(nelems * mscclpp_group.nranks, dtype=cp.float32).astype(dtype)
        sub_arrays = cp.split(buffer, MPI.COMM_WORLD.size)
        sendbuf = cp.zeros(nelems, dtype=dtype)
        for i in range(nelems):
            sendbuf[i] = sub_arrays[MPI.COMM_WORLD.rank][i]
        recvbuf = cp.zeros(nelems * mscclpp_group.nranks, dtype=dtype)
        expected = buffer
    else:
        cp.random.seed(seed)
        nelems = size // cp.dtype(dtype).itemsize
        buffer = cp.empty(nelems * mscclpp_group.nranks, dtype=dtype)
        buffer[:] = cp.random.random(nelems * mscclpp_group.nranks, dtype=cp.float32).astype(dtype)
        sub_arrays = cp.split(buffer, MPI.COMM_WORLD.size)
        sendbuf = cp.zeros(nelems, dtype=dtype)
        for i in range(nelems):
            sendbuf[i] = sub_arrays[MPI.COMM_WORLD.rank][i]
        recvbuf = cp.zeros(nelems, dtype=dtype)
        expected = cp.zeros_like(sendbuf, dtype=dtype)
        for i in range(mscclpp_group.nranks):
            expected += sub_arrays[i]
    mscclpp_group.barrier()

    executor_func = lambda stream: executor.execute(
        MPI.COMM_WORLD.rank,
        sendbuf.data.ptr,
        sendbuf.data.ptr if in_place else recvbuf.data.ptr,
        sendbuf.nbytes,
        sendbuf.nbytes if in_place else recvbuf.nbytes,
        dtype_to_mscclpp_dtype(dtype),
        nthreads_per_block,
        execution_plan,
        stream.ptr,
        packet_type,
    )
    # check correctness
    stream = cp.cuda.Stream(non_blocking=True)
    executor_func(stream)
    stream.synchronize()

    #for i in range(nelems * mscclpp_group.nranks):
    #    print(f"Rank: {MPI.COMM_WORLD.rank} recvbuf[{i}]: {recvbuf[i]} expected[{i}]: {expected[i]}")

    assert cp.allclose(sendbuf if in_place else recvbuf, expected, atol=1e-2 * mscclpp_group.nranks)

    mscclpp_group.barrier()
    execution_time = bench_time(1, 1, executor_func)
    if npkit_dump_dir is not None:
        npkit.dump(npkit_dump_dir)
        npkit.shutdown()
    print(
        f"Rank: {MPI.COMM_WORLD.rank} Execution time: {execution_time} us, "
        f"data size: {sendbuf.nbytes} bytes data type: {dtype().dtype.name} "
        f"packet type: {packet_type} nthreads_per_block: {nthreads_per_block}"
    )
    executor = None
    mscclpp_group = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--execution_plan_name", type=str, required=True)
    parser.add_argument("-path", "--execution_plan_path", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--nthreads_per_block", type=int, required=True)
    parser.add_argument("--in_place", type=str, default="true", help="Choose from true, false")
    parser.add_argument("--dtype", type=str, default="float16", help="Choose from float16, float32, int32")
    parser.add_argument("--packet_type", type=str, default="LL16", help="Choose from LL8, LL16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    packet_type = PacketType.LL16
    if args.packet_type == "LL8":
        packet_type = PacketType.LL8

    buffer_size = parse_size(args.size)
    dtype = parse_dtype(args.dtype)
    main(
        args.execution_plan_name,
        args.execution_plan_path,
        buffer_size,
        args.nthreads_per_block,
        args.in_place.lower() == "true",
        dtype,
        packet_type,
        args.seed,
    )
