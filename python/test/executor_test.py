# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp import (
    DataType,
    Executor,
    ExecutionPlan,
    PacketType,
    npkit,
    env,
)
from mscclpp import CommGroup, GpuBuffer
from mscclpp.utils import KernelBuilder, pack
import os
import struct

import cupy as cp
from mpi4py import MPI


def parse_dtype(dtype_str):
    """Convert a human-readable data type string to a CuPy data type."""
    dtype_str = dtype_str.strip().lower()
    if dtype_str == "float16":
        return cp.float16
    elif dtype_str == "float32":
        return cp.float32
    elif dtype_str == "int32":
        return cp.int32
    else:
        raise ValueError(f"Unknown data type: {dtype_str}")


def bench_time(n_iters: int, n_graph_iters: int, func):
    # Capture CUDA graph for n_iters of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for _ in range(n_iters):
            func(stream)
        graph = stream.end_capture()

    # Warm-up round
    graph.launch(stream)

    # Benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(n_graph_iters):
        graph.launch(stream)
    end.record(stream)
    end.synchronize()

    # Return average execution time in microseconds
    return cp.cuda.get_elapsed_time(start, end) / n_iters * 1000.0 / n_graph_iters


def bench_correctness(
    collective: str,
    input_buf: cp.ndarray,
    result_buf: cp.ndarray,
    test_buf: cp.ndarray,
    dtype_str: str,
    rank: int,
    num_ranks: int,
    n_iters: int,
    func,
):
    type_size = cp.dtype(parse_dtype(dtype_str)).itemsize

    print("collective: ", collective)
    fill_data_kernel_name = "fill_data_%s" % dtype_str
    if "allgather" in collective:
        coll = "all_gather"
    elif "reducescatter" in collective:
        coll = "reduce_scatter"
    elif "allreduce" in collective:
        coll = "all_reduce"
    else:
        coll = "sendrecv"
    test_data_kernel_name = "test_data_%s_%s" % (coll, dtype_str)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    fill_data_kernel = KernelBuilder(
        file="executor_test_verifier.cu",
        kernel_name=fill_data_kernel_name,
        file_dir=file_dir,
    ).get_compiled_kernel()
    test_data_kernel = KernelBuilder(
        file="executor_test_verifier.cu",
        kernel_name=test_data_kernel_name,
        file_dir=file_dir,
    ).get_compiled_kernel()

    nblocks = 64
    nthreads = 1024

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(n_iters):
            fill_data_params = pack(input_buf) + struct.pack("Q", input_buf.nbytes // type_size) + pack(rank, i)
            fill_data_kernel.launch_kernel(fill_data_params, nblocks, nthreads, 0, stream)

            func(stream)

            test_data_params = (
                pack(result_buf, test_buf)
                + struct.pack("Q", input_buf.nbytes // type_size)
                + pack(num_ranks, rank, i)
            )
            test_data_kernel.launch_kernel(test_data_params, nblocks, nthreads, 0, stream)

        graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()


def parse_size(size_str):
    """Convert a human-readable buffer size string to an integer (bytes)."""
    size_str = size_str.strip()
    if not size_str:
        raise ValueError("Size string cannot be empty")

    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if size_str[-1].upper() in units:
        return int(size_str[:-1]) * units[size_str[-1].upper()]
    return int(size_str)

def parse_size_list(size_arg):
    """
    Accept:
      - single size: '1M'
      - comma-separated list: '1K,2K,4K'
      - geometric range: '1K:64K:2' -> start:end:factor

    Returns a list of integer sizes in bytes.
    """
    size_arg = size_arg.strip()

    if "," in size_arg:
        return [parse_size(x) for x in size_arg.split(",")]

    if ":" in size_arg:
        parts = size_arg.split(":")
        if len(parts) != 3:
            raise ValueError("Range format must be start:end:factor, e.g. 1K:64K:2")

        start = parse_size(parts[0])
        end = parse_size(parts[1])
        factor = int(parts[2])

        if start <= 0:
            raise ValueError("Start must be positive")
        if end < start:
            raise ValueError("End must be >= start")
        if factor <= 1:
            raise ValueError("Factor must be greater than 1")

        sizes = []
        current = start
        while current <= end:
            sizes.append(current)
            current *= factor

        return sizes

    return [parse_size(size_arg)]

def dtype_to_mscclpp_dtype(dtype):
    if dtype == cp.float16:
        return DataType.float16
    elif dtype == cp.float32:
        return DataType.float32
    elif dtype == cp.int32:
        return DataType.int32
    else:
        raise ValueError(f"Unknown data type: {dtype}")


def build_bufs(
    collective: str,
    size: int,
    in_place: bool,
    dtype: cp.dtype,
    rank: int,
    num_ranks: int,
):
    type_size = cp.dtype(dtype).itemsize
    assert (size % type_size) == 0, f"size {size} not multiple of type size {type_size}"
    nelems = size // type_size

    if "allgather" in collective:
        assert (nelems % num_ranks) == 0, f"nelems {nelems} not multiple of num_ranks {num_ranks}"
        nelems_input = nelems if in_place else nelems // num_ranks
    else:
        nelems_input = nelems

    if "reducescatter" in collective:
        assert (nelems % num_ranks) == 0, f"nelems {nelems} not multiple of num_ranks {num_ranks}"
        nelems_output = nelems // num_ranks
    else:
        nelems_output = nelems

    result_buf = GpuBuffer(nelems_output, dtype=dtype)

    if in_place:
        if "allgather" in collective:
            input_buf = cp.split(result_buf, num_ranks)[rank]
        elif "reducescatter" in collective:
            input_buf = GpuBuffer(nelems_input, dtype=dtype)
            result_buf = cp.split(input_buf, num_ranks)[rank]
        else:
            input_buf = result_buf
    else:
        input_buf = GpuBuffer(nelems_input, dtype=dtype)

    in_place = False

    test_buf = cp.zeros(nelems, dtype=dtype)

    return input_buf, result_buf, test_buf, nelems


def main(
    execution_plan_path: str,
    sizes: list[int],
    in_place: bool = True,
    dtype_str: str = "float16",
    packet_type: PacketType = PacketType.LL16,
    n_iters: int = 10,
    n_graph_iters: int = 10,
):
    mscclpp_group = CommGroup(MPI.COMM_WORLD)
    nranks = mscclpp_group.nranks
    my_rank = mscclpp_group.my_rank

    cp.cuda.Device(my_rank % mscclpp_group.nranks_per_node).use()

    executor = Executor(mscclpp_group.communicator)
    npkit_dump_dir = env().npkit_dump_dir
    if npkit_dump_dir != "":
        npkit.init(my_rank)

    execution_plan = ExecutionPlan(execution_plan_path, my_rank)
    collective = execution_plan.collective
    dtype = parse_dtype(dtype_str)
    input_buf, result_buf, test_buf, nelem = build_bufs(
        collective,
        size,
        in_place,
        dtype,
        mscclpp_group.my_rank,
        mscclpp_group.nranks,
    )

    # Print header once
    if my_rank == 0:
        print(
            f"{'NRanks':>8} {'Message Size (B)':>18} {'BW (GB/s)':>12} "
            f"{'Latency (us)':>14}      {'Packet Type':>12}"
        )

    for size in sizes:
        input_buf, result_buf, test_buf = build_bufs(
            collective,
            size,
            in_place,
            dtype,
            my_rank,
            nranks,
        )

        executor_func = lambda stream, in_buf=input_buf, out_buf=result_buf: executor.execute(
            my_rank,
            in_buf.data.ptr,
            out_buf.data.ptr,
            in_buf.nbytes,
            out_buf.nbytes,
            dtype_to_mscclpp_dtype(dtype),
            execution_plan,
            stream.ptr,
            packet_type,
        )

        #mscclpp_group.barrier()

        # Optional correctness check
        # bench_correctness(
        #     collective,
        #     input_buf,
        #     result_buf,
        #     test_buf,
        #     dtype_str,
        #     my_rank,
        #     nranks,
        #     n_iters,
        #     executor_func,
        # )

        mscclpp_group.barrier()
        execution_time = bench_time(n_iters, n_graph_iters, executor_func)
        #mscclpp_group.barrier()

        if my_rank == 0:
            msg_size = size
            bw = result_buf.nbytes / execution_time / 1e3  # GB/s
            latency = execution_time  # us

            print(
                f"{nranks:8d} {msg_size:18d} {bw:12.2f} "
                f"{latency:14.2f}       {str(packet_type):>12}"
            )

        # Release buffers for this size
        input_buf = None
        result_buf = None
        test_buf = None

        #mscclpp_group.barrier()

    if npkit_dump_dir != "":
        npkit.dump(npkit_dump_dir)
        npkit.shutdown()

        # Print header once
        print(f"{'NRanks':>8}  {'Message Size (B)':>18} {'BW (GB/s)':>12} {'Latency (us)':>14}     {'Packet Type':>12}")
        print(f"{nranks:8d}  {msg_size:18d} {bw:12.2f} {latency:14.2f}       {str(packet_type):>12}")

    # Sentinel fill: choose something unlikely in your pattern
    result_buf.fill(cp.float16(123.0))
    cp.cuda.runtime.deviceSynchronize()

    # Run ONE execution (no graph), then sync
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        executor_func(stream)
    stream.synchronize()

    # Count how many elements changed
    changed = cp.count_nonzero(result_buf != cp.float16(123.0)).item()
    print("changed elements:", changed, "out of", result_buf.size)

    bench_correctness(
        collective,
        input_buf,
        result_buf,
        test_buf,
        dtype_str,
        mscclpp_group.my_rank,
        mscclpp_group.nranks,
        n_iters,
        executor_func,
    )

        executor_func = lambda stream, in_buf=input_buf, out_buf=result_buf: executor.execute(
            my_rank,
            in_buf.data.ptr,
            out_buf.data.ptr,
            in_buf.nbytes,
            out_buf.nbytes,
            dtype_to_mscclpp_dtype(dtype),
            execution_plan,
            stream.ptr,
            packet_type,
        )

        mscclpp_group.barrier()

        # Optional correctness check
        # bench_correctness(
        #     collective,
        #     input_buf,
        #     result_buf,
        #     test_buf,
        #     dtype_str,
        #     my_rank,
        #     nranks,
        #     n_iters,
        #     executor_func,
        # )

        mscclpp_group.barrier()
        execution_time = bench_time(n_iters, n_graph_iters, executor_func)
        mscclpp_group.barrier()

        if my_rank == 0:
            msg_size = size
            bw = result_buf.nbytes / execution_time / 1e3  # GB/s
            latency = execution_time  # us

            print(
                f"{nranks:8d} {msg_size:18d} {bw:12.2f} "
                f"{latency:14.2f}       {str(packet_type):>12}"
            )

        # Release buffers for this size
        input_buf = None
        result_buf = None
        test_buf = None

        mscclpp_group.barrier()

    if npkit_dump_dir != "":
        npkit.dump(npkit_dump_dir)
        npkit.shutdown()
    print(
        f"Rank: {mscclpp_group.my_rank} Execution time: {execution_time} us, "
        f"data size: {result_buf.nbytes} bytes data type: {dtype().dtype.name} "
        f"packet type: {packet_type}"
    )
    executor = None
    mscclpp_group = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--execution_plan_path", type=str, required=True)
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        help=(
            "Single size (e.g. 1M), comma-separated list (e.g. 1K,2K,4K), "
            "or range start:end:factor (e.g. 1K:64K:2)"
        ),
    )
    parser.add_argument("--in_place", action="store_true", help="Flag to define an in-place operation")
    parser.add_argument("--dtype", type=str, default="float16", help="Choose from float16, float32, int32")
    parser.add_argument("--packet_type", type=str, default="LL16", help="Choose from LL8, LL16")
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_graph_iters", type=int, default=10)
    args = parser.parse_args()

    packet_type = PacketType.LL16
    if args.packet_type == "LL8":
        packet_type = PacketType.LL8

    buffer_sizes = parse_size_list(args.size)

    main(
        args.execution_plan_path,
        buffer_sizes,
        args.in_place,
        args.dtype,
        packet_type,
        args.n_iters,
        args.n_graph_iters,
    )
