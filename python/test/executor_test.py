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
from typing import Callable, Union

import cupy as cp
from mpi4py import MPI


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


def bench_time(n_iters: int, n_graph_iters: int, func: Union[Callable, list[Callable]]):
    """Benchmark execution time. func can be a single callable or a list of 2 for double-buffer."""
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(n_iters):
            if isinstance(func, list):
                func[i % 2](stream)
            else:
                func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(n_graph_iters):
        graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / n_iters * 1000.0 / n_graph_iters


def bench_correctness(
    collective: str,
    input_buf: Union[cp.ndarray, list[cp.ndarray]],
    result_buf: Union[cp.ndarray, list[cp.ndarray]],
    test_buf: Union[cp.ndarray, list[cp.ndarray]],
    dtype_str: str,
    rank: int,
    num_ranks: int,
    n_iters: int,
    func: Union[Callable, list[Callable]],
):
    """Validate correctness. For sendrecv, buffers and func are lists of 2 for double-buffer."""
    type_size = cp.dtype(parse_dtype(dtype_str)).itemsize
    double_buf = isinstance(input_buf, list)

    fill_data_kernel_name = "fill_data_%s" % dtype_str
    if "allgather" in collective:
        coll = "all_gather"
    elif "reducescatter" in collective:
        coll = "reduce_scatter"
    elif "allreduce" in collective:
        coll = "all_reduce"
    elif "sendrecv" in collective:
        coll = "send_recv"
    else:
        raise ValueError(f"Unknown collective: {collective}")
    test_data_kernel_name = "test_data_%s_%s" % (coll, dtype_str)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    fill_data_kernel = KernelBuilder(
        file="executor_test_verifier.cu", kernel_name=fill_data_kernel_name, file_dir=file_dir
    ).get_compiled_kernel()
    test_data_kernel = KernelBuilder(
        file="executor_test_verifier.cu", kernel_name=test_data_kernel_name, file_dir=file_dir
    ).get_compiled_kernel()
    nblocks = 64
    nthreads = 1024

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(n_iters):
            if double_buf:
                idx = i % 2
                cur_input = input_buf[idx]
                cur_result = result_buf[idx]
                cur_test = test_buf[idx]
                cur_func = func[idx]
            else:
                cur_input = input_buf
                cur_result = result_buf
                cur_test = test_buf
                cur_func = func

            fill_data_params = pack(cur_input) + struct.pack("Q", cur_input.nbytes // type_size) + pack(rank, i)
            fill_data_kernel.launch_kernel(fill_data_params, nblocks, nthreads, 0, stream)
            cur_func(stream)
            test_data_params = (
                pack(cur_result, cur_test)
                + struct.pack("Q", cur_input.nbytes // type_size)
                + pack(num_ranks, rank, i)
            )
            test_data_kernel.launch_kernel(test_data_params, nblocks, nthreads, 0, stream)
        graph = stream.end_capture()
    graph.launch(stream)
    stream.synchronize()


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
    assert (size % type_size) == 0, "size %d not multiple of type size %d" % (size, type_size)
    nelems = size // type_size

    # Sendrecv uses double buffering: return lists of 2 buffers
    if "sendrecv" in collective:
        input_bufs = [GpuBuffer(nelems, dtype=dtype) for _ in range(2)]
        result_bufs = [GpuBuffer(nelems, dtype=dtype) for _ in range(2)]
        test_bufs = [cp.zeros(nelems, dtype=dtype) for _ in range(2)]
        return input_bufs, result_bufs, test_bufs, nelems

    if "allgather" in collective:
        assert (nelems % num_ranks) == 0, "nelems %d not multiple of num_ranks %d" % (nelems, num_ranks)
        nelems_input = nelems if in_place else nelems // num_ranks
    else:
        nelems_input = nelems

    if "reducescatter" in collective:
        assert (nelems % num_ranks) == 0, "nelems %d not multiple of num_ranks %d" % (nelems, num_ranks)
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

    test_buf = cp.zeros(nelems, dtype=dtype)

    return input_buf, result_buf, test_buf, nelems


def main(
    execution_plan_path: str,
    size: int,
    in_place: bool = True,
    dtype_str: str = "float16",
    packet_type: PacketType = PacketType.LL16,
    n_iters: int = 10,
    n_graph_iters: int = 10,
):
    mscclpp_group = CommGroup(MPI.COMM_WORLD)
    cp.cuda.Device(mscclpp_group.my_rank % mscclpp_group.nranks_per_node).use()
    executor = Executor(mscclpp_group.communicator)
    npkit_dump_dir = env().npkit_dump_dir
    if npkit_dump_dir != "":
        npkit.init(mscclpp_group.my_rank)
    execution_plan = ExecutionPlan(execution_plan_path, mscclpp_group.my_rank)
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

    sendrecv_mode = "sendrecv" in collective

    if sendrecv_mode:
        # Double-buffer: create two executor funcs, one per buffer pair
        executor_funcs = []
        for idx in range(2):
            func = lambda stream, i=idx: executor.execute(
                mscclpp_group.my_rank,
                input_buf[i].data.ptr,
                result_buf[i].data.ptr,
                input_buf[i].nbytes,
                result_buf[i].nbytes,
                dtype_to_mscclpp_dtype(dtype),
                execution_plan,
                stream.ptr,
                packet_type,
            )
            executor_funcs.append(func)
    else:
        executor_func = lambda stream: executor.execute(
            mscclpp_group.my_rank,
            input_buf.data.ptr,
            result_buf.data.ptr,
            input_buf.nbytes,
            result_buf.nbytes,
            dtype_to_mscclpp_dtype(dtype),
            execution_plan,
            stream.ptr,
            packet_type,
        )

    mscclpp_group.barrier()
    bench_correctness(
        collective,
        input_buf,
        result_buf,
        test_buf,
        dtype_str,
        mscclpp_group.my_rank,
        mscclpp_group.nranks,
        n_iters,
        executor_funcs if sendrecv_mode else executor_func,
    )

    mscclpp_group.barrier()
    execution_time = bench_time(n_iters, n_graph_iters, executor_funcs if sendrecv_mode else executor_func)
    if npkit_dump_dir is not None:
        npkit.dump(npkit_dump_dir)
        npkit.shutdown()

    result_nbytes = result_buf[0].nbytes if sendrecv_mode else result_buf.nbytes
    print(
        f"Rank: {mscclpp_group.my_rank} Execution time: {execution_time} us, "
        f"data size: {result_nbytes} bytes data type: {dtype().dtype.name} "
        f"bandwidth: {result_nbytes / (execution_time * 1e-6) / (1024**3):.2f} GB/s, "
        f"packet type: {packet_type}"
    )
    executor = None
    mscclpp_group = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--execution_plan_path", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--in_place", action="store_true", help="flag to define an in-place operation")
    parser.add_argument("--dtype", type=str, default="float16", help="Choose from float16, float32, int32")
    parser.add_argument("--packet_type", type=str, default="LL16", help="Choose from LL8, LL16")
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_graph_iters", type=int, default=10)
    args = parser.parse_args()

    packet_type = PacketType.LL16
    if args.packet_type == "LL8":
        packet_type = PacketType.LL8

    buffer_size = parse_size(args.size)
    main(
        args.execution_plan_path,
        buffer_size,
        args.in_place,
        args.dtype,
        packet_type,
        args.n_iters,
        args.n_graph_iters,
    )
