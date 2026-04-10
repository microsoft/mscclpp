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
    dtype_str = dtype_str.strip().lower()
    if dtype_str == "float16":
        return cp.float16
    elif dtype_str == "float32":
        return cp.float32
    elif dtype_str == "int32":
        return cp.int32
    else:
        raise ValueError(f"Unknown data type: {dtype_str}")


def parse_size(size_str):
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


def bench_time(n_iters: int, n_graph_iters: int, func_iter):
    """
    Capture CUDA graph for n_iters launches. func_iter(stream, i) must vary slot by i.
    """
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(n_iters):
            func_iter(stream, i)
        graph = stream.end_capture()

    # warmup
    graph.launch(stream)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(n_graph_iters):
        graph.launch(stream)
    end.record(stream)
    end.synchronize()

    # us per iteration
    return cp.cuda.get_elapsed_time(start, end) / n_iters * 1000.0 / n_graph_iters


def bench_correctness(
    collective: str,
    input_slot: cp.ndarray,
    result_slot: cp.ndarray,
    test_buf: cp.ndarray,
    dtype_str: str,
    rank: int,
    num_ranks: int,
    n_iters: int,
    func_iter,
):
    """
    Correctness check on ONE per-iteration slot view (input_slot/result_slot change per i via func_iter).
    We pass the per-iteration element count to verifier kernels.
    """
    type_size = cp.dtype(parse_dtype(dtype_str)).itemsize
    nelems_per_iter = input_slot.nbytes // type_size

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
            # WARNING: input_slot/result_slot variables are placeholders; actual slot views are chosen inside func_iter.
            # We only use these kernels with the CURRENT slot views computed below for this iteration.
            func_iter(stream, i, do_verify=True, fill_kernel=fill_data_kernel, test_kernel=test_data_kernel,
                      nblocks=nblocks, nthreads=nthreads, nelems_per_iter=nelems_per_iter,
                      test_buf=test_buf, rank=rank, num_ranks=num_ranks)
        graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()


def build_bufs_sendrecv_ring(size_bytes: int, slots: int, dtype: cp.dtype):
    """
    Build ring buffers for sendrecv:
      - per-iteration message bytes = size_bytes
      - total allocated bytes per buffer = slots * size_bytes
    """
    type_size = cp.dtype(dtype).itemsize
    assert (size_bytes % type_size) == 0, "size not multiple of dtype size"

    nelems_per_iter = size_bytes // type_size
    total_nelems = nelems_per_iter * slots

    input_buf = GpuBuffer(total_nelems, dtype=dtype)
    result_buf = GpuBuffer(total_nelems, dtype=dtype)
    test_buf = cp.zeros(nelems_per_iter, dtype=dtype)  # expected for one iteration

    return input_buf, result_buf, test_buf, nelems_per_iter


def main(
    execution_plan_path: str,
    size: int,                 # per-iteration bytes
    in_place: bool = True,
    dtype_str: str = "float16",
    packet_type: PacketType = PacketType.LL16,
    n_iters: int = 10,
    n_graph_iters: int = 10,
    slots: int = 4,            # ring buffer depth
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

    # We only change allocation/behavior for sendrecv
    if "sendrecv" in collective.lower():
        input_buf, result_buf, test_buf, nelems_per_iter = build_bufs_sendrecv_ring(size, slots, dtype)
        type_size = cp.dtype(dtype).itemsize
        bytes_per_iter = nelems_per_iter * type_size

        def slot_view(buf, slot_idx):
            start = slot_idx * nelems_per_iter
            end = start + nelems_per_iter
            return buf[start:end]

        # Iteration-aware executor call (rotates slot each iteration)
        def executor_func_iter(stream, i, do_verify=False, **vk):
            slot = i % slots
            in_slot = slot_view(input_buf, slot)
            out_slot = slot_view(result_buf, slot)

            if do_verify:
                # Fill per-iteration input slot with unique (rank, i) pattern
                fill_data_kernel = vk["fill_kernel"]
                test_data_kernel = vk["test_kernel"]
                nblocks = vk["nblocks"]
                nthreads = vk["nthreads"]
                nelems = vk["nelems_per_iter"]
                test_buf_local = vk["test_buf"]
                rank = vk["rank"]
                num_ranks = vk["num_ranks"]

                fill_params = pack(in_slot) + struct.pack("Q", nelems) + pack(rank, i)
                fill_data_kernel.launch_kernel(fill_params, nblocks, nthreads, 0, stream)

            # Execute exactly one per-iteration message: bytes_per_iter == user --size
            executor.execute(
                mscclpp_group.my_rank,
                in_slot.data.ptr,
                out_slot.data.ptr,
                in_slot.nbytes,
                out_slot.nbytes,
                dtype_to_mscclpp_dtype(dtype),
                execution_plan,
                stream.ptr,
                packet_type,
            )

            if do_verify:
                # Validate the output slot for this iteration i
                test_params = (
                    pack(out_slot, test_buf_local)
                    + struct.pack("Q", nelems)
                    + pack(num_ranks, rank, i)
                )
                test_data_kernel.launch_kernel(test_params, nblocks, nthreads, 0, stream)

        # One-shot sentinel check (slot 0)
        mscclpp_group.barrier()
        print("per-iter size= ", bytes_per_iter, "bytes, slots=", slots, "total buffer bytes=", input_buf.nbytes)

        # Fill whole result with sentinel then run ONE iter (i=0)
        result_buf.fill(cp.asarray(123.0, dtype=dtype))
        cp.cuda.runtime.deviceSynchronize()

        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            executor_func_iter(stream, 0)
        stream.synchronize()

        # Count changes only in slot 0 region
        out0 = slot_view(result_buf, 0)
        changed = cp.count_nonzero(out0 != cp.asarray(123.0, dtype=dtype)).item()
        print("changed elements in slot0:", changed, "out of", out0.size)

        cp.cuda.runtime.deviceSynchronize()
        mscclpp_group.barrier()

        # Correctness: fills + executes + tests with unique i and rotating slots
        bench_correctness(
            collective,
            slot_view(input_buf, 0),   # placeholder; real slot chosen per i
            slot_view(result_buf, 0),  # placeholder; real slot chosen per i
            test_buf,
            dtype_str,
            mscclpp_group.my_rank,
            mscclpp_group.nranks,
            n_iters,
            executor_func_iter,
        )

        mscclpp_group.barrier()

        # Timing (CUDA graph captures n_iters launches with varying slot pointers)
        execution_time = bench_time(n_iters, n_graph_iters, executor_func_iter)

        if npkit_dump_dir is not None:
            npkit.dump(npkit_dump_dir)
            npkit.shutdown()

        print(
            f"Rank: {mscclpp_group.my_rank} Execution time: {execution_time} us, "
            f"per-iter data size: {bytes_per_iter} bytes dtype: {dtype().dtype.name} "
            f"bandwidth: {bytes_per_iter / (execution_time * 1e-6) / (1024**3):.2f} GB/s, "
            f"packet type: {packet_type}, slots: {slots}"
        )

    else:
        raise RuntimeError(
            f"This rewritten executor_test.py currently specializes sendrecv. "
            f"Plan collective was: {collective}"
        )

    executor = None
    mscclpp_group = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--execution_plan_path", type=str, required=True)
    parser.add_argument("--size", type=str, required=True, help="PER-ITERATION bytes (e.g., 1K, 4M, 1G)")
    parser.add_argument("--in_place", action="store_true", help="flag to define an in-place operation")
    parser.add_argument("--dtype", type=str, default="float16", help="Choose from float16, float32, int32")
    parser.add_argument("--packet_type", type=str, default="LL16", help="Choose from LL8, LL16")
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--n_graph_iters", type=int, default=10)
    parser.add_argument("--slots", type=int, default=4, help="ring buffer depth; rotates slot = iter % slots")
    args = parser.parse_args()

    packet_type = PacketType.LL16
    if args.packet_type == "LL8":
        packet_type = PacketType.LL8

    per_iter_size = parse_size(args.size)

    main(
        args.execution_plan_path,
        per_iter_size,
        args.in_place,
        args.dtype,
        packet_type,
        args.n_iters,
        args.n_graph_iters,
        args.slots,
    )
