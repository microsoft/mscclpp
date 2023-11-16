# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import cupy as cp
from mscclpp_op import MscclppAllReduce1, MscclppAllReduce2, MscclppAllReduce3, MscclppAllReduce4, MscclppAllReduce5
from nccl_op import NcclAllReduce
from mpi4py import MPI
import cupy.cuda.nccl as nccl
import mscclpp.comm as mscclpp_comm
from mscclpp import ProxyService
from prettytable import PrettyTable
import netifaces as ni

data_type = cp.float16

if data_type == cp.float16:
    dtype_str = "fp16"
elif data_type == cp.float32:
    dtype_str = "fp32"
elif data_type == cp.int32:
    dtype_str = "int32"
else:
    raise RuntimeError("Unknown data type")


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def check_correctness(memory, func):
    rand_gen = cp.random.default_rng(seed=MPI.COMM_WORLD.rank)
    memory[:] = rand_gen.random(memory.shape).astype(data_type)
    cp.cuda.runtime.deviceSynchronize()
    output_memory = func(0)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.zeros_like(memory)
    for i in range(MPI.COMM_WORLD.size):
        rand_gen = cp.random.default_rng(seed=i)
        expected += rand_gen.random(memory.shape).astype(data_type)

    if data_type == cp.float16:
        ac = cp.allclose(output_memory, expected, rtol=1.0e-2, atol=1.0e-4)
    else:
        ac = cp.allclose(output_memory, expected, rtol=1.0e-2, atol=1.0e-4)

    ac = MPI.COMM_WORLD.allreduce(ac, op=MPI.SUM)
    if not ac:
        print(output_memory, expected)
    return ac


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


def find_best_config(mscclpp_call, niter):
    best_time = 10000000.0
    for config in mscclpp_call.auto_tune():
        cur_time = bench_time(niter, mscclpp_call)
        if cur_time < best_time:
            best_time = cur_time
            best_config = config
        if MPI.COMM_WORLD.rank == 0:
            print("t", end="", flush=True)
    best_config = MPI.COMM_WORLD.bcast(best_config, root=0)
    if MPI.COMM_WORLD.rank == 0:
        print(best_config, end="", flush=True)
    return best_config


def run_benchmark(
    mscclpp_group: mscclpp_comm.CommGroup, nccl_op: nccl.NcclCommunicator, table: PrettyTable, niter: int, nelem: int
):
    memory = cp.zeros(nelem, dtype=data_type)
    memory_out = cp.zeros(nelem, dtype=data_type)
    cp.cuda.runtime.deviceSynchronize()

    if MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 1:
        if memory.nbytes < 2**20:
            mscclpp_call = MscclppAllReduce2(mscclpp_group, memory, memory_out)
        elif memory.nbytes < 2**29:
            if memory.nbytes >= 2**20 and memory.nbytes <= 2**22:
                read_only = 0
            else:
                read_only = 1
            mscclpp_call = MscclppAllReduce1(mscclpp_group, memory, read_only=read_only)
        else:
            proxy_service = ProxyService()
            mscclpp_call = MscclppAllReduce3(mscclpp_group, memory, proxy_service)
            proxy_service.start_proxy()
    else:
        if memory.nbytes < 2**22:
            proxy_service = ProxyService()
            mscclpp_call = MscclppAllReduce5(mscclpp_group, memory, memory_out, N_GPUS_PER_NODE, proxy_service)
            proxy_service.start_proxy()
            best_config = find_best_config(mscclpp_call, 100)
            mscclpp_call.set_params(*best_config)
        else:
            proxy_service = ProxyService()
            mscclpp_call = MscclppAllReduce4(mscclpp_group, memory, N_GPUS_PER_NODE, proxy_service)
            proxy_service.start_proxy()
            best_config = find_best_config(mscclpp_call, 20)
            mscclpp_call.set_params(*best_config)

    nccl_call = NcclAllReduce(nccl_op, memory)

    memory_nbytes = memory.nbytes
    mscclpp_time = bench_time(niter, mscclpp_call)
    mscclpp_algBw = memory_nbytes / mscclpp_time / 1e3
    mscclpp_check = "PASS" if check_correctness(memory, mscclpp_call) else "FAIL"

    nccl_time = bench_time(niter, nccl_call)
    nccl_algBw = memory_nbytes / nccl_time / 1e3
    nccl_check = "PASS" if check_correctness(memory, nccl_call) else "FAIL"

    if (
        isinstance(mscclpp_call, MscclppAllReduce3)
        or isinstance(mscclpp_call, MscclppAllReduce5)
        or isinstance(mscclpp_call, MscclppAllReduce4)
    ):
        MPI.COMM_WORLD.barrier()
        proxy_service.stop_proxy()

    if MPI.COMM_WORLD.rank == 0:
        table.add_row(
            [
                human_readable_size(memory_nbytes),
                "{:.2f}".format(mscclpp_time),
                "{:.2f}".format(mscclpp_algBw),
                mscclpp_check,
                "{:.2f}".format(nccl_time),
                "{:.2f}".format(nccl_algBw),
                nccl_check,
                "{:.2f}".format(nccl_time / mscclpp_time),
            ]
        )
    if MPI.COMM_WORLD.rank == 0:
        print(".", end="", flush=True)


if __name__ == "__main__":
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    N_GPUS_PER_NODE = shm_comm.size
    shm_comm.Free()
    cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

    # create a MscclppGroup
    network_interface = "eth0"
    my_ip = ni.ifaddresses(network_interface)[ni.AF_INET][0]["addr"]
    root_ip = MPI.COMM_WORLD.bcast(my_ip, root=0)
    ifIpPortTrio = network_interface + ":" + root_ip + ":50000"  # some random port
    mscclpp_group = mscclpp_comm.CommGroup(
        interfaceIpPortTrio=ifIpPortTrio, rank=MPI.COMM_WORLD.rank, size=MPI.COMM_WORLD.size
    )

    # create a NcclComm
    if MPI.COMM_WORLD.rank == 0:
        uid = nccl.get_unique_id()
    else:
        uid = None
    uid = MPI.COMM_WORLD.bcast(uid, root=0)
    nccl_comm = nccl.NcclCommunicator(MPI.COMM_WORLD.size, uid, MPI.COMM_WORLD.rank)

    table = None
    if MPI.COMM_WORLD.rank == 0:
        # Set table headers
        table = PrettyTable()
        table.field_names = [
            f"Size ({dtype_str})",
            "Time (us)",
            "AlgBW (GB/s)",
            "Correctness",
            "NCCL Time (us)",
            "NCCL AlgBW (GB/s)",
            "NCCL Correctness",
            "Speed Up",
        ]

    for i in range(10, 28):
        if MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 1:
            run_benchmark(mscclpp_group, nccl_comm, table, 100, 2**i)
        elif MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 2:
            run_benchmark(mscclpp_group, nccl_comm, table, 100, 3 * 2**i)
        else:
            raise RuntimeError("Only support one node/two nodes communication")

    if MPI.COMM_WORLD.rank == 0:
        print()
        print(table)
    mscclpp_group = None
    nccl_comm = None
