# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import socket

import cupy as cp
import pytest

from mscclpp import CommGroup, DataType, RawGpuBuffer, ReduceOp, GpuBufferPool, is_nvls_supported
from mscclpp.ext import AlgorithmCollectionBuilder
from mscclpp_benchmark.gpu import capture_graph
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group  # noqa: F401


def _same_host(comm) -> bool:
    hostnames = comm.allgather(socket.gethostname())
    return len(set(hostnames)) == 1


def _build_nvls_zero_algorithm(mpi_group: MpiGroup):
    comm_group = CommGroup(mpi_group.comm)
    scratch = RawGpuBuffer(1 << 27)
    AlgorithmCollectionBuilder.reset()
    builder = AlgorithmCollectionBuilder()
    algorithms = builder.build_default_algorithms(
        scratch_buffer=scratch.data(),
        scratch_buffer_size=scratch.bytes(),
        rank=comm_group.my_rank,
    )
    for algorithm in algorithms:
        if algorithm.name == "default_allreduce_nvls_zero_copy":
            return comm_group, algorithm, scratch
    pytest.skip("default_allreduce_nvls_zero_copy is not available")


def _torch_tensor_from_pool_buffer(torch, buffer, nelems: int):
    return torch.utils.dlpack.from_dlpack(buffer.to_dlpack(data_type=str(torch.float32), shape=[nelems]))


def _run_nvls_zero_copy(algorithm, comm_group, buffer, stream) -> None:
    ret = algorithm.execute(
        comm=comm_group.communicator,
        input_buffer=buffer.data(),
        output_buffer=buffer.data(),
        input_size=buffer.bytes(),
        output_size=buffer.bytes(),
        dtype=DataType.float32,
        op=ReduceOp.SUM,
        stream=stream.ptr,
        nblocks=0,
        nthreads_per_block=0,
        symmetric_memory=True,
        accum_dtype=DataType.float32,
    )
    assert ret == 0


@parametrize_mpi_groups(2, 4, 8)
def test_gpu_buffer_pool_allreduce_nvls_zero_copy_timing(mpi_group: MpiGroup):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is not available")
    if not is_nvls_supported():
        pytest.skip("NVLS is not supported")
    if not _same_host(mpi_group.comm):
        pytest.skip("NVLS zero-copy test requires all ranks on the same host")

    torch.cuda.set_device(cp.cuda.Device().id)
    comm_group, algorithm, scratch = _build_nvls_zero_algorithm(mpi_group)
    stream = cp.cuda.Stream(non_blocking=True)

    message_sizes = (256 * 1024, 1024 * 1024)
    element_size = torch.empty((), dtype=torch.float32, device="cuda").element_size()
    n_warmup = 3
    n_iters = 10
    pool = GpuBufferPool(sum(nbytes + 4096 for nbytes in message_sizes))
    expected = float(comm_group.nranks * (comm_group.nranks + 1) // 2)
    live_tensors = []
    graphs = []

    try:
        for nbytes in message_sizes:
            nelems = nbytes // element_size
            buffer = pool.allocate(nbytes, alignment=4096)
            tensor = _torch_tensor_from_pool_buffer(torch, buffer, nelems)
            tensor.fill_(float(comm_group.my_rank + 1))
            torch.cuda.synchronize()
            mpi_group.comm.barrier()

            _run_nvls_zero_copy(algorithm, comm_group, buffer, stream)
            stream.synchronize()
            assert torch.allclose(tensor, torch.full_like(tensor, expected))

            tensor.fill_(float(comm_group.my_rank + 1))
            torch.cuda.synchronize()
            mpi_group.comm.barrier()

            graph = capture_graph(stream, lambda: _run_nvls_zero_copy(algorithm, comm_group, buffer, stream))
            graphs.append(graph)
            graph.launch(stream)
            stream.synchronize()
            assert torch.allclose(tensor, torch.full_like(tensor, expected))

            for _ in range(n_warmup):
                graph.launch(stream)
            stream.synchronize()
            mpi_group.comm.barrier()

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            for _ in range(n_iters):
                graph.launch(stream)
            end.record(stream)
            end.synchronize()
            mpi_group.comm.barrier()

            elapsed_us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / n_iters
            all_elapsed_us = mpi_group.comm.allgather(elapsed_us)
            if comm_group.my_rank == 0:
                avg_us = max(all_elapsed_us)
                print(
                    f"default_allreduce_nvls_zero_copy graph with GpuBufferPool: "
                    f"nranks={comm_group.nranks}, nbytes={nbytes}, avg={avg_us:.2f} us"
                )
            live_tensors.append(tensor)
            del buffer

    finally:
        for graph in graphs:
            graph.close()
        live_tensors.clear()
        torch.cuda.synchronize()
        AlgorithmCollectionBuilder.reset()
        del scratch
