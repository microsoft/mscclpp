# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Registered contract and multi-rank tests for the MNNVL-only all-reduce."""

import inspect
import os

import cupy as cp
import pytest

import mscclpp
from mscclpp import CommGroup, DataType, Executor, PacketType
from mscclpp.default_algos import allreduce_mnnvl_only
from mscclpp.language.collectives import AllReduce
from mscclpp.language.utils import AlgoSpec

from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups


ROWS = (16, 32, 64, 128, 192, 256, 384, 512)
HIDDEN = 6144


def test_mnnvl_plan_has_no_network_channel_fallback():
    source = inspect.getsource(allreduce_mnnvl_only)
    assert "MemoryChannel" in source
    assert "read_put_packets" in source
    assert "PortChannel" not in source
    assert "Ib" not in source and "RDMA" not in source


def _spec(world_size: int, ranks_per_node: int) -> AlgoSpec:
    return AlgoSpec(
        name="test_mnnvl_allreduce",
        collective=AllReduce(world_size, 1, True),
        nranks_per_node=ranks_per_node,
        world_size=world_size,
        in_place=True,
        instances=1,
        protocol="LL",
        auto_sync=False,
        num_threads_per_block=256,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        min_message_size=1 << 10,
        max_message_size=8 << 20,
        tags={"test": 1},
    )


@parametrize_mpi_groups(4, 8, 16)
def test_mnnvl_exact_rows_graph_reuse_mismatch_and_cleanup(mpi_group: MpiGroup):
    if os.environ.get("MSCCLPP_RUN_MNNVL_TESTS") != "1":
        pytest.skip("set MSCCLPP_RUN_MNNVL_TESTS=1 on an MNNVL allocation")

    world = mpi_group.comm.size
    rank = mpi_group.comm.rank
    ranks_per_node = int(os.environ.get("MSCCLPP_TEST_RANKS_PER_NODE", "4"))
    if world % ranks_per_node:
        pytest.skip("world size is not divisible by MSCCLPP_TEST_RANKS_PER_NODE")

    group = CommGroup(mpi_group.comm)
    executor = Executor(group.communicator)
    algo = mscclpp.compile(
        allreduce_mnnvl_only,
        _spec(world, ranks_per_node),
        rank,
        thread_block_group_size=1,
    )
    stream = cp.cuda.Stream(non_blocking=True)
    buffers = []

    for rows in ROWS:
        if rows % world:
            continue
        buf = cp.zeros((rows, HIDDEN), dtype=cp.float16)
        local_rows = rows // world
        buf[rank * local_rows : (rank + 1) * local_rows].fill(rank + 1)
        rc = algo.prepare_dsl(
            group.communicator,
            buf.data.ptr,
            buf.data.ptr,
            buf.nbytes,
            buf.nbytes,
            DataType.float16,
            executor,
            PacketType.LL16,
        )
        assert int(rc) == 0
        buffers.append(buf)

    assert algo.prepared_context_count == len(buffers)
    assert len(set(algo.prepared_context_ids)) == len(buffers)
    assert algo.prepared_resource_bytes > 0

    # An exact-key mismatch must fail before launching any kernel.
    first = buffers[0]
    mismatch = algo.execute_prepared_dsl(
        group.communicator,
        first.data.ptr,
        first.data.ptr,
        first.nbytes - 16,
        first.nbytes,
        DataType.float16,
        executor,
        stream.ptr,
        PacketType.LL16,
    )
    assert int(mismatch) != 0

    with stream:
        stream.begin_capture()
        for buf in buffers:
            rc = algo.execute_prepared_dsl(
                group.communicator,
                buf.data.ptr,
                buf.data.ptr,
                buf.nbytes,
                buf.nbytes,
                DataType.float16,
                executor,
                stream.ptr,
                PacketType.LL16,
            )
            assert int(rc) == 0
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()

    def assert_exact_rows():
        for buf in buffers:
            expected = cp.repeat(
                cp.arange(1, world + 1, dtype=cp.float16), buf.shape[0] // world
            )[:, None]
            assert bool(cp.all(buf == expected))

    assert_exact_rows()
    for buf in buffers:
        buf.fill(0)
        local_rows = buf.shape[0] // world
        buf[rank * local_rows : (rank + 1) * local_rows].fill(rank + 1)
    stream.synchronize()
    graph.launch(stream)
    stream.synchronize()
    assert_exact_rows()

    # Collective release while bootstrap/control peers are still alive.
    mpi_group.comm.Barrier()
    algo.reset()
    assert algo.prepared_context_count == 0
    assert algo.prepared_resource_bytes == 0
    mpi_group.comm.Barrier()
    del executor, algo, group
    mpi_group.comm.Barrier()
