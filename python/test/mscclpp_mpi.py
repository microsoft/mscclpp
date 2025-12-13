# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from mpi4py import MPI
import pytest

N_GPUS_PER_NODE = 8

logging.basicConfig(level=logging.INFO)

_mpi_group_cache = {}


class MpiGroup:
    def __init__(self, ranks: list = []):
        world_group = MPI.COMM_WORLD.group
        if len(ranks) == 0:
            self.comm = MPI.COMM_WORLD
        else:
            group = world_group.Incl(ranks)
            self.comm = MPI.COMM_WORLD.Create(group)


@pytest.fixture
def mpi_group(request: pytest.FixtureRequest):
    MPI.COMM_WORLD.barrier()

    mpi_group_obj = request.param
    should_skip = mpi_group_obj.comm == MPI.COMM_NULL

    try:
        if should_skip:
            pytest.skip(f"Skip for rank {MPI.COMM_WORLD.rank}")
        yield request.param
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.barrier()


def parametrize_mpi_groups(*tuples: tuple):
    def decorator(func):
        mpi_groups = []
        for group_size in list(tuples):
            if MPI.COMM_WORLD.size < group_size:
                logging.warning(f"MPI.COMM_WORLD.size < {group_size}, skip")
                continue
            ranks = list(range(group_size))
            ranks_key = tuple(ranks)
            if ranks_key not in _mpi_group_cache:
                _mpi_group_cache[ranks_key] = MpiGroup(ranks)

            mpi_groups.append(_mpi_group_cache[ranks_key])
        return pytest.mark.parametrize("mpi_group", mpi_groups, indirect=True)(func)

    return decorator
