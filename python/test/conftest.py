# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mpi4py
import os
import sys

mpi4py.rc.initialize = False
mpi4py.rc.finalize = True

import cupy as cp
import pytest
from mpi4py import MPI


def pytest_configure(config):
    """Initialize MPI before test collection."""
    config.addinivalue_line("markers", "nranks(count): run only with exactly count MPI ranks")
    if not MPI.Is_initialized():
        MPI.Init()
        shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        N_GPUS_PER_NODE = shm_comm.size
        shm_comm.Free()
        cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

        # only print process with rank 0 to avoid bad fd issue
        if MPI.COMM_WORLD.rank != 0:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")


def pytest_collection_modifyitems(items):
    """Apply rank requirements before test fixtures are initialized."""
    nranks = MPI.COMM_WORLD.Get_size()
    for item in items:
        marker = item.get_closest_marker("nranks")
        if marker is None:
            continue
        if len(marker.args) != 1 or marker.kwargs or type(marker.args[0]) is not int or marker.args[0] <= 0:
            raise pytest.UsageError(f"{item.nodeid}: nranks requires one positive integer argument")
        required = marker.args[0]
        if nranks != required:
            item.add_marker(pytest.mark.skip(reason=f"requires {required} MPI ranks (running with {nranks})"))
