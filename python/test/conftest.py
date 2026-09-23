# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mpi4py
import os
import sys

mpi4py.rc.initialize = False
mpi4py.rc.finalize = True

from mpi4py import MPI


def pytest_configure(config):
    """Initialize MPI before test collection."""
    if not MPI.Is_initialized():
        MPI.Init()
        shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        N_GPUS_PER_NODE = shm_comm.size
        shm_comm.Free()
        from mscclpp_benchmark.gpu import set_device

        set_device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE)

        # only print process with rank 0 to avoid bad fd issue
        if MPI.COMM_WORLD.rank != 0:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
