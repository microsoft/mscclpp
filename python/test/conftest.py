import atexit
import mpi4py
import os
import sys

mpi4py.rc.initialize = False
mpi4py.rc.finalize = True

import cupy as cp
from mpi4py import MPI
import pytest


def pytest_configure(config):
    """Initialize MPI before test collection."""
    if not MPI.Is_initialized():
        MPI.Init()
        shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        N_GPUS_PER_NODE = shm_comm.size
        shm_comm.Free()
        cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

        if MPI.COMM_WORLD.rank != 0:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
