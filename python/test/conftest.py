# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Import fixtures to make them available to all test files
from .mscclpp_mpi import mpi_group, MpiGroup, parametrize_mpi_groups  # noqa: F401

# Make sure MPI is initialized
from .mscclpp_mpi import init_mpi

init_mpi()
