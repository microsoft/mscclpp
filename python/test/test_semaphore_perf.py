# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Performance comparison: store-based vs red-based semaphore signal.
#
# Measures the latency of signal+wait ping-pong between GPU peers using
# two approaches: (A) the old incOutbound + atomicStore path, and
# (B) the new atomicAdd (PTX red) path.
#
# Usage:
#   mpirun -np 2 pytest python/test/test_semaphore_perf.py -v -s

import cupy as cp
import numpy as np
import os
import pytest

from mscclpp import CommGroup, MemoryDevice2DeviceSemaphore
from mscclpp.utils import KernelBuilder, pack
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group


def create_connection(group, transport="NVLink"):
    from mscclpp import Transport

    remote_nghrs = list(range(group.nranks))
    remote_nghrs.remove(group.my_rank)
    if transport == "NVLink":
        tran = Transport.CudaIpc
    else:
        tran = group.my_ib_device(group.my_rank % 8)
    return group.make_connection(remote_nghrs, tran)


def build_semaphores(group, connections):
    semaphores = group.make_semaphores(connections)
    return {rank: MemoryDevice2DeviceSemaphore(sema) for rank, sema in semaphores.items()}


def pack_semaphore_handles(semaphores, my_rank, nranks):
    """Pack semaphore device handles into a contiguous GPU buffer."""
    first_arg = next(iter(semaphores.values()))
    handle_size = len(first_arg.device_handle().raw)
    device_handles = []
    for rank in range(nranks):
        if rank == my_rank:
            device_handles.append(bytes(handle_size))
        else:
            device_handles.append(semaphores[rank].device_handle().raw)
    return cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)


def run_perf_kernel(kernel, d_semaphores, my_rank, nranks, niters, use_red, d_elapsed):
    """Launch the perf kernel and return elapsed clock cycles per peer."""
    params = pack(d_semaphores, my_rank, nranks, niters, use_red, d_elapsed)
    kernel.launch_kernel(params, 1, nranks, 0, None)
    cp.cuda.Device().synchronize()
    return cp.asnumpy(d_elapsed)


@parametrize_mpi_groups(2, 4, 8)
def test_semaphore_signal_perf(mpi_group: MpiGroup):
    """Compare store-based vs red-based signal latency."""
    group = CommGroup(mpi_group.comm)
    connections = create_connection(group)
    my_rank = group.my_rank
    nranks = group.nranks

    niters = 1000

    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(
        file="d2d_semaphore_perf_test.cu", kernel_name="d2d_semaphore_perf", file_dir=file_dir
    ).get_compiled_kernel()

    d_elapsed = cp.zeros(nranks, dtype=cp.int64)

    # --- Run store-based signal (use_red=0) ---
    semaphores = build_semaphores(group, connections)
    group.barrier()
    d_sem_handles = pack_semaphore_handles(semaphores, my_rank, nranks)
    elapsed_store = run_perf_kernel(kernel, d_sem_handles, my_rank, nranks, niters, 0, d_elapsed)
    group.barrier()

    # --- Run red-based signal (use_red=1) ---
    semaphores2 = build_semaphores(group, connections)
    group.barrier()
    d_sem_handles2 = pack_semaphore_handles(semaphores2, my_rank, nranks)
    d_elapsed[:] = 0
    elapsed_red = run_perf_kernel(kernel, d_sem_handles2, my_rank, nranks, niters, 1, d_elapsed)
    group.barrier()

    # --- Report results ---
    # Get GPU clock rate for conversion to microseconds
    dev = cp.cuda.Device()
    clock_rate_khz = dev.attributes["ClockRate"]  # in kHz

    peer_indices = [r for r in range(nranks) if r != my_rank]

    store_cycles = [elapsed_store[r] for r in peer_indices]
    red_cycles = [elapsed_red[r] for r in peer_indices]

    avg_store = np.mean(store_cycles) / niters
    avg_red = np.mean(red_cycles) / niters

    avg_store_us = avg_store / clock_rate_khz * 1000.0
    avg_red_us = avg_red / clock_rate_khz * 1000.0

    speedup = avg_store / avg_red if avg_red > 0 else float("inf")

    if my_rank == 0:
        print(f"\n{'='*60}")
        print(f"Semaphore Signal Perf (rank {my_rank}, {nranks} ranks, {niters} iters)")
        print(f"{'='*60}")
        print(f"  Store-based:  {avg_store:.1f} cycles/iter  ({avg_store_us:.3f} us)")
        print(f"  Red-based:    {avg_red:.1f} cycles/iter  ({avg_red_us:.3f} us)")
        print(f"  Speedup:      {speedup:.3f}x")
        print(f"{'='*60}")

    # The test passes as long as both variants complete without errors.
    # Performance comparison is informational.
    assert True
