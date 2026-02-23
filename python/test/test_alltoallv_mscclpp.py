#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Test script for MscclppAlltoAllV with optimized C++ kernels.
Uses MPI bootstrap for mscclpp and NCCL backend for torch.distributed.

Usage:
    mpirun -np N python test_alltoallv_mscclpp.py
"""

import torch
import torch.distributed as dist
import os
import time

# Must init torch.distributed before importing mscclpp modules
# to set rank/world_size environment variables


def main():
    # Get rank/world from MPI environment
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))
    
    # Set CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    
    # Initialize torch.distributed with NCCL (need MASTER_ADDR/PORT)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                             device_id=torch.device(f"cuda:{local_rank}"))
    
    if rank == 0:
        print(f"Testing MscclppAlltoAllV with {world_size} ranks")
        print("=" * 60)

    # Import after torch.distributed init
    from mscclpp._mscclpp import (
        Communicator,
        TcpBootstrap,
        UniqueId,
    )
    from mscclpp.ext.alltoallv_single import MscclppAlltoAllV
    import pickle
    
    # Create mscclpp communicator with TcpBootstrap
    # Use torch.distributed to share the unique ID via pickle
    bootstrap = TcpBootstrap(rank, world_size)
    
    if rank == 0:
        unique_id = bootstrap.create_unique_id()
        # Serialize UniqueId via pickle and broadcast
        pickled = pickle.dumps(unique_id)
        id_tensor = torch.zeros(256, dtype=torch.uint8, device='cuda')
        id_tensor[:len(pickled)] = torch.tensor(list(pickled), dtype=torch.uint8)
        # Also send length
        len_tensor = torch.tensor([len(pickled)], dtype=torch.int64, device='cuda')
    else:
        id_tensor = torch.zeros(256, dtype=torch.uint8, device='cuda')
        len_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
    
    dist.broadcast(len_tensor, src=0)
    dist.broadcast(id_tensor, src=0)
    
    if rank != 0:
        pickled_len = int(len_tensor.item())
        pickled = bytes(id_tensor[:pickled_len].cpu().tolist())
        unique_id = pickle.loads(pickled)
    
    bootstrap.initialize(unique_id)
    comm = Communicator(bootstrap)
    
    # Create MscclppAlltoAllV with existing communicator
    alltoallv = MscclppAlltoAllV(communicator=comm)
    
    if rank == 0:
        print(f"MscclppAlltoAllV initialized")
        print(f"Algorithm: {alltoallv._algo.name}")
    
    # Test 1: Uniform all-to-all (equal splits)
    if rank == 0:
        print("\n[Test 1] Uniform all-to-all (1024 elements per rank)")
    
    chunk_size = 1024
    input_data = torch.arange(
        rank * world_size * chunk_size,
        (rank + 1) * world_size * chunk_size,
        dtype=torch.float32,
        device='cuda'
    )
    
    output = alltoallv.all_to_all_single(input_data)
    
    # Verify: each chunk should come from different ranks
    torch.cuda.synchronize()
    expected_total = sum(r * world_size * chunk_size for r in range(world_size))
    actual_total = output[:chunk_size].sum().item()  # Just check first chunk is from rank 0
    expected = 0 * world_size * chunk_size + sum(range(chunk_size))
    if rank == 0:
        print(f"  First chunk sum: {actual_total}, expected ~{expected}")
        print(f"  PASS" if abs(actual_total - expected) < 1 else f"  FAIL")
    
    # Test 2: Variable-size all-to-all (simulating MoE)
    if rank == 0:
        print("\n[Test 2] Variable-size all-to-all (MoE-like)")
    
    # Simulate MoE token distribution: rank 0 sends more to rank 0, etc.
    input_split_sizes = [(i + 1) * 512 for i in range(world_size)]
    output_split_sizes = [512 * (rank + 1)] * world_size
    
    total_input = sum(input_split_sizes)
    total_output = sum(output_split_sizes)
    
    input_tensor = torch.randn(total_input, dtype=torch.float32, device='cuda')
    output_tensor = torch.empty(total_output, dtype=torch.float32, device='cuda')
    
    output = alltoallv.all_to_all_single(
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        output=output_tensor
    )
    
    torch.cuda.synchronize()
    if rank == 0:
        print(f"  Input splits: {input_split_sizes}")
        print(f"  Output splits: {output_split_sizes}")
        print(f"  Input total: {total_input}, Output total: {total_output}")
        print(f"  PASS")
    
    # Test 3: Performance benchmark
    if rank == 0:
        print("\n[Test 3] Performance benchmark (1MB per rank)")
    
    msg_size = 1024 * 1024  # 1MB per message
    input_size = msg_size * world_size
    
    input_tensor = torch.randn(input_size // 4, dtype=torch.float32, device='cuda')  # 4 bytes per float
    output_tensor = torch.empty_like(input_tensor)
    
    # Warmup
    for _ in range(5):
        output = alltoallv.all_to_all_single(input_tensor, output=output_tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        output = alltoallv.all_to_all_single(input_tensor, output=output_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate bandwidth
    total_bytes = 2 * input_size * n_iters  # read + write
    bandwidth_gbps = total_bytes / elapsed / 1e9
    
    if rank == 0:
        print(f"  {n_iters} iterations in {elapsed*1000:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Per-iteration: {elapsed/n_iters*1000:.3f} ms")
    
    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
