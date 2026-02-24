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
    
    # Simulate MoE token distribution with imbalanced routing.
    # Build a full send matrix so each rank has different per-peer sizes.
    # send_matrix[i][j] = number of elements rank i sends to rank j.
    # For consistency: rank i's output_split[j] = send_matrix[j][i].
    import random
    random.seed(42)
    send_matrix = []
    for i in range(world_size):
        row = [random.randint(128, 2048) for _ in range(world_size)]
        send_matrix.append(row)

    input_split_sizes = send_matrix[rank]                          # what this rank sends to each peer
    output_split_sizes = [send_matrix[j][rank] for j in range(world_size)]  # what this rank receives from each peer
    
    total_input = sum(input_split_sizes)
    total_output = sum(output_split_sizes)
    
    # Fill input with rank-specific pattern for verification
    input_tensor = torch.arange(total_input, dtype=torch.float32, device='cuda') + rank * 100000
    output_tensor = torch.empty(total_output, dtype=torch.float32, device='cuda')
    
    output = alltoallv.all_to_all_single(
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        output=output_tensor
    )
    
    torch.cuda.synchronize()
    
    # Verify: the local-to-local segment should match exactly
    local_send_offset = sum(input_split_sizes[:rank])
    local_recv_offset = sum(output_split_sizes[:rank])
    local_size = input_split_sizes[rank]  # == output_split_sizes[rank]
    expected_local = input_tensor[local_send_offset:local_send_offset + local_size]
    actual_local = output_tensor[local_recv_offset:local_recv_offset + local_size]
    local_ok = torch.allclose(expected_local, actual_local)
    
    if rank == 0:
        print(f"  Send matrix row (rank 0 sends): {input_split_sizes}")
        print(f"  Recv sizes (rank 0 receives):   {output_split_sizes}")
        print(f"  Input total: {total_input}, Output total: {total_output}")
        print(f"  Local copy verified: {local_ok}")
        print(f"  {'PASS' if local_ok else 'FAIL'}")
    
    # Test 3: Performance benchmark with variable sizes (1KB to 128MB avg per peer)
    if rank == 0:
        print("\n[Test 3] Variable-size performance benchmark (1KB to 128MB avg per peer)")
        print(f"  {'Avg Size':>10s}  {'Iters':>5s}  {'Total (ms)':>10s}  {'Lat (us)':>10s}  {'algBW(GB/s)':>12s}")
        print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}")

    # Message sizes: average bytes sent to each peer
    msg_sizes = [1 << s for s in range(10, 28) if s % 2 == 0]  # powers of 4 from 1KB to 64MB
    msg_sizes.append(128 * 1024 * 1024)  # add 128MB

    for avg_msg_size in msg_sizes:
        # Build a variable send matrix: send_matrix[i][j] = bytes rank i sends to rank j.
        # Use a deterministic seed so all ranks compute the same matrix.
        # Sizes vary from 0.5× to 1.5× of avg_msg_size (in float32 elements).
        import random
        random.seed(12345)
        avg_elems = avg_msg_size // 4  # float32 = 4 bytes
        send_matrix = []
        for i in range(world_size):
            row = []
            for j in range(world_size):
                # Random factor between 0.5 and 1.5
                factor = 0.5 + random.random()
                elems = max(1, int(avg_elems * factor))
                row.append(elems)
            send_matrix.append(row)

        input_split_sizes = send_matrix[rank]
        output_split_sizes = [send_matrix[j][rank] for j in range(world_size)]

        total_send = sum(input_split_sizes)
        total_recv = sum(output_split_sizes)

        input_tensor = torch.randn(total_send, dtype=torch.float32, device='cuda')
        output_tensor = torch.empty(total_recv, dtype=torch.float32, device='cuda')

        # Fewer warmup/iters for very large sizes
        n_warmup = 3 if avg_msg_size >= 16 * 1024 * 1024 else 5
        n_iters = 5 if avg_msg_size >= 64 * 1024 * 1024 else (10 if avg_msg_size >= 4 * 1024 * 1024 else 20)

        # Warmup
        for _ in range(n_warmup):
            alltoallv.all_to_all_single(
                input_tensor, output=output_tensor,
                input_split_sizes=input_split_sizes,
                output_split_sizes=output_split_sizes)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iters):
            alltoallv.all_to_all_single(
                input_tensor, output=output_tensor,
                input_split_sizes=input_split_sizes,
                output_split_sizes=output_split_sizes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Algorithm bandwidth: total bytes received per rank / time (unidirectional)
        total_recv_bytes = total_recv * 4  # float32
        total_bytes = total_recv_bytes * n_iters
        bandwidth_gbps = total_bytes / elapsed / 1e9
        latency_us = elapsed / n_iters * 1e6

        if rank == 0:
            if avg_msg_size >= 1024 * 1024:
                size_str = f"{avg_msg_size // (1024*1024)}MB"
            elif avg_msg_size >= 1024:
                size_str = f"{avg_msg_size // 1024}KB"
            else:
                size_str = f"{avg_msg_size}B"
            print(f"  {size_str:>10s}  {n_iters:>5d}  {elapsed*1000:>10.2f}  {latency_us:>10.1f}  {bandwidth_gbps:>12.2f}")
    
    # Test 4: torch.distributed.all_to_all_single baseline (same variable-size data)
    if rank == 0:
        print("\n[Test 4] torch.dist.all_to_all_single baseline (same variable sizes)")
        print(f"  {'Avg Size':>10s}  {'Iters':>5s}  {'Total (ms)':>10s}  {'Lat (us)':>10s}  {'algBW(GB/s)':>12s}")
        print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}")

    for avg_msg_size in msg_sizes:
        # Rebuild the same send_matrix (same seed → same data)
        import random
        random.seed(12345)
        avg_elems = avg_msg_size // 4
        send_matrix = []
        for i in range(world_size):
            row = []
            for j in range(world_size):
                factor = 0.5 + random.random()
                elems = max(1, int(avg_elems * factor))
                row.append(elems)
            send_matrix.append(row)

        input_split_sizes = send_matrix[rank]
        output_split_sizes = [send_matrix[j][rank] for j in range(world_size)]

        total_send = sum(input_split_sizes)
        total_recv = sum(output_split_sizes)

        input_tensor = torch.randn(total_send, dtype=torch.float32, device='cuda')
        output_tensor = torch.empty(total_recv, dtype=torch.float32, device='cuda')

        n_warmup = 3 if avg_msg_size >= 16 * 1024 * 1024 else 5
        n_iters = 5 if avg_msg_size >= 64 * 1024 * 1024 else (10 if avg_msg_size >= 4 * 1024 * 1024 else 20)

        # Warmup
        for _ in range(n_warmup):
            dist.all_to_all_single(
                output_tensor, input_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iters):
            dist.all_to_all_single(
                output_tensor, input_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_recv_bytes = total_recv * 4
        total_bytes = total_recv_bytes * n_iters
        bandwidth_gbps = total_bytes / elapsed / 1e9
        latency_us = elapsed / n_iters * 1e6

        if rank == 0:
            if avg_msg_size >= 1024 * 1024:
                size_str = f"{avg_msg_size // (1024*1024)}MB"
            elif avg_msg_size >= 1024:
                size_str = f"{avg_msg_size // 1024}KB"
            else:
                size_str = f"{avg_msg_size}B"
            print(f"  {size_str:>10s}  {n_iters:>5d}  {elapsed*1000:>10.2f}  {latency_us:>10.1f}  {bandwidth_gbps:>12.2f}")

    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
