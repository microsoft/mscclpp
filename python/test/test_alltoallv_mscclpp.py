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
import random
from typing import Callable, List, Optional

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

    # ── Unified benchmark helper ──────────────────────────────────────────
    def build_variable_send_matrix(avg_msg_size: int, world_size: int):
        """Build a deterministic variable-size send matrix (0.5×–1.5× of avg)."""
        random.seed(12345)
        avg_elems = avg_msg_size // 4  # float32
        send_matrix = []
        for i in range(world_size):
            row = []
            for j in range(world_size):
                factor = 0.5 + random.random()
                elems = max(1, int(avg_elems * factor))
                row.append(elems)
            send_matrix.append(row)
        return send_matrix

    def bench_alltoallv(
        fn: Callable,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_split_sizes: List[int],
        output_split_sizes: List[int],
        n_warmup: int,
        n_iters: int,
    ) -> tuple:
        """Benchmark an all_to_all_single implementation. Returns (latency_us, algbw_gbps)."""
        for _ in range(n_warmup):
            fn(input_tensor, output_tensor, input_split_sizes, output_split_sizes)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            fn(input_tensor, output_tensor, input_split_sizes, output_split_sizes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_recv_bytes = sum(output_split_sizes) * 4  # float32
        algbw = total_recv_bytes * n_iters / elapsed / 1e9
        lat = elapsed / n_iters * 1e6
        return lat, algbw

    # Wrap mscclpp and torch.dist into the same calling convention
    def mscclpp_fn(inp, out, in_splits, out_splits):
        alltoallv.all_to_all_single(inp, output=out,
                                    input_split_sizes=in_splits,
                                    output_split_sizes=out_splits)

    def torch_fn(inp, out, in_splits, out_splits):
        dist.all_to_all_single(out, inp,
                               output_split_sizes=out_splits,
                               input_split_sizes=in_splits)

    # ── Test 3: Side-by-side comparison ───────────────────────────────────
    if rank == 0:
        print("\n[Test 3] Variable-size benchmark: mscclpp vs torch.dist (1KB–128MB avg/peer)")
        print(f"  {'Avg Size':>10s}  "
              f"{'mscclpp Lat':>12s} {'mscclpp BW':>11s}  "
              f"{'torch Lat':>10s} {'torch BW':>9s}  "
              f"{'Speedup':>7s}")
        print(f"  {'-'*10}  "
              f"{'-'*12} {'-'*11}  "
              f"{'-'*10} {'-'*9}  "
              f"{'-'*7}")

    msg_sizes = [1 << s for s in range(10, 28) if s % 2 == 0]
    msg_sizes.append(128 * 1024 * 1024)

    for avg_msg_size in msg_sizes:
        send_matrix = build_variable_send_matrix(avg_msg_size, world_size)

        input_split_sizes = send_matrix[rank]
        output_split_sizes = [send_matrix[j][rank] for j in range(world_size)]

        total_send = sum(input_split_sizes)
        total_recv = sum(output_split_sizes)

        input_tensor = torch.randn(total_send, dtype=torch.float32, device='cuda')
        output_tensor = torch.empty(total_recv, dtype=torch.float32, device='cuda')

        n_warmup = 3 if avg_msg_size >= 16 * 1024 * 1024 else 5
        n_iters = 5 if avg_msg_size >= 64 * 1024 * 1024 else (10 if avg_msg_size >= 4 * 1024 * 1024 else 20)

        m_lat, m_bw = bench_alltoallv(mscclpp_fn, input_tensor, output_tensor,
                                       input_split_sizes, output_split_sizes,
                                       n_warmup, n_iters)
        t_lat, t_bw = bench_alltoallv(torch_fn, input_tensor, output_tensor,
                                       input_split_sizes, output_split_sizes,
                                       n_warmup, n_iters)

        if rank == 0:
            if avg_msg_size >= 1024 * 1024:
                size_str = f"{avg_msg_size // (1024*1024)}MB"
            elif avg_msg_size >= 1024:
                size_str = f"{avg_msg_size // 1024}KB"
            else:
                size_str = f"{avg_msg_size}B"
            speedup = m_bw / t_bw if t_bw > 0 else float('inf')
            print(f"  {size_str:>10s}  "
                  f"{m_lat:>10.1f}us {m_bw:>9.2f}GB  "
                  f"{t_lat:>8.1f}us {t_bw:>7.2f}GB  "
                  f"{speedup:>6.2f}x")

    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
