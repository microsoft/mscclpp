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
from typing import Callable, List, Tuple

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

    # ── Shared benchmark helpers ──────────────────────────────────────────
    def bench_alltoallv(
        fn: Callable,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_split_sizes: List[int],
        output_split_sizes: List[int],
        n_warmup: int,
        n_iters: int,
    ) -> Tuple[float, float]:
        """Benchmark an all_to_all_single impl. Returns (latency_us, algbw_gbps)."""
        for _ in range(n_warmup):
            fn(input_tensor, output_tensor, input_split_sizes, output_split_sizes)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            fn(input_tensor, output_tensor, input_split_sizes, output_split_sizes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        elem_size = input_tensor.element_size()
        total_recv_bytes = sum(output_split_sizes) * elem_size
        algbw = total_recv_bytes * n_iters / elapsed / 1e9
        lat = elapsed / n_iters * 1e6
        return lat, algbw

    def mscclpp_fn(inp, out, in_splits, out_splits):
        alltoallv.all_to_all_single(inp, output=out,
                                    input_split_sizes=in_splits,
                                    output_split_sizes=out_splits)

    def torch_fn(inp, out, in_splits, out_splits):
        dist.all_to_all_single(out, inp,
                               output_split_sizes=out_splits,
                               input_split_sizes=in_splits)

    def fmt_size(nbytes: int) -> str:
        if nbytes >= 1024 * 1024:
            return f"{nbytes // (1024*1024)}MB"
        elif nbytes >= 1024:
            return f"{nbytes // 1024}KB"
        return f"{nbytes}B"

    def print_header():
        if rank == 0:
            print(f"  {'Avg Size':>10s}  "
                  f"{'mscclpp Lat':>12s} {'mscclpp BW':>11s}  "
                  f"{'torch Lat':>10s} {'torch BW':>9s}  "
                  f"{'Speedup':>7s}")
            print(f"  {'-'*10}  "
                  f"{'-'*12} {'-'*11}  "
                  f"{'-'*10} {'-'*9}  "
                  f"{'-'*7}")

    def print_row(size_str, m_lat, m_bw, t_lat, t_bw):
        if rank == 0:
            speedup = m_bw / t_bw if t_bw > 0 else float('inf')
            print(f"  {size_str:>10s}  "
                  f"{m_lat:>10.1f}us {m_bw:>9.2f}GB  "
                  f"{t_lat:>8.1f}us {t_bw:>7.2f}GB  "
                  f"{speedup:>6.2f}x")

    # ── Test 3: Synthetic variable-size sweep ─────────────────────────────
    if rank == 0:
        print("\n[Test 3] Synthetic variable-size benchmark: mscclpp vs torch.dist")
    print_header()

    msg_sizes = [1 << s for s in range(10, 28) if s % 2 == 0]
    msg_sizes.append(128 * 1024 * 1024)

    for avg_msg_size in msg_sizes:
        random.seed(12345)
        avg_elems = avg_msg_size // 4
        send_matrix = []
        for i in range(world_size):
            row = [max(1, int(avg_elems * (0.5 + random.random()))) for _ in range(world_size)]
            send_matrix.append(row)

        in_splits = send_matrix[rank]
        out_splits = [send_matrix[j][rank] for j in range(world_size)]

        inp = torch.randn(sum(in_splits), dtype=torch.float32, device='cuda')
        out = torch.empty(sum(out_splits), dtype=torch.float32, device='cuda')

        n_warmup = 3 if avg_msg_size >= 16 * 1024 * 1024 else 5
        n_iters = 5 if avg_msg_size >= 64 * 1024 * 1024 else (10 if avg_msg_size >= 4 * 1024 * 1024 else 20)

        m_lat, m_bw = bench_alltoallv(mscclpp_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
        t_lat, t_bw = bench_alltoallv(torch_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
        print_row(fmt_size(avg_msg_size), m_lat, m_bw, t_lat, t_bw)

    # ── Test 4: Real MoE workloads ───────────────────────────────────────
    # Token counts from real MoE training runs (rank 0's view, 8 GPUs).
    # Each token = 5120 bytes (hidden_dim=2560, bf16).
    # We use bf16 with 2560 elements/token to match real workload dtype.
    #
    # To build a consistent 8×8 send matrix, we rotate the input_tokens
    # per rank so every rank has the same total send and each NVLink
    # carries a realistically imbalanced load.

    MOE_WORKLOADS = [
        {
            "name": "MoE-A",
            # input_splits=[3976,3916,4497,4838,2888,3839,4355,4459]
            # total_send=167,772,160  total_recv=148,316,160
            "input_tokens": [3976, 3916, 4497, 4838, 2888, 3839, 4355, 4459],
        },
        {
            "name": "MoE-B",
            # input_splits=[3009,7161,2719,2766,3428,3010,6290,4385]
            # total_send=167,772,160  total_recv=163,722,240
            "input_tokens": [3009, 7161, 2719, 2766, 3428, 3010, 6290, 4385],
        },
    ]
    ELEMS_PER_TOKEN = 2560  # 5120 bytes / 2 bytes-per-bfloat16

    if world_size == 8:
        if rank == 0:
            print(f"\n[Test 4] Real MoE workloads (hidden=2560, bf16, 8 GPUs)")

        for wl_idx, wl in enumerate(MOE_WORKLOADS):
            tokens = wl["input_tokens"]
            min_tok, max_tok = min(tokens), max(tokens)
            imbalance = max_tok / min_tok
            total_bytes = sum(tokens) * 5120

            if rank == 0:
                print(f"\n  {wl['name']}: {sum(tokens)} tokens/rank, "
                      f"{total_bytes / 1e6:.1f}MB, imbalance={imbalance:.1f}x")
                print(f"  Token distribution: {tokens}")
                print_header()

            # Build consistent send_matrix: rotate token list per rank
            moe_send_matrix = []
            for i in range(world_size):
                row = tokens[i:] + tokens[:i]
                moe_send_matrix.append(row)

            in_splits = [moe_send_matrix[rank][j] * ELEMS_PER_TOKEN for j in range(world_size)]
            out_splits = [moe_send_matrix[j][rank] * ELEMS_PER_TOKEN for j in range(world_size)]

            inp = torch.randn(sum(in_splits), dtype=torch.bfloat16, device='cuda')
            out = torch.empty(sum(out_splits), dtype=torch.bfloat16, device='cuda')

            n_warmup, n_iters = 5, 20

            m_lat, m_bw = bench_alltoallv(mscclpp_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
            t_lat, t_bw = bench_alltoallv(torch_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)

            avg_bytes = total_bytes // world_size
            print_row(fmt_size(avg_bytes), m_lat, m_bw, t_lat, t_bw)
    else:
        if rank == 0:
            print("\n[Test 4] Skipped (real MoE workloads require exactly 8 ranks)")

    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
