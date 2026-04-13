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
import sys
import time

_DEBUG = os.environ.get("MSCCLPP_DEBUG_ALLTOALLV", "0") == "1"
import random
import socket
import struct
import pickle
from typing import Callable, List, Tuple

# Must init torch.distributed before importing mscclpp modules
# to set rank/world_size environment variables


def _get_routable_ip() -> str:
    """Get a routable IP address for this host (not 127.0.0.1)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # doesn't actually send data
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def _tcp_broadcast_unique_id(unique_id_bytes: bytes, rank: int, world_size: int,
                              master_addr: str, port: int = 18515) -> bytes:
    """Broadcast UniqueId bytes from rank 0 to all other ranks via TCP."""
    if rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(world_size - 1)
        for _ in range(world_size - 1):
            conn, _ = server.accept()
            length = len(unique_id_bytes)
            conn.sendall(struct.pack("!I", length) + unique_id_bytes)
            conn.close()
        server.close()
        return unique_id_bytes
    else:
        # Retry connection to rank 0
        for attempt in range(120):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((master_addr, port))
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
                if attempt == 119:
                    raise RuntimeError(f"Rank {rank}: failed to connect to {master_addr}:{port}")
        raw_len = b""
        while len(raw_len) < 4:
            raw_len += s.recv(4 - len(raw_len))
        length = struct.unpack("!I", raw_len)[0]
        data = b""
        while len(data) < length:
            data += s.recv(length - len(data))
        s.close()
        return data


def main():
    # Do NOT set CUDA_LAUNCH_BLOCKING=1 — it prevents the proxy thread from
    # delivering IB data while the kernel is running (deadlock).
    
    # Get rank/world from MPI environment
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))
    
    # Set CUDA device — prefer MPI-provided local rank to handle any rank mapping
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",
                     os.environ.get("MPI_LOCALRANKID",
                     os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))))
    torch.cuda.set_device(local_rank)
    
    # Disable UCX in OpenMPI to avoid version mismatch crashes
    os.environ.setdefault("OMPI_MCA_pml", "ob1")
    os.environ.setdefault("OMPI_MCA_btl", "tcp,vader,self")
    
    # Initialize torch.distributed — use NCCL when torch_fn benchmarks are needed,
    # otherwise gloo avoids IB configuration issues on some clusters.
    # Set ALLTOALLV_BACKEND=nccl to enable torch baseline comparison.
    backend = os.environ.get("ALLTOALLV_BACKEND", "gloo")
    # For multi-node: MASTER_ADDR must be set to rank 0's routable IP.
    # Single-node auto-detects; multi-node requires it from the launcher.
    if "MASTER_ADDR" not in os.environ:
        if rank == 0:
            os.environ["MASTER_ADDR"] = _get_routable_ip()
        else:
            # Check if we're single-node (all ranks on same host)
            n_gpus = torch.cuda.device_count()
            if world_size <= n_gpus:
                # Likely single-node – 127.0.0.1 works
                os.environ["MASTER_ADDR"] = "127.0.0.1"
            else:
                raise RuntimeError(
                    f"Rank {rank}: MASTER_ADDR not set for multi-node run "
                    f"(world_size={world_size} > local GPUs={n_gpus}). "
                    f"Set it in your launcher, e.g.:\n"
                    f"  mpirun -x MASTER_ADDR=<node0_ip> -x MASTER_PORT=29500 ..."
                )
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        # Don't use device_id= eager init — it triggers an immediate NCCL allreduce
        # that fails on some platforms (e.g. GB200 with NCCL 2.28.9).
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    if rank == 0:
        print(f"Testing MscclppAlltoAllV with {world_size} ranks")
        print("=" * 60)

    # Import after torch.distributed init
    from mscclpp import (
        Communicator,
        TcpBootstrap,
    )
    from mscclpp._mscclpp import CppUniqueId as UniqueId
    from mscclpp.ext.alltoallv_single import MscclppAlltoAllV
    
    # Create mscclpp communicator with TcpBootstrap
    # Broadcast UniqueId via TCP sockets (avoids mpi4py/UCX issues)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    uid_port = int(os.environ.get("MSCCLPP_UID_PORT", "18515"))
    bootstrap = TcpBootstrap(rank, world_size)
    
    if rank == 0:
        unique_id = bootstrap.create_unique_id()
        uid_bytes = pickle.dumps(unique_id)
    else:
        uid_bytes = b""
    
    uid_bytes = _tcp_broadcast_unique_id(uid_bytes, rank, world_size, master_addr, uid_port)
    if rank != 0:
        unique_id = pickle.loads(uid_bytes)
    
    bootstrap.initialize(unique_id)
    
    # ── Multi-node diagnostics ─────────────────────────────────────────
    import subprocess, platform
    hostname = platform.node()
    n_ranks_per_node = bootstrap.get_n_ranks_per_node()
    is_multi_node = (world_size > n_ranks_per_node)
    
    # Check IB device availability
    try:
        ib_out = subprocess.check_output(["ibv_devinfo", "-l"], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        ib_devices = [l.strip() for l in ib_out.splitlines() if l.strip() and "device" not in l.lower()]
    except Exception:
        ib_devices = []
    
    if rank == 0 and _DEBUG:
        print(f"  Hostname: {hostname}")
        print(f"  nRanksPerNode: {n_ranks_per_node}, isMultiNode: {is_multi_node}")
        print(f"  IB devices: {ib_devices if ib_devices else 'NONE FOUND'}")
        print(f"  MSCCLPP_SOCKET_IFNAME: {os.environ.get('MSCCLPP_SOCKET_IFNAME', '<not set>')}")
        if is_multi_node and not ib_devices:
            print(f"  NOTE: Multi-node detected but no IB devices. "
                  f"GB200 NVSwitch can handle cross-node without IB; "
                  f"on Hopper/Ampere IB is required.")
    # Also print from rank n_ranks_per_node (first rank on node 1) for comparison
    if is_multi_node and rank == n_ranks_per_node and _DEBUG:
        print(f"  [Node 1] Hostname: {hostname}, rank={rank}")
        print(f"  [Node 1] IB devices: {ib_devices if ib_devices else 'NONE FOUND'}")
    # ── End diagnostics ────────────────────────────────────────────────
    
    comm = Communicator(bootstrap)
    
    # Create MscclppAlltoAllV with existing communicator
    alltoallv = MscclppAlltoAllV(communicator=comm)
    
    if rank == 0 and _DEBUG:
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
    
    # ── DEBUG: print tensor sizes before all_to_all_single ──
    if _DEBUG:
        print(f"  [rank {rank}] input_data: numel={input_data.numel()}, shape={input_data.shape}, "
              f"dtype={input_data.dtype}, device={input_data.device}, "
              f"storage_size={input_data.untyped_storage().size()}, "
              f"data_ptr=0x{input_data.data_ptr():x}")
        print(f"  [rank {rank}] world_size={world_size}, chunk_size={chunk_size}, "
              f"expected_total_elems={world_size * chunk_size}, "
              f"scratch_buffer_size={alltoallv._scratch_size}")
        sys.stdout.flush()
    dist.barrier()

    try:
        output = alltoallv.all_to_all_single(input_data)
    except Exception as e:
        print(f"  [rank {rank}] all_to_all_single RAISED: {e}")
        # Try to get the actual CUDA error
        try:
            torch.cuda.synchronize()
        except Exception as e2:
            print(f"  [rank {rank}] CUDA error after all_to_all_single: {e2}")
        sys.stdout.flush()
        raise

    # ── DEBUG: print output tensor sizes ──
    if _DEBUG:
        print(f"  [rank {rank}] output: numel={output.numel()}, shape={output.shape}, "
              f"dtype={output.dtype}, device={output.device}, "
              f"storage_size={output.untyped_storage().size()}, "
              f"data_ptr=0x{output.data_ptr():x}")
        sys.stdout.flush()

    # Verify: each chunk should come from different ranks
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  [rank {rank}] cuda.synchronize FAILED: {e}")
        sys.stdout.flush()
        raise

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

    # Detect whether torch comparison is possible:
    # - Need NCCL backend (gloo doesn't support all_to_all_single)
    # - Need native NCCL (mscclpp shim doesn't implement all collectives)
    use_torch_baseline = (backend == "nccl")
    if use_torch_baseline:
        try:
            tiny_in = torch.zeros(world_size, dtype=torch.float32, device='cuda')
            tiny_out = torch.zeros(world_size, dtype=torch.float32, device='cuda')
            dist.all_to_all_single(tiny_out, tiny_in)
            torch.cuda.synchronize()
        except Exception:
            use_torch_baseline = False
            if rank == 0:
                print("  [INFO] torch all_to_all_single unavailable, skipping torch baseline")

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

    def fmt_size_decimal(nbytes: int) -> str:
        """Format size using decimal MB (÷1000000) to match NCCL EP reporting."""
        if nbytes >= 1000000:
            return f"{nbytes / 1000000:.2f}MB"
        elif nbytes >= 1000:
            return f"{nbytes / 1000:.1f}KB"
        return f"{nbytes}B"

    def print_header():
        if rank == 0:
            if use_torch_baseline:
                print(f"  {'Avg Size':>10s}  "
                      f"{'mscclpp Lat':>12s} {'mscclpp BW':>11s}  "
                      f"{'torch Lat':>10s} {'torch BW':>9s}  "
                      f"{'Speedup':>7s}")
                print(f"  {'-'*10}  "
                      f"{'-'*12} {'-'*11}  "
                      f"{'-'*10} {'-'*9}  "
                      f"{'-'*7}")
            else:
                print(f"  {'Avg Size':>10s}  "
                      f"{'mscclpp Lat':>12s} {'mscclpp BW':>11s}")
                print(f"  {'-'*10}  "
                      f"{'-'*12} {'-'*11}")

    def print_row(size_str, m_lat, m_bw, t_lat=None, t_bw=None):
        if rank == 0:
            if t_bw is not None and t_bw > 0:
                speedup = m_bw / t_bw
                print(f"  {size_str:>10s}  "
                      f"{m_lat:>10.1f}us {m_bw:>9.2f}GB  "
                      f"{t_lat:>8.1f}us {t_bw:>7.2f}GB  "
                      f"{speedup:>6.2f}x")
            else:
                print(f"  {size_str:>10s}  "
                      f"{m_lat:>10.1f}us {m_bw:>9.2f}GB")

    # ── Test 3: Synthetic variable-size sweep ─────────────────────────────
    if rank == 0:
        print("\n[Test 3] Synthetic variable-size benchmark: mscclpp vs torch.dist")
    print_header()

    msg_sizes = [1 << s for s in range(10, 28) if s % 2 == 0]
    msg_sizes.append(128 * 1024 * 1024)

    # Pre-compute max split sizes across all sweep iterations to allocate
    # fixed-size tensors. Reusing the same tensors keeps the NativeAlgorithm
    # context key stable (same ptrs + sizes) and avoids the context cache
    # leak that causes SIGSEGV when stale RegisteredMemory accumulates.
    max_in_elems = 0
    max_out_elems = 0
    sweep_params = []  # (avg_msg_size, in_splits, out_splits)
    for avg_msg_size in msg_sizes:
        random.seed(12345)
        avg_elems = avg_msg_size // 4
        send_matrix = []
        for i in range(world_size):
            row = [max(1, int(avg_elems * (0.5 + random.random()))) for _ in range(world_size)]
            send_matrix.append(row)
        in_splits = send_matrix[rank]
        out_splits = [send_matrix[j][rank] for j in range(world_size)]
        max_in_elems = max(max_in_elems, sum(in_splits))
        max_out_elems = max(max_out_elems, sum(out_splits))
        sweep_params.append((avg_msg_size, in_splits, out_splits))

    # Allocate once at max size
    inp = torch.randn(max_in_elems, dtype=torch.float32, device='cuda')
    out = torch.empty(max_out_elems, dtype=torch.float32, device='cuda')

    for avg_msg_size, in_splits, out_splits in sweep_params:
        n_warmup = 3 if avg_msg_size >= 16 * 1024 * 1024 else 5
        n_iters = 5 if avg_msg_size >= 64 * 1024 * 1024 else (10 if avg_msg_size >= 4 * 1024 * 1024 else 20)

        # Use views into the fixed buffers (same data_ptr → same context key)
        inp_view = inp[:sum(in_splits)]
        out_view = out[:sum(out_splits)]

        m_lat, m_bw = bench_alltoallv(mscclpp_fn, inp_view, out_view, in_splits, out_splits, n_warmup, n_iters)
        if use_torch_baseline:
            try:
                t_lat, t_bw = bench_alltoallv(torch_fn, inp_view, out_view, in_splits, out_splits, n_warmup, n_iters)
                print_row(fmt_size(avg_msg_size), m_lat, m_bw, t_lat, t_bw)
            except Exception as e:
                if rank == 0:
                    print(f"  [WARN] torch baseline failed: {e}")
                    print(f"  [INFO] Disabling torch baseline for remaining sizes")
                use_torch_baseline = False
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                print_row(fmt_size(avg_msg_size), m_lat, m_bw)
        else:
            print_row(fmt_size(avg_msg_size), m_lat, m_bw)

    # ── Test 4: Real MoE workloads ───────────────────────────────────────
    # Token counts from real MoE training runs (rank 0's view, 8 GPUs).
    # Each token = 5120 bytes (hidden_dim=2560, bf16).
    # We use bf16 with 2560 elements/token to match real workload dtype.
    #
    # To build a consistent 8×8 send matrix, we rotate the input_tokens
    # per rank so every rank has the same total send and each NVLink
    # carries a realistically imbalanced load.

    # 10 workloads picked from 3M dispatch records in a real MoE training run,
    # covering the full imbalance spectrum from nearly uniform (1.05×) to
    # extremely skewed (10×).  Each has 32768 total tokens → 167.8MB.
    MOE_WORKLOADS = [
        {"name": "MoE-A",  # imbalance ≈ 1.05×  (near-uniform)
         "input_tokens": [4122, 4115, 4000, 4200, 4126, 4046, 4035, 4124]},
        {"name": "MoE-B",  # imbalance ≈ 1.20×
         "input_tokens": [3770, 4236, 3966, 4046, 4524, 4132, 3825, 4269]},
        {"name": "MoE-C",  # imbalance ≈ 1.35×
         "input_tokens": [4142, 4489, 4563, 3380, 3957, 4133, 3958, 4146]},
        {"name": "MoE-D",  # imbalance ≈ 1.50×  (median)
         "input_tokens": [4232, 3697, 4619, 4788, 4420, 3192, 3971, 3849]},
        {"name": "MoE-E",  # imbalance ≈ 1.75×
         "input_tokens": [4178, 3209, 4678, 5085, 3108, 3365, 5439, 3706]},
        {"name": "MoE-F",  # imbalance ≈ 2.00×
         "input_tokens": [4582, 3903, 3949, 3727, 4823, 5106, 2553, 4125]},
        {"name": "MoE-G",  # imbalance ≈ 2.50×
         "input_tokens": [4036, 4438, 4804, 6180, 2913, 2472, 4105, 3820]},
        {"name": "MoE-H",  # imbalance ≈ 3.50×
         "input_tokens": [3152, 1722, 4406, 4027, 5365, 6027, 4895, 3174]},
        {"name": "MoE-I",  # imbalance ≈ 5.00×
         "input_tokens": [4384, 4194, 7840, 3079, 3460, 3506, 1568, 4737]},
        {"name": "MoE-J",  # imbalance ≈ 10.00× (extreme skew)
         "input_tokens": [2710, 7661, 3354, 4457, 4609, 766, 3423, 5788]},
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
            avg_bytes = total_bytes // world_size
            if use_torch_baseline:
                try:
                    t_lat, t_bw = bench_alltoallv(torch_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
                    print_row(fmt_size(avg_bytes), m_lat, m_bw, t_lat, t_bw)
                except Exception as e:
                    if rank == 0:
                        print(f"  [WARN] torch baseline failed: {e}")
                        print(f"  [INFO] Disabling torch baseline for remaining workloads")
                    use_torch_baseline = False
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    print_row(fmt_size(avg_bytes), m_lat, m_bw)
            else:
                print_row(fmt_size(avg_bytes), m_lat, m_bw)
    else:
        if rank == 0:
            print("\n[Test 4] Skipped (real MoE workloads require exactly 8 ranks)")

    # ── Test 5: NCCL EP Low-Latency equivalent workload ──────────────────
    # Detect if torch baseline is available for Tests 5 & 6
    use_torch_baseline = True
    try:
        tiny_in = torch.zeros(world_size, dtype=torch.float32, device='cuda')
        tiny_out = torch.zeros(world_size, dtype=torch.float32, device='cuda')
        dist.all_to_all_single(tiny_out, tiny_in)
    except Exception:
        use_torch_baseline = False
        if rank == 0:
            print("  [INFO] torch all_to_all_single unavailable, skipping torch baseline in Tests 5/6")

    # Matches the data volume of:
    #   mpirun -np N ep_bench -a ll -t 128 -d 7168
    #
    # ep_bench LL config: 128 tokens/rank, 256 experts, top_k=8,
    # hidden=7168, bf16.
    # Target byte counts: dispatch=14.55 MB, combine=14.55 MB, selections=1015
    #
    # Expert assignment: for each token, generate 256 scores = abs(N(0,1))+1,
    # pick top-8 expert indices. Then mask 9 random (token,k) slots with -1
    # to get exactly 1015 valid selections (128*8 - 9 = 1015).
    # Seed: mt19937(1 + rank).

    LL_NUM_TOKENS = 128      # tokens per rank
    LL_NUM_EXPERTS = 256
    LL_TOP_K = 8
    LL_HIDDEN = 7168         # bf16 elements per token
    LL_NUM_MASKED = 9        # 128*8 - 9 = 1015 valid selections

    if world_size >= 2:
        num_local_experts = LL_NUM_EXPERTS // world_size

        # Replicate LL expert assignment with numpy mt19937
        import numpy as np
        rng = np.random.RandomState(1 + rank)

        # For each token: generate 256 scores, pick top-8 expert indices
        topk_idx = np.zeros((LL_NUM_TOKENS, LL_TOP_K), dtype=np.int64)
        for i in range(LL_NUM_TOKENS):
            scores = np.abs(rng.randn(LL_NUM_EXPERTS)) + 1.0
            top_experts = np.argpartition(scores, -LL_TOP_K)[-LL_TOP_K:]
            topk_idx[i] = top_experts

        # Mask ~10 random positions with -1
        for _ in range(LL_NUM_MASKED):
            ti = rng.randint(0, LL_NUM_TOKENS)
            ki = rng.randint(0, LL_TOP_K)
            topk_idx[ti, ki] = -1

        # Count tokens sent from this rank to each target rank
        send_counts = [0] * world_size
        for i in range(LL_NUM_TOKENS):
            target_ranks_seen = set()
            for k in range(LL_TOP_K):
                eid = topk_idx[i, k]
                if eid >= 0:
                    target_rank = int(eid) // num_local_experts
                    target_ranks_seen.add(target_rank)
            for tr in target_ranks_seen:
                send_counts[tr] += 1

        # Normalize send_counts so each rank sends exactly TARGET_SELECTIONS
        # tokens total, matching ep_bench's reported selections=1015.
        # This ensures total_send_bytes = 1015 × 7168 × 2 = 14,551,040 bytes.
        TARGET_SELECTIONS = 1015
        raw_total = sum(send_counts)
        if raw_total > 0:
            # Scale proportionally, then fix rounding to hit exact target
            scaled = [int(c * TARGET_SELECTIONS / raw_total) for c in send_counts]
            remainder = TARGET_SELECTIONS - sum(scaled)
            # Distribute remainder to largest buckets first
            indices = sorted(range(world_size), key=lambda i: send_counts[i], reverse=True)
            for i in range(remainder):
                scaled[indices[i % world_size]] += 1
            send_counts = scaled

        # Gather 8×8 send matrix
        send_tensor = torch.tensor(send_counts, dtype=torch.int32, device='cuda')
        all_sends = [torch.zeros(world_size, dtype=torch.int32, device='cuda')
                     for _ in range(world_size)]
        dist.all_gather(all_sends, send_tensor)
        send_matrix = [t.cpu().tolist() for t in all_sends]

        in_splits_tokens = send_matrix[rank]
        out_splits_tokens = [send_matrix[j][rank] for j in range(world_size)]

        in_splits = [t * LL_HIDDEN for t in in_splits_tokens]
        out_splits = [t * LL_HIDDEN for t in out_splits_tokens]

        total_send_tokens = sum(in_splits_tokens)
        total_recv_tokens = sum(out_splits_tokens)
        total_send_bytes = sum(in_splits) * 2
        total_recv_bytes = sum(out_splits) * 2

        if rank == 0:
            print(f"\n[Test 5] NCCL EP LL-equivalent workload "
                  f"(tokens={LL_NUM_TOKENS}, experts={LL_NUM_EXPERTS}, "
                  f"top_k={LL_TOP_K}, hidden={LL_HIDDEN}, bf16, {world_size} ranks)")
            print(f"  Rank 0 send tokens: {in_splits_tokens} (total {total_send_tokens})")
            print(f"  Rank 0 recv tokens: {out_splits_tokens} (total {total_recv_tokens})")
            print(f"  Send {total_send_bytes / 1e6:.2f}MB, "
                  f"Recv {total_recv_bytes / 1e6:.2f}MB")
            print(f"  Target: dispatch=14.55 MB, selections=1015")
            max_out = max(out_splits_tokens)
            min_out = min(out_splits_tokens)
            print(f"  Recv imbalance: {max_out/min_out:.2f}x "
                  f"(min={min_out}, max={max_out})")
            print_header()

        inp = torch.randn(sum(in_splits), dtype=torch.bfloat16, device='cuda')
        out = torch.empty(sum(out_splits), dtype=torch.bfloat16, device='cuda')

        n_warmup, n_iters = 10, 50

        m_lat, m_bw = bench_alltoallv(mscclpp_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
        if use_torch_baseline:
            t_lat, t_bw = bench_alltoallv(torch_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
            print_row(fmt_size_decimal(total_send_bytes), m_lat, m_bw, t_lat, t_bw)
        else:
            print_row(fmt_size_decimal(total_send_bytes), m_lat, m_bw)
    else:
        if rank == 0:
            print("\n[Test 5] Skipped (NCCL EP LL-equivalent requires >= 2 ranks)")

    # ── Test 6: NCCL EP High-Throughput equivalent workload ──────────────
    # Matches the data volume of:
    #   mpirun -np N ep_bench -a ht -t 4096 -d 7168
    #
    # ep_bench config: 4096 tokens/rank, 256 experts, top_k=8,
    # hidden=7168, bf16.  Each token is dispatched to top_k=8 experts,
    # so each rank receives ~4096 token-expert pairs from each peer.
    #
    # We replicate the ep_bench expert assignment logic:
    #   srand(rank + 42), for each of 4096 tokens pick a random first_expert
    #   in [0, num_experts), then assign top_k=8 consecutive experts.
    #   target_rank = expert_id // num_local_experts.
    #
    # Target send bytes vary by GPU count (to match ep_bench reports):
    #    8 GPUs: 4096 tokens/rank → 58.72 MB  (no cross-boundary inflation)
    #   16 GPUs: 4317 tokens/rank → 61.88 MB  (matches ep_bench RDMA_send)

    EP_NUM_TOKENS = 4096    # tokens per rank (input)
    EP_NUM_EXPERTS = 256
    EP_TOP_K = 8
    EP_HIDDEN = 7168        # bf16 elements per token

    # Target send tokens per rank, keyed by world_size.
    # 8 GPUs:  top_k=8 = num_local_experts=32, so no boundary-crossing → 4096
    # 16 GPUs: num_local_experts=16, boundary crossing inflates to ~4317
    EP_TARGET_TOKENS = {8: 4096, 16: 4317}

    if world_size >= 2:
        num_local_experts = EP_NUM_EXPERTS // world_size

        # Use C's srand/rand to replicate ep_bench's exact token distribution
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(rank + 42)

        # Count tokens sent from this rank to each target rank.
        # ep_bench dispatches each token to all ranks hosting its top_k experts.
        # A token with experts spanning 2 ranks sends a copy to each.
        send_counts = [0] * world_size
        for i in range(EP_NUM_TOKENS):
            first_expert = libc.rand() % EP_NUM_EXPERTS
            target_ranks_seen = set()
            for k in range(EP_TOP_K):
                expert_id = (first_expert + k) % EP_NUM_EXPERTS
                target_rank = expert_id // num_local_experts
                target_ranks_seen.add(target_rank)
            for tr in target_ranks_seen:
                send_counts[tr] += 1

        # Normalize send_counts to the target for this world_size.
        # For unknown world_size, keep raw counts.
        TARGET_SEND_TOKENS = EP_TARGET_TOKENS.get(world_size, sum(send_counts))
        raw_total = sum(send_counts)
        if raw_total > 0 and raw_total != TARGET_SEND_TOKENS:
            scaled = [int(c * TARGET_SEND_TOKENS / raw_total) for c in send_counts]
            remainder = TARGET_SEND_TOKENS - sum(scaled)
            indices = sorted(range(world_size), key=lambda i: send_counts[i], reverse=True)
            for i in range(abs(remainder)):
                if remainder > 0:
                    scaled[indices[i % world_size]] += 1
                else:
                    scaled[indices[i % world_size]] -= 1
            send_counts = scaled

        # Gather send matrix via allgather
        send_tensor = torch.tensor(send_counts, dtype=torch.int32, device='cuda')
        all_sends = [torch.zeros(world_size, dtype=torch.int32, device='cuda')
                     for _ in range(world_size)]
        dist.all_gather(all_sends, send_tensor)
        send_matrix = [t.cpu().tolist() for t in all_sends]

        in_splits_tokens = send_matrix[rank]
        out_splits_tokens = [send_matrix[j][rank] for j in range(world_size)]

        # Convert tokens to bf16 elements
        in_splits = [t * EP_HIDDEN for t in in_splits_tokens]
        out_splits = [t * EP_HIDDEN for t in out_splits_tokens]

        total_send_tokens = sum(in_splits_tokens)
        total_recv_tokens = sum(out_splits_tokens)
        total_send_bytes = sum(in_splits) * 2
        total_recv_bytes = sum(out_splits) * 2

        target_send_mb = TARGET_SEND_TOKENS * EP_HIDDEN * 2 / 1e6
        target_recv_tokens = world_size * EP_NUM_TOKENS
        target_recv_mb = target_recv_tokens * EP_HIDDEN * 2 / 1e6

        if rank == 0:
            print(f"\n[Test 6] NCCL EP HT-equivalent workload "
                  f"(tokens={EP_NUM_TOKENS}, experts={EP_NUM_EXPERTS}, "
                  f"top_k={EP_TOP_K}, hidden={EP_HIDDEN}, bf16, {world_size} ranks)")
            print(f"  Rank 0 send tokens: {in_splits_tokens} (total {total_send_tokens})")
            print(f"  Rank 0 recv tokens: {out_splits_tokens} (total {total_recv_tokens})")
            print(f"  Send {total_send_bytes / 1e6:.2f}MB, "
                  f"Recv {total_recv_bytes / 1e6:.2f}MB")
            print(f"  Target: RDMA_send={target_send_mb:.2f} MB "
                  f"({TARGET_SEND_TOKENS} tokens), "
                  f"total_recv={target_recv_mb:.2f} MB "
                  f"({target_recv_tokens} tokens)")
            max_out = max(out_splits_tokens)
            min_out = min(out_splits_tokens)
            print(f"  Recv imbalance: {max_out/min_out:.2f}x "
                  f"(min={min_out}, max={max_out})")
            print_header()

        inp = torch.randn(sum(in_splits), dtype=torch.bfloat16, device='cuda')
        out = torch.empty(sum(out_splits), dtype=torch.bfloat16, device='cuda')

        n_warmup, n_iters = 10, 50  # match ep_bench defaults

        m_lat, m_bw = bench_alltoallv(mscclpp_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
        if use_torch_baseline:
            t_lat, t_bw = bench_alltoallv(torch_fn, inp, out, in_splits, out_splits, n_warmup, n_iters)
            print_row(fmt_size_decimal(total_send_bytes), m_lat, m_bw, t_lat, t_bw)
        else:
            print_row(fmt_size_decimal(total_send_bytes), m_lat, m_bw)
    else:
        if rank == 0:
            print("\n[Test 6] Skipped (NCCL EP HT-equivalent requires >= 2 ranks)")

    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
