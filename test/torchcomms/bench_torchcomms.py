#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ collective benchmark via TorchComms.

Measures collective latency and bandwidth through the torchcomms path
(torchcomms.new_comm("mscclpp") → TorchCommMSCCLPP → executeCollective).

Supported collectives and their native MSCCL++ algorithms (H100, single-node):

  AllReduce:
    <=16KB      allpair_packet
    16KB-32KB   nvls_packet
    32KB-1MB    packet
    1MB-16MB    nvls_warp_pipeline
    >=16MB      nvls_block_pipeline

  AllGather:
    <=32MB      fullmesh2
    >32MB       fullmesh

Run:
  torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allreduce
  torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allgather
  torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allreduce --warmup 100 --iters 500
  torchrun --nproc_per_node=8 test/torchcomms/bench_torchcomms.py --collective allreduce --dtype fp16
"""

import argparse
import json
import os
import sys

import torch
import torchcomms


def sync_cuda():
    torch.cuda.synchronize()


def cuda_timed(fn, warmup, iters):
    """Time fn() using CUDA events. Returns average microseconds."""
    for _ in range(warmup):
        fn()
    sync_cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    sync_cuda()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync_cuda()

    return (start.elapsed_time(end) * 1000.0) / iters


def format_size(nbytes):
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f}MB"
    elif nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f}KB"
    return f"{nbytes}B"


# --- Curated size tables per collective ---
# Each entry: (nbytes, expected_algorithm_name)

ALLREDUCE_SIZES = [
    (1024, "allpair_packet"),
    (4096, "allpair_packet"),
    (16384, "allpair_packet"),
    (24576, "nvls_packet"),
    (32768, "nvls_packet"),
    (65536, "packet"),
    (262144, "packet"),
    (524288, "packet"),
    (1048576, "packet"),
    (2 * 1024 * 1024, "nvls_warp_pipeline"),
    (4 * 1024 * 1024, "nvls_warp_pipeline"),
    (8 * 1024 * 1024, "nvls_warp_pipeline"),
    (16 * 1024 * 1024, "nvls_block_pipeline"),
    (32 * 1024 * 1024, "nvls_block_pipeline"),
    (64 * 1024 * 1024, "nvls_block_pipeline"),
    (128 * 1024 * 1024, "nvls_block_pipeline"),
    (256 * 1024 * 1024, "nvls_block_pipeline"),
    (512 * 1024 * 1024, "nvls_block_pipeline"),
    (1024 * 1024 * 1024, "nvls_block_pipeline"),
    (2048 * 1024 * 1024, "nvls_block_pipeline"),
]

ALLGATHER_SIZES = [
    (1024, "fullmesh2"),
    (4096, "fullmesh2"),
    (16384, "fullmesh2"),
    (65536, "fullmesh2"),
    (262144, "fullmesh2"),
    (1048576, "fullmesh2"),
    (4 * 1024 * 1024, "fullmesh2"),
    (8 * 1024 * 1024, "fullmesh2"),
    (16 * 1024 * 1024, "fullmesh2"),
    (32 * 1024 * 1024, "fullmesh2"),
    (64 * 1024 * 1024, "fullmesh"),
    (128 * 1024 * 1024, "fullmesh"),
    (256 * 1024 * 1024, "fullmesh"),
    (512 * 1024 * 1024, "fullmesh"),
    (1024 * 1024 * 1024, "fullmesh"),
]

COLLECTIVE_SIZES = {
    "allreduce": ALLREDUCE_SIZES,
    "allgather": ALLGATHER_SIZES,
}


def busbw_factor(collective, world_size):
    """Bus bandwidth correction factor."""
    n = world_size
    if collective == "allreduce":
        return 2.0 * (n - 1) / n
    elif collective == "allgather":
        return (n - 1.0) / n
    return 1.0


def bench_allreduce(comm, tensor, warmup, iters):
    return cuda_timed(
        lambda t=tensor: comm.all_reduce(t, torchcomms.ReduceOp.SUM, False),
        warmup,
        iters,
    )


def bench_allgather(comm, input_tensor, output_tensor, warmup, iters):
    return cuda_timed(
        lambda i=input_tensor, o=output_tensor: comm.all_gather_single(o, i, False),
        warmup,
        iters,
    )


def main():
    parser = argparse.ArgumentParser(description="MSCCL++ TorchComms collective benchmark")
    parser.add_argument("--collective", type=str, required=True, choices=list(COLLECTIVE_SIZES.keys()))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--json-output", type=str, default=None, help="Write results to JSON file")
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype.lower()]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm("mscclpp", device, name="bench")

    element_size = torch.tensor([], dtype=dtype).element_size()
    bw_factor = busbw_factor(args.collective, world_size)
    sizes = COLLECTIVE_SIZES[args.collective]

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        cc = torch.cuda.get_device_capability(device)
        print(f"MSCCL++ TorchComms {args.collective.upper()} Benchmark")
        print(f"  GPU: {gpu_name} (CC {cc[0]}.{cc[1]})")
        print(f"  GPUs: {world_size}, dtype: {args.dtype}")
        print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")
        print()
        print(f"{'Size':<10} {'Time(us)':<12} {'AlgBW(GB/s)':<14} {'BusBW(GB/s)':<14} {'Algorithm':<25}")
        print("-" * 80)

    results = []

    for nbytes, algo_name in sizes:
        nelem = max(1, nbytes // element_size)
        if args.collective == "allgather":
            # nelem is the TOTAL output size; ensure divisible by world_size
            nelem = ((nelem + world_size - 1) // world_size) * world_size
        actual_bytes = nelem * element_size

        # Run the appropriate collective
        time_us = None
        try:
            if args.collective == "allreduce":
                tensor = torch.full((nelem,), float(rank + 1), device=device, dtype=dtype)
                time_us = bench_allreduce(comm, tensor, args.warmup, args.iters)
            elif args.collective == "allgather":
                chunk_size = nelem // world_size
                input_tensor = torch.full((chunk_size,), float(rank), device=device, dtype=dtype)
                output_tensor = torch.empty(nelem, device=device, dtype=dtype)
                time_us = bench_allgather(comm, input_tensor, output_tensor, args.warmup, args.iters)
        except RuntimeError as e:
            if "No algorithm" not in str(e):
                raise

        if time_us is not None and time_us > 0:
            alg_bw = (actual_bytes / time_us) / 1000.0
            bus_bw = alg_bw * bw_factor
        else:
            alg_bw = 0
            bus_bw = 0

        results.append(
            {
                "collective": args.collective,
                "size": actual_bytes,
                "time_us": time_us,
                "algbw_gbps": alg_bw,
                "busbw_gbps": bus_bw,
                "algorithm": algo_name,
            }
        )

        if rank == 0:
            if time_us is not None:
                print(
                    f"{format_size(actual_bytes):<10} {time_us:<12.1f} {alg_bw:<14.1f} {bus_bw:<14.1f} {algo_name:<25}"
                )
            else:
                print(f"{format_size(actual_bytes):<10} {'N/A':<12} {'N/A':<14} {'N/A':<14} {algo_name:<25}")

    comm.finalize()

    if rank == 0:
        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(results, f, indent=2)
        print()


if __name__ == "__main__":
    main()
