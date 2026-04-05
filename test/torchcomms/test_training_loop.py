# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simulated training loop test for MSCCL++ TorchComms backend.

Verifies that the MSCCL++ backend works correctly in a multi-iteration
training loop pattern: allocate gradient tensors, run allreduce each
iteration, verify correctness.

Prerequisites:
  - torchcomms >= 0.2.0 installed
  - MSCCL++ built with -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON
  - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP env var set

Run:
  torchrun --nproc_per_node=2 test/torchcomms/test_training_loop.py
  torchrun --nproc_per_node=2 test/torchcomms/test_training_loop.py --iterations 50 --nelem 2097152
"""

import argparse
import os
import sys

import torch
import torchcomms


def main():
    parser = argparse.ArgumentParser(description="TorchComms MSCCL++ training loop test")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--nelem", type=int, default=1048576, help="Gradient tensor size")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== TorchComms MSCCL++ Training Loop Test ===")
        print(f"  world_size={world_size}, iterations={args.iterations}, nelem={args.nelem}")

    comm = torchcomms.new_comm("mscclpp", device, name="training")

    for i in range(args.iterations):
        # Simulate gradient computation: each rank produces rank-specific values
        # that change each iteration to catch stale-buffer bugs
        grad = torch.full((args.nelem,), float(rank + 1) * (i + 1), device=device, dtype=torch.float32)

        # AllReduce SUM (gradient synchronization)
        comm.all_reduce(grad, torchcomms.ReduceOp.SUM, False)
        torch.cuda.synchronize()

        # Verify: sum of (r+1)*(i+1) for r in 0..N-1 = (i+1) * N*(N+1)/2
        expected_val = (i + 1) * world_size * (world_size + 1) / 2.0
        if not torch.allclose(grad, torch.full_like(grad, expected_val), atol=1e-4, rtol=1e-5):
            max_diff = (grad - torch.full_like(grad, expected_val)).abs().max().item()
            print(f"[rank {rank}] iteration {i} FAILED: max_diff={max_diff}")
            comm.finalize()
            sys.exit(1)

    comm.finalize()

    if rank == 0:
        print(f"  {args.iterations} iterations: PASSED")
        print(f"\n=== Training loop test PASSED ===")


if __name__ == "__main__":
    main()
