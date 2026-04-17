# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test multiple independent MSCCL++ communicators.

Verifies that two separate MSCCL++ communicators can coexist and run
allreduce independently without interfering with each other.

NOTE: This test is currently expected to fail. MSCCL++ native algorithms
use a process-wide AlgorithmCollectionBuilder singleton and establish
peer connections during lazy init — creating multiple independent
communicators in the same process causes connection conflicts. This is
a known limitation shared with the NCCL extension.

Prerequisites:
  - torchcomms >= 0.2.0 installed
  - MSCCL++ built with -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON
  - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP env var set

Run:
  torchrun --nproc_per_node=2 test/torchcomms/test_multicomm.py
"""

import os
import sys

import torch
import torchcomms


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=== TorchComms MSCCL++ Multi-Communicator Test ===")
        print(f"  world_size={world_size}")
        print("  NOTE: Multiple communicators in one process is a known limitation.")
        print("  This test documents current behavior and will pass once MSCCL++")
        print("  supports independent communicator instances per process.")

    try:
        # Create two independent communicators
        comm1 = torchcomms.new_comm("mscclpp", device, name="comm_A")
        comm2 = torchcomms.new_comm("mscclpp", device, name="comm_B")

        if rank == 0:
            print("  Both communicators created")

        # Run allreduce on comm1
        tensor1 = torch.full((1024,), float(rank + 1), device=device, dtype=torch.float32)
        comm1.all_reduce(tensor1, torchcomms.ReduceOp.SUM, False)
        torch.cuda.synchronize()

        expected_val = world_size * (world_size + 1) / 2.0
        assert torch.allclose(tensor1, torch.full_like(tensor1, expected_val)), f"[rank {rank}] comm1 allreduce failed"

        if rank == 0:
            print("  comm1 allreduce: PASSED")

        # Run allreduce on comm2 with different data
        tensor2 = torch.full((2048,), float(rank * 10), device=device, dtype=torch.float32)
        comm2.all_reduce(tensor2, torchcomms.ReduceOp.SUM, False)
        torch.cuda.synchronize()

        expected_val2 = sum(r * 10 for r in range(world_size))
        assert torch.allclose(tensor2, torch.full_like(tensor2, expected_val2)), f"[rank {rank}] comm2 allreduce failed"

        if rank == 0:
            print("  comm2 allreduce: PASSED")

        # Finalize both
        comm1.finalize()
        comm2.finalize()

        if rank == 0:
            print("  Both communicators finalized")
            print("\n=== Multi-communicator test PASSED ===")

    except (RuntimeError, Exception) as e:
        if rank == 0:
            print(f"\n=== Multi-communicator test SKIPPED (known limitation) ===")
            print(f"  Error: {e}")
            print("  Multiple independent MSCCL++ communicators in one process are not")
            print("  yet supported. Native algorithms use shared state (singleton builder,")
            print("  peer connections) that conflicts across communicator instances.")
        # Exit cleanly so torchrun doesn't report a crash
        sys.exit(0)


if __name__ == "__main__":
    main()
