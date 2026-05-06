# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FSDP2 training test for the MSCCL++ TorchComms backend.

Verifies that MSCCL++ works as the communication backend for FSDP2 training
via TorchComms' DeviceMesh integration. Creates a small transformer-like model,
wraps it with fully_shard(), and runs a training loop comparing FSDP2 results
against a non-sharded reference model.

Prerequisites:
  - torchcomms >= 0.2.0 installed
  - mscclpp-torchcomms installed (python -m pip install ./python/mscclpp_torchcomms)

Run:
  torchrun --nproc_per_node=2 test/torchcomms/test_fsdp2.py
  torchrun --nproc_per_node=8 test/torchcomms/test_fsdp2.py --iterations 20 --dim 128
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torchcomms
from torch.distributed.fsdp import fully_shard, FSDPModule
from torchcomms.device_mesh import init_device_mesh

import mscclpp_torchcomms  # noqa: F401 — auto-registers backend .so path


def get_env():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description="FSDP2 training test with MSCCL++ TorchComms backend")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--dim", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--nlayers", type=int, default=4, help="Number of linear layers")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    rank, world_size, local_rank = get_env()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== FSDP2 + MSCCL++ TorchComms Training Test ===")
        print(f"  world_size={world_size}, dim={args.dim}, nlayers={args.nlayers}")
        print(f"  iterations={args.iterations}, lr={args.lr}")

    # --- Create MSCCL++ communicator 
    # The MSCCL++ backend dlopens libnccl.so.2 internally and transparently
    # falls back to NCCL for collectives without native MSCCL++ algorithms
    # (broadcast, barrier, reduce_scatter on certain configurations).
    comm = torchcomms.new_comm("mscclpp", device, name="fsdp2_test")

    try:
        device_mesh = init_device_mesh(
            mesh_dim_comms=(comm,),
            mesh_dim_names=("main",),
        )
    except TypeError as e:
        # PyTorch < 2.10 may not support _rank kwarg
        if "_rank" in str(e):
            if rank == 0:
                print(f"  SKIPPED: PyTorch version does not support init_device_mesh with _rank")
            comm.finalize()
            return
        raise

    if rank == 0:
        print(f"  DeviceMesh created successfully")

    # --- Build model ---
    torch.manual_seed(42)
    model = nn.Sequential(*[nn.Linear(args.dim, args.dim, bias=False, device=device) for _ in range(args.nlayers)])
    ref_model = copy.deepcopy(model)

    # --- Apply FSDP2 ---
    for layer in model:
        fully_shard(layer, mesh=device_mesh)
        if isinstance(layer, FSDPModule):
            # Use gradient_divide_factor=1.0 so reduce op is SUM (not AVG).
            # MSCCL++ supports SUM and MIN but not AVG.
            layer.set_gradient_divide_factor(1.0)
    fully_shard(model, mesh=device_mesh)

    if rank == 0:
        print(f"  FSDP2 applied to model ({args.nlayers} layers, dim={args.dim})")

    # --- Optimizers ---
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    ref_optim = torch.optim.Adam(ref_model.parameters(), lr=args.lr)

    # --- Training loop ---
    # Use the same input across all ranks (seeded) so the reference model
    # (non-sharded) produces identical results.
    torch.manual_seed(123)
    inp = torch.randn((4, args.dim), device=device)

    for i in range(args.iterations):
        # FSDP2 forward: triggers all_gather to reassemble parameters
        loss = model(inp).sum()
        ref_loss = ref_model(inp).sum()

        # Check forward pass matches
        if not torch.allclose(loss, ref_loss, atol=1e-5, rtol=1e-4):
            if rank == 0:
                print(f"  iteration {i} FAILED: loss mismatch fsdp={loss.item():.6f} ref={ref_loss.item():.6f}")
            comm.finalize()
            sys.exit(1)

        # FSDP2 backward: triggers reduce_scatter for gradient sync
        loss.backward()
        ref_loss.backward()

        optim.step()
        ref_optim.step()
        optim.zero_grad()
        ref_optim.zero_grad()

        if rank == 0 and (i == 0 or (i + 1) % 5 == 0):
            print(f"  iteration {i + 1}/{args.iterations}: loss={loss.item():.6f} PASSED")

    comm.finalize()

    if rank == 0:
        print(f"\n=== FSDP2 training test PASSED ({args.iterations} iterations) ===")


if __name__ == "__main__":
    main()
