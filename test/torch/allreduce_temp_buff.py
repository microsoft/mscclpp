# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# run with:
# LD_PRELOAD=<MSCCLPP_REPO>/build/lib/libmscclpp_nccl.so  MSCCLPP_NCCL_SYMMETRIC_MEMORY=false  torchrun --nproc_per_node=8 ./allreduce_temp_buff.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return rank, world, local


class SimpleModel(nn.Module):
    def __init__(self, DIN, DH, DOUT):
        super().__init__()
        self.layer1 = nn.Linear(DIN, DH, bias=False)
        self.layer2 = nn.Linear(DH, DOUT, bias=False)
        self.rank = int(os.environ["RANK"])
        self.eval()

    @torch.no_grad()
    def forward(self, x_bf16: torch.Tensor, out_bf16: torch.Tensor):
        """
        x_bf16:   [B, DIN]    (bf16) input
        out_bf16: [B, DOUT]   (bf16) output buffer
        Returns:
            out_bf16: [B, DOUT] (bf16) output
        """
        out = self.layer1(x_bf16)
        temp = torch.empty_like(out, dtype=torch.bfloat16)
        temp.copy_(out)
        dist.all_reduce(temp, op=dist.ReduceOp.SUM)
        temp2 = temp
        if self.rank == 0:
            # If we are on rank 0, we can use a different temp buffer, make sure msccl++ can handle buffer address changes
            temp2 = torch.empty_like(temp, dtype=torch.bfloat16)
            temp2.copy_(temp)
        dist.all_reduce(temp2, op=dist.ReduceOp.SUM)
        out = self.layer2(temp2)
        out_bf16.copy_(out)
        return out


def main():
    rank, _, local = init_dist()
    device = torch.device("cuda", local)
    torch.set_grad_enabled(False)

    # message size B * DH * dtype_size = 32MB
    B, DIN, DH, DOUT = 2048, 1024, 8192, 8
    dtype = torch.bfloat16

    # Warm up comms
    dist.all_reduce(torch.ones(1, device=device).to(dtype))

    # Build model
    model = SimpleModel(DIN, DH, DOUT).to(device).to(dtype)

    # Static I/O buffers for capture (stable addresses)
    x_bf16 = torch.empty(B, DIN, dtype=dtype, device=device)
    out_bf16 = torch.empty(B, DOUT, dtype=dtype, device=device)

    # Eager warmup
    x_bf16.normal_()
    _ = model(x_bf16, out_bf16)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model(x_bf16, out_bf16)

    for step in range(5):
        x_bf16.normal_()
        g.replay()
        if rank == 0:
            print(f"[step {step}] out_mean={out_bf16.float().mean().item():.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
