#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Token-major LL dispatch+combine round-trip correctness test.

Identity round-trip: dispatch each token to its top-k expert rows, then combine
weight-reduces them. With an identity "GEMM" (the dispatched token is its own
expert output), combine must reconstruct:

    output[t] = sum_slot weight[t, slot] * x[t]  =  x[t] * sum(weights[t, :])

Run under mpirun (4 ranks/node). Prints PASS/FAIL per rank; rank 0 aggregates.
"""

from __future__ import annotations

import os
import torch
from mpi4py import MPI
from mscclpp import CommGroup
import mscclpp.ep as ep


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    local_rank = int(os.environ.get("MSCCLPP_EP_LOCAL_WORLD_SIZE", "4"))
    torch.cuda.set_device(rank % local_rank)

    num_tokens = 128
    hidden = 7168
    num_experts = 256
    num_topk = 8
    num_local_experts = num_experts // num_ranks

    torch.manual_seed(1234 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()

    ep_group = CommGroup(mpi_comm=comm)
    moe = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        topk=num_topk,
        max_tokens_per_rank=num_tokens,
        mode=ep.MoEMode.LOW_LATENCY,
        low_latency_combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
        output_layout=ep.DispatchLayout.TOKEN_MAJOR,
    )
    assert moe.is_available()
    if rank == 0:
        print(
            f"[cfg] TOKEN_MAJOR round-trip: ranks={num_ranks} tokens={num_tokens} "
            f"hidden={hidden} experts={num_experts} topk={num_topk}",
            flush=True,
        )

    # Dispatch (identity expert output = the dispatched token buffer itself).
    dispatch_out, handle = moe.dispatch(x, topk_idx, topk_weights, output_buffer=None)
    expert_output = moe.get_expert_output_buffer()
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    moe.combine(expert_output, handle, out=out)
    torch.cuda.synchronize()

    # Expected: x[t] * sum(weights[t, :]) over all top-k slots (every slot valid).
    weight_sum = topk_weights.sum(dim=1, keepdim=True)  # [num_tokens, 1]
    expected = (x.float() * weight_sum).to(torch.bfloat16)

    got = out.float()
    exp = expected.float()
    abs_err = (got - exp).abs()
    denom = exp.abs().clamp_min(1e-3)
    rel = (abs_err / denom).max().item()
    max_abs = abs_err.max().item()
    ok = rel < 5e-2  # bf16 tolerance

    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
    max_rel = comm.allreduce(rel, op=MPI.MAX)
    max_abs_all = comm.allreduce(max_abs, op=MPI.MAX)
    if rank == 0:
        status = "PASS" if all_ok else "FAIL"
        print(f"[token-major round-trip] {status}: max_rel_err={max_rel:.4e} max_abs_err={max_abs_all:.4e}", flush=True)


if __name__ == "__main__":
    main()
