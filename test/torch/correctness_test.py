# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Collective correctness verification using PyTorch distributed.

Run examples:
  torchrun --nproc_per_node=4 test/torch/correctness_test.py --collective allreduce --nelem 1048576 --dtype fp16
  torchrun --nproc_per_node=4 test/torch/correctness_test.py --collective allgather --dtype bfloat16
  torchrun --nproc_per_node=4 test/torch/correctness_test.py --collective reduce_scatter --dtype float32
"""

from __future__ import annotations
import os
import argparse
import torch
import torch.distributed as dist
from typing import Tuple

_A = 1664525
_C = 1013904223
_MASK = 0xFFFFFFFF
_N_DIFFERENT_FLOAT = 4096


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float", "float32", "f32"}:
        return torch.float32
    if name in {"fp16", "half", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"int32", "i32"}:
        return torch.int32
    raise ValueError(f"Unsupported dtype: {name}")


def _default_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-3, 1e-3
    return 1e-4, 1e-5


def generate_rank_tensor(
    num_elems: int, rank: int, seq: int = 0, device: torch.device | None = None, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate deterministic pseudo-random values in [0,1) (for integer types, scaled & cast)."""
    if device is None:
        device = torch.device("cuda", rank % torch.cuda.device_count())
    seeds = (torch.arange(num_elems, device=device, dtype=torch.int64) + rank + seq) & _MASK
    seeds = (seeds * _A + _C) & _MASK
    base = (seeds.remainder(_N_DIFFERENT_FLOAT).to(torch.float32)) / float(_N_DIFFERENT_FLOAT)
    if dtype.is_floating_point:
        return base.to(dtype)
    # Integer path: scale to 0..(2^31-1) approximately then cast
    return (base * (2**31 - 1)).to(dtype)


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Distributed environment variables not set. Run with torchrun.")
    backend = "nccl"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=local_rank)
    torch.cuda.set_device(local_rank)


def run_allreduce_test(num_elems: int, iters: int, dtype: torch.dtype, rtol: float, atol: float):
    _init_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for iter in range(iters):
        x = generate_rank_tensor(num_elems, rank, seq=iter, dtype=dtype)
        expected = torch.empty_like(x)
        for r in range(world_size):
            t = generate_rank_tensor(num_elems, r, seq=iter, dtype=dtype, device=x.device)
            expected = t if r == 0 else expected.add(t)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        _assert_close(x, expected, f"AllReduce (rank {rank}, iter {iter})", rtol, atol)


def run_allgather_test(num_elems: int, iters: int, dtype: torch.dtype, rtol: float, atol: float):
    _init_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for iter in range(iters):
        inp = generate_rank_tensor(num_elems, rank, seq=iter, dtype=dtype)
        out = torch.empty(world_size * num_elems, dtype=inp.dtype, device=inp.device)
        out_views = [out[i * num_elems : (i + 1) * num_elems] for i in range(world_size)]
        expected = torch.cat(
            [generate_rank_tensor(num_elems, r, seq=iter, dtype=dtype, device=inp.device) for r in range(world_size)],
            dim=0,
        )
        dist.all_gather(out_views, inp)
        _assert_close(out, expected, f"AllGather (rank {rank})", rtol, atol)


def run_reducescatter_test(num_elems: int, iters: int, dtype: torch.dtype, rtol: float, atol: float):
    _init_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for i in range(iters):
        input = generate_rank_tensor(num_elems * world_size, rank, seq=i, dtype=dtype)
        output = torch.empty(num_elems, dtype=input.dtype, device=input.device)
        input_list = list(input.chunk(world_size))

        expected = None
        for r in range(world_size):
            t = generate_rank_tensor(num_elems * world_size, r, seq=i, dtype=dtype, device=output.device)
            expected = t if expected is None else expected.add(t)
        expected = expected.chunk(world_size)[rank]
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
        _assert_close(output, expected, f"ReduceScatter (rank {rank})", rtol, atol)


def _assert_close(result: torch.Tensor, expected: torch.Tensor, context: str, rtol: float, atol: float):
    # Promote for comparison when needed
    if result.dtype != torch.float32:
        result_f = result.to(torch.float32)
    else:
        result_f = result
    if expected.dtype != torch.float32:
        expected_f = expected.to(torch.float32)
    else:
        expected_f = expected
    if not torch.allclose(result_f, expected_f, rtol=rtol, atol=atol):
        max_abs = (result_f - expected_f).abs().max().item()
        rel = max_abs / (expected_f.abs().max().item() + 1e-12)
        raise AssertionError(f"{context} failed: max_abs={max_abs:.3e} rel={rel:.3e} (rtol={rtol} atol={atol})")
    assert torch.isfinite(result_f).all(), f"{context} produced non-finite values"


def main():
    parser = argparse.ArgumentParser(description="MSCCL++ torch CUDA graph collective correctness tester")
    parser.add_argument("--collective", choices=["allreduce", "allgather", "reduce_scatter"], default="allreduce")
    parser.add_argument(
        "--num-elems",
        "--nelem",
        dest="num_elems",
        type=int,
        default=1 << 18,
        help="Elements per rank (or per chunk for reduce_scatter)",
    )
    parser.add_argument(
        "--iters", type=int, default=4, help="Number of collective iterations captured in the CUDA graph"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type: float32|fp16|bfloat16|int32 (only float dtypes fully validated)",
    )
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    rtol, atol = _default_tolerances(dtype)
    if args.rtol is not None:
        rtol = args.rtol
    if args.atol is not None:
        atol = args.atol

    if args.collective == "allreduce":
        run_allreduce_test(args.num_elems, args.iters, dtype, rtol, atol)
    elif args.collective == "allgather":
        run_allgather_test(args.num_elems, args.iters, dtype, rtol, atol)
    elif args.collective == "reduce_scatter":
        run_reducescatter_test(args.num_elems, args.iters, dtype, rtol, atol)
    else:
        raise ValueError("Unknown collective")
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"{args.collective} test passed for dtype={dtype} num_elems={args.num_elems} iters={args.iters}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
