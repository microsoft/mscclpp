# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Collective correctness tests for the MSCCL++ TorchComms backend.

These tests verify that collectives dispatched through torchcomms.new_comm("mscclpp", ...)
produce correct results. Each test creates an MSCCL++ communicator, runs the collective,
checks the output against a reference, and finalizes.

When --sweep is used, tests run across multiple message sizes and dtypes to exercise
both packet (<=1MB) and non-packet (>1MB) algorithm paths.

Prerequisites:
  - torchcomms >= 0.2.0 installed (pip install --pre torchcomms)
  - MSCCL++ built with -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON
  - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP env var pointing to the built _comms_mscclpp .so

Run examples:
  torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --collective allreduce
  torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --collective allreduce --nelem 4194304 --dtype fp16
  torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --all
  torchrun --nproc_per_node=2 test/torchcomms/test_correctness.py --all --sweep
"""

import argparse
import os
import sys

import torch
import torchcomms

# Size sweep: covers packet path (<=1MB), boundary, and non-packet path (>1MB)
SWEEP_NELEMS = [1, 64, 1024, 16384, 262144, 1048576, 4194304]
SWEEP_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float", "float32"}:
        return torch.float32
    if name in {"fp16", "half", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def tolerances(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-3, 1e-3
    return 1e-4, 1e-5


def get_env():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


def make_comm(device, name="test"):
    """Create an MSCCL++ communicator via torchcomms."""
    return torchcomms.new_comm("mscclpp", device, name=name)


def test_allreduce(comm, rank, world_size, device, nelem, dtype):
    """AllReduce SUM: each rank fills tensor with (rank+1), result should be sum(1..N)."""
    tensor = torch.full((nelem,), float(rank + 1), device=device, dtype=dtype)
    expected_val = world_size * (world_size + 1) / 2.0
    expected = torch.full((nelem,), expected_val, device=device, dtype=dtype)

    comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    torch.cuda.synchronize()

    atol, rtol = tolerances(dtype)
    if not torch.allclose(tensor, expected, atol=atol, rtol=rtol):
        max_diff = (tensor - expected).abs().max().item()
        raise AssertionError(
            f"[rank {rank}] allreduce FAILED: max_diff={max_diff}, "
            f"expected={expected_val}, got sample={tensor[0].item()}"
        )


def test_allreduce_inplace(comm, rank, world_size, device, nelem, dtype):
    """AllReduce SUM in-place: verify the same buffer is both input and output."""
    tensor = torch.full((nelem,), float(rank + 1), device=device, dtype=dtype)
    original_ptr = tensor.data_ptr()
    expected_val = world_size * (world_size + 1) / 2.0

    comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    torch.cuda.synchronize()

    if tensor.data_ptr() != original_ptr:
        raise AssertionError(f"[rank {rank}] allreduce in-place: buffer address changed")

    atol, rtol = tolerances(dtype)
    expected = torch.full((nelem,), expected_val, device=device, dtype=dtype)
    if not torch.allclose(tensor, expected, atol=atol, rtol=rtol):
        max_diff = (tensor - expected).abs().max().item()
        raise AssertionError(f"[rank {rank}] allreduce in-place FAILED: max_diff={max_diff}")


def test_allreduce_repeated(comm, rank, world_size, device, nelem, dtype):
    """AllReduce SUM repeated on the same buffer: catches stale context/semaphore bugs."""
    tensor = torch.empty((nelem,), device=device, dtype=dtype)
    for i in range(5):
        tensor.fill_(float(rank + 1) * (i + 1))
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
        torch.cuda.synchronize()

        expected_val = (i + 1) * world_size * (world_size + 1) / 2.0
        atol, rtol = tolerances(dtype)
        expected = torch.full((nelem,), expected_val, device=device, dtype=dtype)
        if not torch.allclose(tensor, expected, atol=atol, rtol=rtol):
            max_diff = (tensor - expected).abs().max().item()
            raise AssertionError(f"[rank {rank}] allreduce repeated iter {i} FAILED: max_diff={max_diff}")


def test_allgather(comm, rank, world_size, device, nelem, dtype):
    """AllGatherSingle: each rank contributes input, output has all ranks concatenated."""
    input_tensor = torch.full((nelem,), float(rank), device=device, dtype=dtype)
    output_tensor = torch.empty(nelem * world_size, device=device, dtype=dtype)

    comm.all_gather_single(output_tensor, input_tensor, False)
    torch.cuda.synchronize()

    for r in range(world_size):
        chunk = output_tensor[r * nelem : (r + 1) * nelem]
        expected = torch.full((nelem,), float(r), device=device, dtype=dtype)
        if not torch.equal(chunk, expected):
            max_diff = (chunk - expected).abs().max().item()
            raise AssertionError(f"[rank {rank}] allgather FAILED at chunk {r}: max_diff={max_diff}")


def test_reducescatter(comm, rank, world_size, device, nelem, dtype):
    """ReduceScatterSingle: SUM-reduce then scatter so each rank gets its chunk."""
    input_tensor = torch.full((nelem * world_size,), float(rank + 1), device=device, dtype=dtype)
    output_tensor = torch.empty(nelem, device=device, dtype=dtype)

    comm.reduce_scatter_single(output_tensor, input_tensor, torchcomms.ReduceOp.SUM, False)
    torch.cuda.synchronize()

    expected_val = world_size * (world_size + 1) / 2.0
    expected = torch.full((nelem,), expected_val, device=device, dtype=dtype)

    atol, rtol = tolerances(dtype)
    if not torch.allclose(output_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (output_tensor - expected).abs().max().item()
        raise AssertionError(
            f"[rank {rank}] reducescatter FAILED: max_diff={max_diff}, "
            f"expected={expected_val}, got sample={output_tensor[0].item()}"
        )


# Maps collective name -> list of (test_func, label) tuples
COLLECTIVE_TESTS = {
    "allreduce": [
        (test_allreduce, "allreduce"),
        (test_allreduce_inplace, "allreduce_inplace"),
        (test_allreduce_repeated, "allreduce_repeated"),
    ],
    "allgather": [
        (test_allgather, "allgather"),
    ],
    "reducescatter": [
        (test_reducescatter, "reducescatter"),
    ],
}


def run_single(comm, rank, world_size, device, collectives, nelem, dtype):
    """Run specified collectives with a single nelem/dtype combination."""
    failed = []
    skipped = []

    for coll_name in collectives:
        for test_func, label in COLLECTIVE_TESTS[coll_name]:
            try:
                test_func(comm, rank, world_size, device, nelem, dtype)
                if rank == 0:
                    print(f"  {label} {dtype} nelem={nelem}: PASSED")
            except RuntimeError as e:
                err_msg = str(e)
                if "No algorithm registered" in err_msg or "No algorithm" in err_msg:
                    skipped.append(label)
                    if rank == 0:
                        print(f"  {label} {dtype} nelem={nelem}: SKIPPED (no algorithm)")
                else:
                    failed.append((label, err_msg))
                    if rank == 0:
                        print(f"  {label} {dtype} nelem={nelem}: FAILED - {err_msg}")
            except Exception as e:
                failed.append((label, str(e)))
                if rank == 0:
                    print(f"  {label} {dtype} nelem={nelem}: FAILED - {e}")

    return failed, skipped


def run_sweep(comm, rank, world_size, device, collectives):
    """Run collectives across multiple sizes and dtypes."""
    all_failed = []
    all_skipped = []
    total = 0

    for dtype in SWEEP_DTYPES:
        for nelem in SWEEP_NELEMS:
            total += len(collectives)
            failed, skipped = run_single(comm, rank, world_size, device, collectives, nelem, dtype)
            all_failed.extend(failed)
            all_skipped.extend(skipped)

    return all_failed, all_skipped, total


def main():
    parser = argparse.ArgumentParser(description="TorchComms MSCCL++ correctness tests")
    parser.add_argument(
        "--collective", type=str, choices=list(COLLECTIVE_TESTS.keys()), help="Which collective to test"
    )
    parser.add_argument("--all", action="store_true", help="Run all collective tests")
    parser.add_argument("--sweep", action="store_true", help="Sweep across multiple sizes and dtypes")
    parser.add_argument("--nelem", type=int, default=1048576, help="Number of elements (default: 1M)")
    parser.add_argument("--dtype", type=str, default="fp32", help="Data type (fp32, fp16, bf16)")
    args = parser.parse_args()

    if not args.collective and not args.all:
        parser.error("Specify --collective <name> or --all")

    rank, world_size, local_rank = get_env()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    collectives = list(COLLECTIVE_TESTS.keys()) if args.all else [args.collective]

    if rank == 0:
        print(f"=== TorchComms MSCCL++ Correctness Tests ===")
        print(f"  world_size={world_size}, sweep={args.sweep}")
        if not args.sweep:
            print(f"  nelem={args.nelem}, dtype={args.dtype}")

    comm = make_comm(device, name="correctness_test")

    if args.sweep:
        failed, skipped, total = run_sweep(comm, rank, world_size, device, collectives)
    else:
        dtype = parse_dtype(args.dtype)
        failed, skipped = run_single(comm, rank, world_size, device, collectives, args.nelem, dtype)

    comm.finalize()

    if rank == 0:
        if failed:
            print(f"\n=== {len(failed)} test(s) FAILED ===")
            for name, err in failed:
                print(f"  {name}: {err}")
            sys.exit(1)
        else:
            skip_msg = f" ({len(skipped)} skipped)" if skipped else ""
            print(f"\n=== All tests PASSED{skip_msg} ===")


if __name__ == "__main__":
    main()
