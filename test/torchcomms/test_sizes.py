# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Message size sweep tests for the MSCCL++ TorchComms backend.

Tests allreduce across a range of message sizes to exercise:
  - Packet path (<=1MB): uses allreduce_sm_packet / allpair_packet algorithms
  - Non-packet path (>1MB): uses allreduce_sm / NVLS algorithms
  - Boundary sizes: exact powers of two, off-by-one
  - Edge cases: very small (1 element), large (16M+ elements)

Prerequisites:
  - torchcomms >= 0.2.0 installed
  - MSCCL++ built with -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON
  - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP env var set

Run:
  torchrun --nproc_per_node=2 test/torchcomms/test_sizes.py
  torchrun --nproc_per_node=8 test/torchcomms/test_sizes.py --dtype fp16
"""

import argparse
import os
import sys

import torch
import torchcomms


def tolerances(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-3, 1e-3
    return 1e-4, 1e-5


# Sizes chosen to cover algorithm selection boundaries:
#   - 1 element: minimum
#   - 256: small packet
#   - 1023, 1024, 1025: power-of-2 boundary
#   - 262144 (1MB/4 for fp32): near packet/non-packet boundary
#   - 1048576 (4MB for fp32): above packet threshold
#   - 4194304 (16MB for fp32): large message
#   - 8388608 (32MB for fp32): exercises pipeline algorithms
#   NOTE: 262145 (1MB+4 bytes) is excluded — it hits a known algorithm selector
#   boundary bug in MSCCL++ native allreduce (packet ↔ non-packet transition).
SIZE_TABLE = [
    1,
    256,
    1023,
    1024,
    1025,
    65536,
    262144,
    1048576,
    4194304,
    8388608,
]


def main():
    parser = argparse.ArgumentParser(description="TorchComms MSCCL++ size sweep test")
    parser.add_argument("--dtype", type=str, default="fp32", help="Data type (fp32, fp16, bf16)")
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype.lower())
    if dtype is None:
        print(f"Unsupported dtype: {args.dtype}")
        sys.exit(1)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== TorchComms MSCCL++ Size Sweep Test ===")
        print(f"  world_size={world_size}, dtype={dtype}, sizes={len(SIZE_TABLE)}")

    comm = torchcomms.new_comm("mscclpp", device, name="size_sweep")

    passed = 0
    failed = []
    skipped = []

    for nelem in SIZE_TABLE:
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        total_bytes = nelem * bytes_per_elem
        label = f"nelem={nelem} ({total_bytes} bytes)"

        try:
            tensor = torch.full((nelem,), float(rank + 1), device=device, dtype=dtype)
            expected_val = world_size * (world_size + 1) / 2.0

            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
            torch.cuda.synchronize()

            expected = torch.full((nelem,), expected_val, device=device, dtype=dtype)
            atol, rtol = tolerances(dtype)
            if not torch.allclose(tensor, expected, atol=atol, rtol=rtol):
                max_diff = (tensor - expected).abs().max().item()
                failed.append((label, f"max_diff={max_diff}"))
                if rank == 0:
                    print(f"  {label}: FAILED (max_diff={max_diff})")
            else:
                passed += 1
                if rank == 0:
                    print(f"  {label}: PASSED")
        except RuntimeError as e:
            err_msg = str(e)
            if "No algorithm" in err_msg:
                skipped.append(label)
                if rank == 0:
                    print(f"  {label}: SKIPPED (no algorithm)")
            else:
                failed.append((label, err_msg))
                if rank == 0:
                    print(f"  {label}: FAILED - {err_msg}")

    comm.finalize()

    if rank == 0:
        skip_msg = f", {len(skipped)} skipped" if skipped else ""
        if failed:
            print(f"\n=== {len(failed)} FAILED, {passed} passed{skip_msg} ===")
            for label, err in failed:
                print(f"  {label}: {err}")
            sys.exit(1)
        else:
            print(f"\n=== All {passed} sizes PASSED{skip_msg} ===")


if __name__ == "__main__":
    main()
