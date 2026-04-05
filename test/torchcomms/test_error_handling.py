# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Error handling tests for the MSCCL++ TorchComms backend.

Verifies that unsupported operations, invalid arguments, and lifecycle errors
produce clear error messages rather than crashes or hangs.

Prerequisites:
  - torchcomms >= 0.2.0 installed
  - MSCCL++ built with -DMSCCLPP_BUILD_EXT_TORCHCOMMS=ON
  - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP env var set

Run:
  torchrun --nproc_per_node=2 test/torchcomms/test_error_handling.py
"""

import os
import sys
import traceback

import torch
import torchcomms


def get_env():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


def expect_error(name, callable_fn, expected_substring=None):
    """Run callable_fn and verify it raises RuntimeError with expected message."""
    try:
        callable_fn()
        return False, f"Expected RuntimeError but no exception raised"
    except RuntimeError as e:
        msg = str(e)
        if expected_substring and expected_substring not in msg:
            return False, f"Expected '{expected_substring}' in error, got: {msg}"
        return True, msg
    except Exception as e:
        return False, f"Expected RuntimeError, got {type(e).__name__}: {e}"


def test_unsupported_ops(comm, device):
    """Verify unsupported collectives raise clear errors."""
    results = []
    tensor = torch.ones(1024, device=device, dtype=torch.float32)
    tensor_list = [torch.ones(1024, device=device) for _ in range(2)]

    # broadcast
    ok, msg = expect_error("broadcast", lambda: comm.broadcast(tensor, 0, False), "not supported")
    results.append(("broadcast", ok, msg))

    # send
    ok, msg = expect_error("send", lambda: comm.send(tensor, 0, False), "not supported")
    results.append(("send", ok, msg))

    # recv
    ok, msg = expect_error("recv", lambda: comm.recv(tensor, 0, False), "not supported")
    results.append(("recv", ok, msg))

    # barrier
    ok, msg = expect_error("barrier", lambda: comm.barrier(False), "not supported")
    results.append(("barrier", ok, msg))

    return results


def test_unsupported_reduce_op(comm, device):
    """Verify unsupported reduce ops raise clear errors."""
    results = []
    tensor = torch.ones(1024, device=device, dtype=torch.float32)

    # PRODUCT not supported
    for op_name in ["PRODUCT", "MAX"]:
        op = getattr(torchcomms.ReduceOp, op_name, None)
        if op is not None:
            ok, msg = expect_error(
                f"allreduce with {op_name}",
                lambda op=op: comm.all_reduce(tensor.clone(), op, False),
                "does not support",
            )
            results.append((f"allreduce_{op_name}", ok, msg))

    return results


def test_metadata(comm, rank, world_size):
    """Verify metadata accessors return correct values."""
    results = []

    if comm.get_rank() != rank:
        results.append(("get_rank", False, f"Expected {rank}, got {comm.get_rank()}"))
    else:
        results.append(("get_rank", True, ""))

    if comm.get_size() != world_size:
        results.append(("get_size", False, f"Expected {world_size}, got {comm.get_size()}"))
    else:
        results.append(("get_size", True, ""))

    backend_name = comm.get_backend()
    if backend_name != "mscclpp":
        results.append(("get_backend", False, f"Expected 'mscclpp', got '{backend_name}'"))
    else:
        results.append(("get_backend", True, ""))

    return results


def main():
    rank, world_size, local_rank = get_env()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=== TorchComms MSCCL++ Error Handling Tests ===")
        print(f"  world_size={world_size}")

    comm = torchcomms.new_comm("mscclpp", device, name="error_test")

    all_results = []
    all_results.extend(test_unsupported_ops(comm, device))
    all_results.extend(test_unsupported_reduce_op(comm, device))
    all_results.extend(test_metadata(comm, rank, world_size))

    comm.finalize()

    if rank == 0:
        passed = sum(1 for _, ok, _ in all_results if ok)
        failed = [(name, msg) for name, ok, msg in all_results if not ok]

        for name, ok, msg in all_results:
            status = "PASSED" if ok else "FAILED"
            detail = f" - {msg}" if not ok else ""
            print(f"  {name}: {status}{detail}")

        if failed:
            print(f"\n=== {len(failed)} FAILED, {passed} passed ===")
            sys.exit(1)
        else:
            print(f"\n=== All {passed} tests PASSED ===")


if __name__ == "__main__":
    main()
