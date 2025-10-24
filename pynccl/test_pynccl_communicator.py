import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp

# from simplertransformer.comms import (
#     PyNCCLCommunicator,
#     RedisStore,
#     broadcast_tensor_simple,
#     get_redis,
# )
# from simplertransformer.utils import get_memory_usage

from .pynccl import (
    PyNCCLCommunicator,
    RedisStore,
    broadcast_tensor_simple,
)

from .redis_store import get_redis

REQUIRED_NUM_GPUS = 8
TENSOR_SIZE = 2 * 1024 * 1024 * 1024
DTYPES_TO_TEST = (
    torch.float32,
    torch.uint8,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)
TEST_MEMORY_BUFFER = 1e9


# def get_non_torch_memory() -> int:
#     initial_memory_report = get_memory_usage()
#     return initial_memory_report.total_memory_used - initial_memory_report.torch_reserved


def run_mp_test_pynccl_communicator_success(rank, world_size, uid_str, set_device=True):
    if set_device:
        torch.cuda.set_device(rank)
    r = get_redis()
    store = RedisStore(uid_str, rank=rank, r=r)
    # initial_non_torch_memory = get_non_torch_memory()

    with PyNCCLCommunicator(rank, world_size, store) as comm:
        assert comm.rank == rank
        assert comm.world_size == world_size
        for dtype in DTYPES_TO_TEST:
            for i in range(world_size):
                if i == rank:
                    tensor = torch.full((TENSOR_SIZE,), rank, dtype=dtype, device="cuda")
                else:
                    tensor = torch.empty((TENSOR_SIZE,), dtype=dtype, device="cuda")
                comm.broadcast(tensor, src=i)
                comm.synchronize()
                if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    # FP8 types need special comparison handling
                    expected = torch.full_like(tensor, i, dtype=dtype)
                    assert torch.allclose(tensor.float(), expected.float(), rtol=1e-2, atol=1e-2)
                else:
                    assert torch.all(tensor == i)

    # final_non_torch_memory = get_non_torch_memory()
    # assert final_non_torch_memory <= initial_non_torch_memory + TEST_MEMORY_BUFFER, (
    #     f"Memory usage increased by more than {TEST_MEMORY_BUFFER} bytes. "
    #     f"Initial memory usage: {initial_non_torch_memory}, "
    #     f"Final memory usage: {final_non_torch_memory}.",
    #     "Possible memory leak in PyNCCLCommunicator.",
    # )


def test_pynccl_communicator_success():
    if torch.cuda.device_count() < REQUIRED_NUM_GPUS:
        pytest.skip(f"Requires at least {REQUIRED_NUM_GPUS} GPUs")

    uid_str = str(uuid.uuid4())
    mp.spawn(
        run_mp_test_pynccl_communicator_success,
        args=(REQUIRED_NUM_GPUS, uid_str),
        nprocs=REQUIRED_NUM_GPUS,
        join=True,
    )


def get_rand_tensor(size, dtype, device):
    if dtype == torch.uint8:
        return torch.randint(0, 255, (size,), dtype=dtype, device=device)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Create FP8 tensors from float32 with appropriate scaling
        tensor_f32 = torch.randn(size, dtype=torch.float32, device=device)
        if dtype == torch.float8_e4m3fn:
            tensor_f32 = torch.clamp(tensor_f32 * 100, -400, 400)
        else:  # torch.float8_e5m2
            tensor_f32 = torch.clamp(tensor_f32 * 1000, -10000, 10000)
        return tensor_f32.to(dtype)
    else:
        return torch.randn(size, dtype=dtype, device=device)


def run_mp_test_pynccl_simple_broadcast(rank, world_size, uid_str, seed, set_device=True):
    if set_device:
        torch.cuda.set_device(rank)
    r = get_redis()
    store = RedisStore(uid_str, rank=rank, r=r)

    torch.manual_seed(seed)

    with PyNCCLCommunicator(rank, world_size, store) as comm:
        assert comm.rank == rank
        assert comm.world_size == world_size
        for device in ["cpu", "cuda"]:
            for dtype in DTYPES_TO_TEST:
                # Skip FP8 types on CPU as they're typically GPU-only
                if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and device == "cpu":
                    continue

                size = torch.randint(1, 10000, (1,)).item()
                data_ref = get_rand_tensor(size, dtype=dtype, device=device)
                for i in range(world_size):
                    if i == rank:
                        data = data_ref.clone()
                    else:
                        data = None
                    data = broadcast_tensor_simple(comm, data, src_rank=i)
                    comm.synchronize()
                    assert data.device == data_ref.device
                    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        # FP8 types don't support torch.equal, convert to float32 for comparison
                        assert torch.allclose(data.float(), data_ref.float(), rtol=1e-2, atol=1e-2)
                    else:
                        assert torch.equal(data, data_ref)


def test_pynccl_simple_broadcast():
    if torch.cuda.device_count() < REQUIRED_NUM_GPUS:
        pytest.skip(f"Requires at least {REQUIRED_NUM_GPUS} GPUs")

    uid_str = str(uuid.uuid4())
    seed = 1234
    mp.spawn(
        run_mp_test_pynccl_simple_broadcast,
        args=(REQUIRED_NUM_GPUS, uid_str, seed),
        nprocs=REQUIRED_NUM_GPUS,
        join=True,
    )


def run_mp_test_pynccl_watchdog_timeout(rank, world_size, uid_str, timeout_secs):
    """Test function where only rank 0 participates in broadcast while other ranks don't, forcing a timeout."""
    torch.cuda.set_device(rank)
    r = get_redis()
    store = RedisStore(uid_str, rank=rank, r=r)

    with PyNCCLCommunicator(rank, world_size, store) as comm:
        if rank == 0:
            tensor = torch.ones(1000, dtype=torch.float32, device="cuda")
            comm.broadcast(tensor, src=0, timeout_secs=timeout_secs)
        else:
            time.sleep(timeout_secs + 10)


def test_pynccl_watchdog_timeout():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs")

    uid_str = str(uuid.uuid4())
    timeout_secs = 5.0
    world_size = 2

    with pytest.raises((mp.ProcessRaisedException, mp.ProcessExitedException)):
        mp.spawn(
            run_mp_test_pynccl_watchdog_timeout,
            args=(world_size, uid_str, timeout_secs),
            nprocs=world_size,
            join=True,
        )


def run_mp_test_pynccl_reduce(rank, world_size, uid_str, seed, set_device=True):
    if set_device:
        torch.cuda.set_device(rank)
    r = get_redis()
    store = RedisStore(uid_str, rank=rank, r=r)

    torch.manual_seed(seed)

    with PyNCCLCommunicator(rank, world_size, store) as comm:
        assert comm.rank == rank
        assert comm.world_size == world_size

        for dtype in DTYPES_TO_TEST:
            # It's very hard to do reliable reductions with these dtypes, so support for them is disabled
            if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                continue

            size = torch.randint(1, 1000, (1,)).item()

            for op in ["sum", "prod", "max", "min", "avg"]:
                # Skip prod for uint8 to avoid overflow issues
                if op == "prod" and dtype == torch.uint8:
                    continue

                # Each rank creates its own tensor with rank + 2 as the value (avoid 0 and 1 for prod)
                base_value = rank + 2
                data = torch.full((size,), base_value, dtype=dtype, device="cuda")

                for dst_rank in range(world_size):
                    tensor_copy = data.clone()
                    comm.reduce(tensor_copy, dst=dst_rank, op=op)

                    if rank == dst_rank:
                        # Calculate expected result based on operation
                        expected_val = 0  # Default value
                        if op == "sum":
                            # Sum of (2 + 3 + ... + world_size + 1) = sum from i=2 to world_size+1
                            expected_val = sum(range(2, world_size + 2))
                        elif op == "prod":
                            # Product of (2 * 3 * ... * world_size + 1)
                            expected_val = 1
                            for i in range(2, world_size + 2):
                                expected_val *= i
                        elif op == "max":
                            # Maximum of (2, 3, ..., world_size + 1) = world_size + 1
                            expected_val = world_size + 1
                        elif op == "min":
                            # Minimum of (2, 3, ..., world_size + 1) = 2
                            expected_val = 2
                        elif op == "avg":
                            # Average of (2 + 3 + ... + world_size + 1) / world_size
                            expected_val = sum(range(2, world_size + 2)) / world_size

                        expected_tensor = torch.full(
                            (size,),
                            expected_val,
                            dtype=dtype,
                            device="cuda",
                        )

                        if dtype in (torch.bfloat16,) or op == "avg":
                            # Use relaxed comparison for these dtypes and avg due to precision
                            assert torch.allclose(
                                tensor_copy.float(), expected_tensor.float(), rtol=1e-2, atol=1e-2
                            ), (
                                f"op={op}, rank={rank}, dtype={dtype}, tensor: {tensor_copy.float().mean():.6f}, expected: {expected_tensor.float().mean():.6f}"
                            )
                        else:
                            assert torch.equal(tensor_copy, expected_tensor), (
                                f"op={op}, rank={rank}, dtype={dtype}, tensor: {tensor_copy.float().mean():.6f}, expected: {expected_tensor.float().mean():.6f}"
                            )


def test_pynccl_reduce():
    if torch.cuda.device_count() < REQUIRED_NUM_GPUS:
        pytest.skip(f"Requires at least {REQUIRED_NUM_GPUS} GPUs")

    uid_str = str(uuid.uuid4())
    seed = 1234
    mp.spawn(
        run_mp_test_pynccl_reduce,
        args=(REQUIRED_NUM_GPUS, uid_str, seed),
        nprocs=REQUIRED_NUM_GPUS,
        join=True,
    )
