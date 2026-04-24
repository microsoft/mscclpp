# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Correctness test for FP8 allreduce with different accumulation types.
#
# Verifies that FP8 allreduce with higher-precision accumulation produces
# results at least as accurate as native FP8 accumulation, by comparing
# against a float32 reference.
#
# Usage:
#   mpirun -np 8 pytest python/test/test_fp8_accum.py -v

import cupy as cp
import numpy as np
import pytest

from mscclpp import CommGroup, GpuBuffer, DataType, ReduceOp, is_nvls_supported
from mscclpp.ext import AlgorithmCollectionBuilder
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group

# FP8 E4M3 (hardware) requires SM >= 89 (Ada / Hopper) on NVIDIA GPUs.
# On AMD/ROCm (e.g. MI300X), FP8 is supported natively — no skip needed.
_is_hip = hasattr(cp.cuda.runtime, "is_hip") and cp.cuda.runtime.is_hip
_gcn_arch_name = ""
if _is_hip:
    _gcn_arch_name = cp.cuda.runtime.getDeviceProperties(0).get("gcnArchName", b"")
    if isinstance(_gcn_arch_name, bytes):
        _gcn_arch_name = _gcn_arch_name.decode()
    _gcn_arch_name = _gcn_arch_name.split(":", maxsplit=1)[0]
_is_cdna4 = _gcn_arch_name.startswith("gfx95")
_skip_fp8 = not _is_hip and int(cp.cuda.Device().compute_capability) < 89
pytestmark = pytest.mark.skipif(_skip_fp8, reason="FP8 accum tests require SM >= 89 on CUDA")

# ---------------------------------------------------------------------------
# FP8 E4M3FN helpers (bias=7, no infinity, NaN = exp=15 & mant=7)
# ---------------------------------------------------------------------------


def e4m3fn_to_float(uint8_array):
    """Decode a cupy uint8 array of E4M3FN bit patterns to float32."""
    bits = uint8_array.astype(cp.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7

    # Normal: (-1)^s * 2^(exp-7) * (1 + mant/8)
    normal_val = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), (exp - 7).astype(cp.int32))
    # Subnormal (exp==0): (-1)^s * 2^(-6) * (mant/8)
    subnormal_val = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-6))

    result = cp.where(exp == 0, subnormal_val, normal_val)
    result = cp.where(sign == 1, -result, result)
    # Zero
    result = cp.where((exp == 0) & (mant == 0), cp.float32(0.0), result)
    # NaN: exp==15 & mant==7
    nan_mask = (exp == 15) & (mant == 7)
    result = cp.where(nan_mask, cp.float32(float("nan")), result)
    return result


def float_to_e4m3fn(f32_array, chunk_size=65536):
    """Encode a cupy float32 array to uint8 E4M3FN bit patterns.

    Uses a lookup-table approach: precompute all 128 positive E4M3FN values,
    then find nearest match per element via chunked broadcast comparison.
    """
    # Build lookup table of all 128 positive E4M3FN values (0x00..0x7F)
    all_bytes = cp.arange(128, dtype=cp.uint8)
    all_floats = e4m3fn_to_float(all_bytes)  # (128,) float32
    # Mark NaN entries as inf so they're never selected as nearest
    all_floats = cp.where(cp.isnan(all_floats), cp.float32(float("inf")), all_floats)

    # Clamp input and extract sign
    clamped = f32_array.astype(cp.float32)
    clamped = cp.clip(clamped, -448.0, 448.0)
    signs = (clamped < 0).astype(cp.uint8)
    absval = cp.abs(clamped)

    result = cp.zeros(absval.shape, dtype=cp.uint8)
    n = absval.size
    absval_flat = absval.ravel()
    result_flat = result.ravel()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = absval_flat[start:end]
        # (chunk_size, 128) difference matrix
        diffs = cp.abs(chunk[:, None] - all_floats[None, :])
        result_flat[start:end] = cp.argmin(diffs, axis=1).astype(cp.uint8)

    # Combine with sign bit
    result = result_flat.reshape(absval.shape)
    result = result | (signs << 7)
    # Handle exact zero
    result = cp.where(absval == 0, cp.uint8(0), result)
    return result


# ---------------------------------------------------------------------------
# FP8 E4M3FNUZ helpers (AMD/ROCm; bias=8, max=240, NaN = bits==0x80, no -0)
# ---------------------------------------------------------------------------


def e4m3fnuz_to_float(uint8_array):
    """Decode a cupy uint8 array of E4M3FNUZ bit patterns to float32."""
    bits = uint8_array.astype(cp.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7

    # Normal: (-1)^s * 2^(exp-8) * (1 + mant/8)
    normal_val = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), (exp - 8).astype(cp.int32))
    # Subnormal (exp==0): (-1)^s * 2^(-7) * (mant/8)
    subnormal_val = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-7))

    result = cp.where(exp == 0, subnormal_val, normal_val)
    result = cp.where(sign == 1, -result, result)
    # Zero is only 0x00; the 0x80 encoding is reserved for NaN under fnuz.
    result = cp.where(uint8_array.astype(cp.int32) == 0, cp.float32(0.0), result)
    nan_mask = uint8_array.astype(cp.int32) == 0x80
    result = cp.where(nan_mask, cp.float32(float("nan")), result)
    return result


def float_to_e4m3fnuz(f32_array, chunk_size=65536):
    """Encode a cupy float32 array to uint8 E4M3FNUZ bit patterns.

    Same lookup-table approach as float_to_e4m3fn but using the fnuz table.
    """
    all_bytes = cp.arange(128, dtype=cp.uint8)
    all_floats = e4m3fnuz_to_float(all_bytes)
    all_floats = cp.where(cp.isnan(all_floats), cp.float32(float("inf")), all_floats)

    clamped = f32_array.astype(cp.float32)
    clamped = cp.clip(clamped, -240.0, 240.0)
    signs = (clamped < 0).astype(cp.uint8)
    absval = cp.abs(clamped)

    result = cp.zeros(absval.shape, dtype=cp.uint8)
    n = absval.size
    absval_flat = absval.ravel()
    result_flat = result.ravel()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = absval_flat[start:end]
        diffs = cp.abs(chunk[:, None] - all_floats[None, :])
        result_flat[start:end] = cp.argmin(diffs, axis=1).astype(cp.uint8)

    result = result_flat.reshape(absval.shape)
    result = result | (signs << 7)
    # 0x80 is NaN under fnuz (no negative zero). Collapse any encoding that
    # landed on 0x80 (small negatives quantised to zero magnitude) to 0x00.
    result = cp.where(result == 0x80, cp.uint8(0), result)
    return result


# Platform-aware E4M3 native helpers: ROCm CDNA4 and CUDA use OCP fn; older ROCm uses fnuz.
if _is_hip and not _is_cdna4:
    e4m3_native_to_float = e4m3fnuz_to_float
    float_to_e4m3_native = float_to_e4m3fnuz
    fp8_native_dtype = DataType.float8_e4m3fnuz
else:
    e4m3_native_to_float = e4m3fn_to_float
    float_to_e4m3_native = float_to_e4m3fn
    fp8_native_dtype = DataType.float8_e4m3fn


# ---------------------------------------------------------------------------
# FP8 E4M3B15 helpers (bias=15, max=0.9375, NaN = exp==15 or bits==0x80)
# ---------------------------------------------------------------------------


def e4m3b15_to_float(uint8_array):
    """Decode a cupy uint8 array of E4M3B15 bit patterns to float32."""
    bits = uint8_array.astype(cp.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7

    # Normal: (-1)^s * 2^(exp-15) * (1 + mant/8)
    normal_val = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), (exp - 15).astype(cp.int32))
    # Subnormal (exp==0): (-1)^s * 2^(-14) * (mant/8)
    subnormal_val = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-14))

    result = cp.where(exp == 0, subnormal_val, normal_val)
    result = cp.where(sign == 1, -result, result)
    # Zero
    result = cp.where((exp == 0) & (mant == 0), cp.float32(0.0), result)
    # NaN: exp==15 or negative zero (0x80)
    nan_mask = (exp == 15) | (uint8_array.astype(cp.int32) == 0x80)
    result = cp.where(nan_mask, cp.float32(float("nan")), result)
    return result


def float_to_e4m3b15(f32_array, chunk_size=65536):
    """Encode a cupy float32 array to uint8 E4M3B15 bit patterns.

    Same lookup-table approach as float_to_e4m3fn.
    """
    # Build lookup table of all 128 positive E4M3B15 values (0x00..0x7F)
    all_bytes = cp.arange(128, dtype=cp.uint8)
    all_floats = e4m3b15_to_float(all_bytes)  # (128,) float32
    # Mark NaN entries as inf so they're never selected as nearest
    all_floats = cp.where(cp.isnan(all_floats), cp.float32(float("inf")), all_floats)

    # Clamp input and extract sign
    clamped = f32_array.astype(cp.float32)
    clamped = cp.clip(clamped, -0.9375, 0.9375)
    signs = (clamped < 0).astype(cp.uint8)
    absval = cp.abs(clamped)

    result = cp.zeros(absval.shape, dtype=cp.uint8)
    n = absval.size
    absval_flat = absval.ravel()
    result_flat = result.ravel()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = absval_flat[start:end]
        # (chunk_size, 128) difference matrix
        diffs = cp.abs(chunk[:, None] - all_floats[None, :])
        result_flat[start:end] = cp.argmin(diffs, axis=1).astype(cp.uint8)

    # Combine with sign bit
    result = result_flat.reshape(absval.shape)
    result = result | (signs << 7)
    # Handle exact zero
    result = cp.where(absval == 0, cp.uint8(0), result)
    return result


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def setup_algorithms(mpi_group):
    """Build default algorithms and return (comm_group, algo_map, scratch_buf)."""
    comm_group = CommGroup(mpi_group.comm)
    scratch = GpuBuffer(1 << 27, dtype=cp.uint8)  # 128 MB
    AlgorithmCollectionBuilder.reset()
    builder = AlgorithmCollectionBuilder()
    algorithms = builder.build_default_algorithms(
        scratch_buffer=scratch.data.ptr,
        scratch_buffer_size=scratch.nbytes,
        rank=comm_group.my_rank,
    )
    algo_map = {a.name: a for a in algorithms}
    return comm_group, algo_map, scratch


def run_allreduce(algo, comm_group, buffer, dtype, accum_dtype=None, nblocks=0, nthreads_per_block=0):
    """Run allreduce in-place on buffer and return a copy of the result."""
    ret = algo.execute(
        comm=comm_group.communicator,
        input_buffer=buffer.data.ptr,
        output_buffer=buffer.data.ptr,
        input_size=buffer.nbytes,
        output_size=buffer.nbytes,
        dtype=dtype,
        op=ReduceOp.SUM,
        stream=cp.cuda.get_current_stream().ptr,
        nblocks=nblocks,
        nthreads_per_block=nthreads_per_block,
        symmetric_memory=True,
        accum_dtype=accum_dtype,
    )
    cp.cuda.Device().synchronize()
    assert ret == 0, f"Allreduce failed with error code {ret}"
    return buffer.copy()


# ---------------------------------------------------------------------------
# Test: FP8 E4M3 accumulation correctness
# ---------------------------------------------------------------------------


@parametrize_mpi_groups(8)
@pytest.mark.parametrize(
    "algo_name",
    [
        "default_allreduce_packet",
        "default_allreduce_nvls_packet",
        "default_allreduce_fullmesh",
        "default_allreduce_rsag_zero_copy",
        "default_allreduce_allpair_packet",
    ],
)
@pytest.mark.parametrize("size", [1024, 4096, 16384, 65536, 262144, 1048576])
def test_fp8_e4m3_accum(mpi_group: MpiGroup, algo_name: str, size: int):
    """Verify that FP8 E4M3 allreduce with higher-precision accumulation is at
    least as accurate as native FP8 accumulation, across all algorithm variants."""
    rank = mpi_group.comm.rank
    world_size = mpi_group.comm.size

    comm_group, algo_map, scratch = setup_algorithms(mpi_group)
    if algo_name not in algo_map:
        pytest.skip(f"{algo_name} not available")
    if "nvls" in algo_name and not is_nvls_supported():
        pytest.skip(f"{algo_name} requires NVLS which is not supported on this platform")
    algo = algo_map[algo_name]

    buf = GpuBuffer(size, dtype=cp.uint8)

    # rsag_zero_copy and fullmesh need explicit block/thread counts
    if "rsag" in algo_name:
        nb = max(1, min(32, size // (world_size * 32)))
        nt = 1024
    elif "fullmesh" in algo_name:
        nb = 35
        nt = 512
    else:
        nb = 0
        nt = 0

    accum_configs = [
        ("fp8_native", fp8_native_dtype),
        ("float16", DataType.float16),
        ("float32", DataType.float32),
    ]

    errors = {}
    for accum_label, accum_dtype in accum_configs:
        # Generate deterministic per-rank data (use numpy to avoid hipRAND issues on ROCm)
        rng = np.random.RandomState(42 + rank)
        src_f32 = cp.asarray(rng.randn(size).astype(np.float32))
        src_f32 = cp.clip(src_f32, -240.0, 240.0)
        src_fp8 = float_to_e4m3_native(src_f32)

        # Copy into symmetric buffer
        buf[:] = src_fp8
        cp.cuda.Device().synchronize()

        # Run allreduce
        result = run_allreduce(
            algo,
            comm_group,
            buf,
            dtype=fp8_native_dtype,
            accum_dtype=accum_dtype,
            nblocks=nb,
            nthreads_per_block=nt,
        )
        result_f32 = e4m3_native_to_float(result)

        # Compute float32 reference: sum all ranks' quantized FP8 inputs in float32
        ref_f32 = cp.zeros(size, dtype=cp.float32)
        for r in range(world_size):
            rng_r = np.random.RandomState(42 + r)
            rank_data = cp.asarray(rng_r.randn(size).astype(np.float32))
            rank_data = cp.clip(rank_data, -240.0, 240.0)
            rank_data_fp8 = float_to_e4m3_native(rank_data)
            ref_f32 += e4m3_native_to_float(rank_data_fp8)

        # Compute errors (only on valid, non-NaN entries)
        valid = ~cp.isnan(result_f32) & ~cp.isnan(ref_f32)
        abs_err = cp.abs(result_f32[valid] - ref_f32[valid])
        mean_abs_err = float(cp.mean(abs_err)) if abs_err.size > 0 else 0.0
        errors[accum_label] = mean_abs_err

        # Reset between runs
        algo.reset()

    # Higher-precision accumulation should be at least as accurate as native fp8
    assert (
        errors["float16"] <= errors["fp8_native"] + 1e-6
    ), f"float16 accum ({errors['float16']:.6f}) worse than native ({errors['fp8_native']:.6f})"
    assert (
        errors["float32"] <= errors["fp8_native"] + 1e-6
    ), f"float32 accum ({errors['float32']:.6f}) worse than native ({errors['fp8_native']:.6f})"


# ---------------------------------------------------------------------------
# Test: FP8 E4M3B15 accumulation correctness
# ---------------------------------------------------------------------------


@parametrize_mpi_groups(8)
@pytest.mark.parametrize(
    "algo_name",
    [
        "default_allreduce_packet",
        "default_allreduce_nvls_packet",
        "default_allreduce_rsag_zero_copy",
        "default_allreduce_fullmesh",
        "default_allreduce_allpair_packet",
    ],
)
@pytest.mark.parametrize("size", [1024, 4096, 65536])
def test_fp8_e4m3b15_accum(mpi_group: MpiGroup, algo_name: str, size: int):
    """Verify that FP8 E4M3B15 allreduce with higher-precision accumulation is at
    least as accurate as native E4M3B15 accumulation."""
    rank = mpi_group.comm.rank
    world_size = mpi_group.comm.size

    comm_group, algo_map, scratch = setup_algorithms(mpi_group)
    if algo_name not in algo_map:
        pytest.skip(f"{algo_name} not available")
    if "nvls" in algo_name and not is_nvls_supported():
        pytest.skip(f"{algo_name} requires NVLS which is not supported on this platform")

    algo = algo_map[algo_name]
    buf = GpuBuffer(size, dtype=cp.uint8)

    accum_configs = [
        ("e4m3b15_native", DataType.float8_e4m3b15),
        ("float16", DataType.float16),
        ("float32", DataType.float32),
    ]

    # rsag_zero_copy needs explicit block/thread counts, scaled to data size
    if "rsag" in algo_name:
        nb = max(1, min(32, size // (world_size * 32)))
        nt = 1024
    else:
        nb = 0
        nt = 0

    errors = {}
    for accum_label, accum_dtype in accum_configs:
        # Generate deterministic per-rank random uint8 values in valid e4m3b15 range
        rng = np.random.RandomState(42 + rank)
        raw = cp.asarray(rng.randint(0, 0x78, (size,)).astype(np.uint8))
        signs = cp.asarray(rng.randint(0, 2, (size,)).astype(np.uint8)) << 7
        src_uint8 = raw | signs
        # Fix negative zero -> positive zero
        src_uint8 = cp.where(src_uint8 == 0x80, cp.uint8(0), src_uint8)

        # Copy into symmetric buffer
        buf[:] = src_uint8
        cp.cuda.Device().synchronize()

        # Run allreduce
        result = run_allreduce(
            algo,
            comm_group,
            buf,
            dtype=DataType.float8_e4m3b15,
            accum_dtype=accum_dtype,
            nblocks=nb,
            nthreads_per_block=nt,
        )

        # Decode result
        result_f32 = e4m3b15_to_float(result)

        # Compute float32 reference
        ref_f32 = cp.zeros(size, dtype=cp.float32)
        for r in range(world_size):
            rng_r = np.random.RandomState(42 + r)
            raw_r = cp.asarray(rng_r.randint(0, 0x78, (size,)).astype(np.uint8))
            signs_r = cp.asarray(rng_r.randint(0, 2, (size,)).astype(np.uint8)) << 7
            bits_r = raw_r | signs_r
            bits_r = cp.where(bits_r == 0x80, cp.uint8(0), bits_r)
            ref_f32 += e4m3b15_to_float(bits_r)

        # Clamp reference to e4m3b15 representable range
        ref_f32 = cp.clip(ref_f32, -0.9375, 0.9375)

        # Compute errors (only on valid entries)
        valid = ~cp.isnan(result_f32) & ~cp.isnan(ref_f32)
        abs_err = cp.abs(result_f32[valid] - ref_f32[valid])
        mean_abs_err = float(cp.mean(abs_err)) if abs_err.size > 0 else 0.0
        errors[accum_label] = mean_abs_err

        algo.reset()

    # Higher-precision accumulation should be at least as accurate as native
    assert (
        errors["float16"] <= errors["e4m3b15_native"] + 1e-8
    ), f"float16 accum ({errors['float16']:.8f}) worse than native ({errors['e4m3b15_native']:.8f})"
    assert (
        errors["float32"] <= errors["e4m3b15_native"] + 1e-8
    ), f"float32 accum ({errors['float32']:.8f}) worse than native ({errors['e4m3b15_native']:.8f})"
