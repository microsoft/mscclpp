# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cupy as cp
from mpi4py import MPI

_mscclpp_module = None


def _mscclpp():
    global _mscclpp_module
    if _mscclpp_module is None:
        import mscclpp

        _mscclpp_module = mscclpp
    return _mscclpp_module


@dataclass(frozen=True)
class CorrectnessStats:
    ok: bool
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    mismatches: int = 0
    total: int = 0

    def __bool__(self) -> bool:
        return self.ok


def config_accum_dtype(case: Any) -> Any:
    return case.dtype_spec.accum_dtype or case.dtype_spec.mscclpp_dtype


def fill_case_for_benchmark(case: Any, rank: int) -> None:
    values = _benchmark_input_values(case, rank)
    encoded = _encode_correctness_input(case, values)
    if case.collective == "allreduce":
        case.input[...] = encoded
        return
    case.output.fill(0)
    case.input[...] = encoded


def check_correctness(
    comm: Any,
    case: Any,
    config: Any,
    *,
    niter: int = 1,
) -> CorrectnessStats:
    all_ok = True
    local_max_abs_diff = 0.0
    local_sum_abs_diff = 0.0
    local_mismatches = 0
    local_total = 0
    for iteration in range(niter):
        _fill_case_for_correctness(case, comm.rank, iteration)
        ret = comm.run(case, config)
        cp.cuda.runtime.deviceSynchronize()
        comm.comm_group.barrier()
        if ret != 0:
            all_ok = False
            continue

        expected, stats_expected = _expected_outputs(case, comm.nranks, iteration)
        iter_stats = _local_diff_stats(case, case.output, expected, comm.nranks, stats_expected=stats_expected)
        local_ok = _compare_output(case, case.output, expected, comm.nranks)
        all_ok = all_ok and local_ok
        local_max_abs_diff = max(local_max_abs_diff, iter_stats.max_abs_diff)
        local_sum_abs_diff += iter_stats.mean_abs_diff * iter_stats.total
        local_mismatches += iter_stats.mismatches
        local_total += iter_stats.total

        if not local_ok:
            mismatch = _mismatch_mask(case, case.output, expected, comm.nranks)
            print(
                "not close: "
                f"iter={iteration}, rank={comm.rank}, output={case.output[mismatch][0]}, "
                f"expected={expected[mismatch][0]}, max_abs_diff={iter_stats.max_abs_diff:.6g}, "
                f"mean_abs_diff={iter_stats.mean_abs_diff:.6g}, mismatches={iter_stats.mismatches}/{iter_stats.total}",
                flush=True,
            )

    global_ok = bool(MPI.COMM_WORLD.allreduce(all_ok, op=MPI.LAND))
    global_max_abs_diff = float(MPI.COMM_WORLD.allreduce(local_max_abs_diff, op=MPI.MAX))
    global_sum_abs_diff = float(MPI.COMM_WORLD.allreduce(local_sum_abs_diff, op=MPI.SUM))
    global_mismatches = int(MPI.COMM_WORLD.allreduce(local_mismatches, op=MPI.SUM))
    global_total = int(MPI.COMM_WORLD.allreduce(local_total, op=MPI.SUM))
    global_mean_abs_diff = global_sum_abs_diff / global_total if global_total else 0.0
    return CorrectnessStats(
        ok=global_ok,
        max_abs_diff=global_max_abs_diff,
        mean_abs_diff=global_mean_abs_diff,
        mismatches=global_mismatches,
        total=global_total,
    )


def _fill_case_for_correctness(case: Any, rank: int, iteration: int) -> None:
    values = _correctness_input_values(case, rank, iteration)
    encoded = _encode_correctness_input(case, values)
    if case.collective == "allreduce":
        case.input[...] = encoded
        return
    case.output.fill(0)
    case.input[...] = encoded


def _correctness_input_values(case: Any, rank: int, iteration: int):
    shape = case.input.shape
    rng = cp.random.RandomState(_correctness_seed(rank, iteration))
    return _random_input_values(case, rng, shape)


def _benchmark_input_values(case: Any, rank: int):
    rng = cp.random.RandomState(17_000_003 + rank)
    return _random_input_values(case, rng, case.input.shape)


def _random_input_values(case: Any, rng, shape):
    if case.dtype_spec.fp8_format is not None:
        value_range = _fp8_correctness_input_range(case)
        return rng.uniform(-value_range, value_range, size=shape).astype(cp.float32)
    if case.dtype_spec.cupy_dtype == cp.int32:
        return rng.randint(-1, 2, size=shape).astype(cp.int32)
    if case.dtype_spec.cupy_dtype == cp.uint8:
        return rng.randint(0, 2, size=shape).astype(cp.uint8)
    return rng.uniform(-1.0, 1.0, size=shape).astype(cp.float32)


def _correctness_seed(rank: int, iteration: int) -> int:
    return (iteration + 1) * 1_000_003 + rank


def _fp8_correctness_input_range(case: Any) -> float:
    if case.collective != "allreduce":
        return 1.0
    fp8_format = case.dtype_spec.fp8_format
    if fp8_format is None:
        return 1.0
    return min(1.0, _fp8_max_abs_value(fp8_format) / max(1, MPI.COMM_WORLD.size))


def _encode_correctness_input(case: Any, values):
    if case.dtype_spec.fp8_format is not None:
        # FP8 buffers are stored as uint8 raw bytes, so a normal astype(uint8) cast would not produce FP8 bits.
        return _encode_fp8_values(case.dtype_spec.fp8_format, values)
    return values.astype(case.dtype_spec.cupy_dtype)


def _local_diff_stats(case: Any, output, expected, nranks: int, *, stats_expected=None) -> CorrectnessStats:
    mismatch = _mismatch_mask(case, output, expected, nranks)
    mismatches = int(cp.count_nonzero(mismatch).item())
    total = int(output.size)
    if total == 0:
        return CorrectnessStats(ok=mismatches == 0)

    output_values = _stats_values(case, output)
    expected_values = _stats_values(case, expected) if stats_expected is None else stats_expected.astype(cp.float64)
    abs_diff = cp.abs(output_values - expected_values)
    return CorrectnessStats(
        ok=mismatches == 0,
        max_abs_diff=float(cp.max(abs_diff).item()),
        mean_abs_diff=float(cp.mean(abs_diff).item()),
        mismatches=mismatches,
        total=total,
    )


def _stats_values(case: Any, values):
    # Convert storage buffers into numeric values before computing max/mean diff.
    if case.dtype_spec.fp8_format is not None:
        return _decode_fp8_array(case.dtype_spec.fp8_format, values)
    if cp.issubdtype(values.dtype, cp.floating):
        return values.astype(cp.float64)
    return values.astype(cp.int64)


def _expected_outputs(case: Any, nranks: int, iteration: int):
    if case.collective == "allreduce":
        encoded_inputs = _encoded_rank_inputs(case, nranks, iteration)
        if case.dtype_spec.fp8_format is not None:
            stats_expected = _expected_fp8_accum_values(case, encoded_inputs)
            return _encode_reduced_output(case, stats_expected), stats_expected
        return _encode_reduced_output(case, sum(values.astype(cp.float32) for values in encoded_inputs)), None

    expected = cp.empty_like(case.output)
    chunk = case.input.size
    for rank, values in enumerate(_encoded_rank_inputs(case, nranks, iteration)):
        expected[rank * chunk : (rank + 1) * chunk] = values.reshape(-1)
    return expected, None


def _encoded_rank_inputs(case: Any, nranks: int, iteration: int) -> list[Any]:
    return [_encode_correctness_input(case, _correctness_input_values(case, rank, iteration)) for rank in range(nranks)]


def _expected_fp8_accum_values(case: Any, encoded_inputs: list[Any]):
    fp8_format = case.dtype_spec.fp8_format
    if fp8_format is None:
        raise ValueError("FP8 format is required")

    accum_dtype = config_accum_dtype(case)
    if accum_dtype == _mscclpp().DataType.float16:
        acc = cp.zeros_like(_decode_fp8_array(fp8_format, encoded_inputs[0]), dtype=cp.float16)
        for values in encoded_inputs:
            acc = (acc + _decode_fp8_array(fp8_format, values).astype(cp.float16)).astype(cp.float16)
        return acc.astype(cp.float32)

    if accum_dtype == _mscclpp().DataType.float32:
        acc = cp.zeros_like(_decode_fp8_array(fp8_format, encoded_inputs[0]), dtype=cp.float32)
        for values in encoded_inputs:
            acc += _decode_fp8_array(fp8_format, values).astype(cp.float32)
        return acc

    acc = encoded_inputs[0]
    for values in encoded_inputs[1:]:
        acc = _encode_fp8_values(fp8_format, _decode_fp8_array(fp8_format, acc) + _decode_fp8_array(fp8_format, values))
    return _decode_fp8_array(fp8_format, acc).astype(cp.float32)


def _encode_reduced_output(case: Any, values):
    if case.dtype_spec.fp8_format is not None:
        return _encode_fp8_values(case.dtype_spec.fp8_format, values)
    return values.astype(case.output.dtype)


def _compare_output(case: Any, output, expected, nranks: int) -> bool:
    return bool(cp.all(~_mismatch_mask(case, output, expected, nranks)).item())


def _mismatch_mask(case: Any, output, expected, nranks: int):
    tolerance = _comparison_tolerance(case, nranks)
    if tolerance is None:
        return output != expected
    rtol, atol = tolerance
    return ~cp.isclose(_stats_values(case, output), _stats_values(case, expected), rtol=rtol, atol=atol)


def _comparison_tolerance(case: Any, nranks: int) -> tuple[float, float] | None:
    scale = max(1, nranks) if case.collective == "allreduce" else 1
    if case.dtype_spec.fp8_format is not None:
        accum_dtype = config_accum_dtype(case)
        if accum_dtype == _mscclpp().DataType.float32:
            return None
        atol = _max_fp8_spacing(case.dtype_spec.fp8_format, float(scale))
        if accum_dtype == _mscclpp().DataType.float16:
            return (0.0, atol)
        return (0.0, atol * 2)
    if case.dtype_spec.cupy_dtype == cp.float16:
        return (1.0e-2, 5.0e-4 * scale)
    if case.dtype_spec.cupy_dtype == cp.float32:
        return (1.0e-5 * scale, 1.0e-6 * scale)
    return None


_FP8_TABLES: dict[str, list[tuple[int, float]]] = {}
_FP8_LOOKUP_CACHE: dict[str, tuple[Any, Any]] = {}
_FP8_SPACING_CACHE: dict[tuple[str, float], float] = {}


def _encode_fp8_values(fp8_format: str, values):
    values = values.astype(cp.float32)
    if fp8_format == "e4m3b15":
        return _encode_e4m3b15_values(values)

    # Round each value to the nearest representable FP8 value (ties to even).
    table_values, table_bytes = _fp8_lookup_arrays(fp8_format)
    flat_values = values.ravel()

    # For each value find its two surrounding table entries: lower <= value <= upper.
    upper = cp.clip(cp.searchsorted(table_values, flat_values), 1, table_values.size - 1)
    lower = upper - 1

    # Pick the closer neighbor; on an exact tie pick the one with an even byte.
    dist_to_upper = table_values[upper] - flat_values
    dist_to_lower = flat_values - table_values[lower]
    upper_is_even = (table_bytes[upper] & cp.uint8(1)) == 0
    pick_upper = (dist_to_upper < dist_to_lower) | ((dist_to_upper == dist_to_lower) & upper_is_even)

    return cp.where(pick_upper, table_bytes[upper], table_bytes[lower]).reshape(values.shape)


def _fp8_lookup_arrays(fp8_format: str):
    # Cache a sorted (value -> byte) table per format for fast nearest-value lookup.
    if fp8_format in _FP8_LOOKUP_CACHE:
        return _FP8_LOOKUP_CACHE[fp8_format]

    # Different bytes can decode to the same value (e.g. +0 and -0); keep one byte per value.
    byte_for_value: dict[float, int] = {}
    for byte, value in _FP8_TABLES.setdefault(fp8_format, _build_fp8_table(fp8_format)):
        if value not in byte_for_value or byte < byte_for_value[value]:
            byte_for_value[value] = byte

    table = sorted(byte_for_value.items())
    table_values = cp.asarray([value for value, _ in table], dtype=cp.float32)
    table_bytes = cp.asarray([byte for _, byte in table], dtype=cp.uint8)
    _FP8_LOOKUP_CACHE[fp8_format] = (table_values, table_bytes)
    return _FP8_LOOKUP_CACHE[fp8_format]


def _max_fp8_spacing(fp8_format: str, max_abs_value: float) -> float:
    cache_key = (fp8_format, max_abs_value)
    if cache_key in _FP8_SPACING_CACHE:
        return _FP8_SPACING_CACHE[cache_key]

    values = sorted(
        {
            value
            for _, value in _FP8_TABLES.setdefault(fp8_format, _build_fp8_table(fp8_format))
            if abs(value) <= max_abs_value
        }
    )
    if len(values) < 2:
        spacing = 0.0
    else:
        spacing = max(right - left for left, right in zip(values, values[1:]))
    _FP8_SPACING_CACHE[cache_key] = spacing
    return spacing


def _fp8_max_abs_value(fp8_format: str) -> float:
    return max(abs(value) for _, value in _FP8_TABLES.setdefault(fp8_format, _build_fp8_table(fp8_format)))


def _encode_e4m3b15_values(values):
    # Mirrors the device e4m3b15 encode (gpu_data_types.hpp): clamp the fp16 intermediate
    # to 0x3F80 (+/-1.875) so the max encodable byte is 0x7F/0xFF.
    fp16_bits = values.astype(cp.float16).view(cp.uint16)
    abs_fp16 = fp16_bits & cp.uint16(0x7FFF)
    abs_fp16 = cp.minimum(abs_fp16, cp.uint16(0x3F80)).astype(cp.uint32)
    sign16 = (fp16_bits & cp.uint16(0x8000)).astype(cp.uint32)
    adjusted = abs_fp16 * cp.uint32(2) + cp.uint32(0x0080)
    return (((sign16 | adjusted) >> cp.uint32(8)) & cp.uint32(0xFF)).astype(cp.uint8)


def _build_fp8_table(fp8_format: str) -> list[tuple[int, float]]:
    table = []
    for byte in range(256):
        value = _decode_fp8_scalar(fp8_format, byte)
        if not math.isnan(value):
            table.append((byte, value))
    return table


def _decode_fp8_scalar(fp8_format: str, byte: int) -> float:
    if fp8_format == "e4m3fnuz" and byte == 0x80:
        return float("nan")
    sign = -1.0 if byte & 0x80 else 1.0
    return sign * _decode_fp8_positive(fp8_format, byte & 0x7F)


def _decode_fp8_positive(fp8_format: str, byte: int) -> float:
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7
    if fp8_format == "e4m3fn" and exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0 and mant == 0:
        return 0.0
    if fp8_format == "e4m3fn":
        return math.ldexp(mant / 8.0, -6) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 7)
    if fp8_format == "e4m3fnuz":
        return math.ldexp(mant / 8.0, -7) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 8)
    if fp8_format == "e4m3b15":
        return math.ldexp(mant / 8.0, -14) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 15)
    raise ValueError(f"Unknown FP8 format: {fp8_format}")


def _decode_fp8_array(fp8_format: str, values):
    bits = values.astype(cp.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7

    if fp8_format == "e4m3fn":
        subnormal = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-6))
        normal = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), exp.astype(cp.int32) - 7)
        decoded = cp.where(exp == 0, subnormal, normal)
        decoded = cp.where((exp == 0xF) & (mant == 0x7), cp.nan, decoded)
    elif fp8_format == "e4m3fnuz":
        subnormal = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-7))
        normal = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), exp.astype(cp.int32) - 8)
        decoded = cp.where(exp == 0, subnormal, normal)
    elif fp8_format == "e4m3b15":
        subnormal = cp.ldexp(mant.astype(cp.float32) / cp.float32(8.0), cp.int32(-14))
        normal = cp.ldexp(cp.float32(1.0) + mant.astype(cp.float32) / cp.float32(8.0), exp.astype(cp.int32) - 15)
        decoded = cp.where(exp == 0, subnormal, normal)
    else:
        raise ValueError(f"Unknown FP8 format: {fp8_format}")

    result = cp.where(sign == 1, -decoded, decoded)
    if fp8_format == "e4m3fnuz":
        result = cp.where(bits == 0x80, cp.float32(float("nan")), result)
    return result
