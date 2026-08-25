# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace

import cupy as cp

from mscclpp_benchmark.correctness import (
    _comparison_tolerance,
    _decode_bfloat16_array,
    _encode_bfloat16_values,
    _encode_correctness_input,
    _stats_values,
)


def test_allgather_requires_exact_match():
    case = SimpleNamespace(
        collective="allgather",
        dtype_spec=SimpleNamespace(name="float16", fp8_format=None, cupy_dtype=cp.float16),
    )

    assert _comparison_tolerance(case, 8) is None


def test_bfloat16_round_to_nearest_even():
    values = cp.asarray([1.0, -2.5, 1.00390625, 1.01171875], dtype=cp.float32)

    encoded = _encode_bfloat16_values(values)

    cp.testing.assert_array_equal(encoded, cp.asarray([0x3F80, 0xC020, 0x3F80, 0x3F82], dtype=cp.uint16))


def test_bfloat16_raw_storage_round_trip():
    case = SimpleNamespace(dtype_spec=SimpleNamespace(name="bfloat16", fp8_format=None, cupy_dtype=cp.uint16))
    values = cp.asarray([-1.0, -0.25, 0.5, 1.0], dtype=cp.float32)

    encoded = _encode_correctness_input(case, values)

    assert encoded.dtype == cp.uint16
    cp.testing.assert_array_equal(_decode_bfloat16_array(encoded), values)
    cp.testing.assert_array_equal(_stats_values(case, encoded), values)
