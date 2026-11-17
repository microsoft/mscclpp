# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from types import SimpleNamespace

import numpy as np

from mscclpp_benchmark.correctness import (
    _comparison_tolerance,
    _decode_bfloat16_array,
    _decode_fp8_array,
    _decode_fp8_scalar,
    _encode_bfloat16_values,
    _encode_correctness_input,
    _encode_fp8_values,
    _stats_values,
)
from mscclpp_benchmark.tuning_config import HardwareProfile, TunedConfig, TunedConfigStore


def test_allgather_requires_exact_match():
    case = SimpleNamespace(
        collective="allgather",
        dtype_spec=SimpleNamespace(name="float16", fp8_format=None, storage_dtype=np.float16),
    )

    assert _comparison_tolerance(case, 8) is None


def test_bfloat16_round_to_nearest_even():
    values = np.asarray([1.0, -2.5, 1.00390625, 1.01171875], dtype=np.float32)

    encoded = _encode_bfloat16_values(values)

    np.testing.assert_array_equal(encoded, np.asarray([0x3F80, 0xC020, 0x3F80, 0x3F82], dtype=np.uint16))


def test_bfloat16_raw_storage_round_trip():
    case = SimpleNamespace(dtype_spec=SimpleNamespace(name="bfloat16", fp8_format=None, storage_dtype=np.uint16))
    values = np.asarray([-1.0, -0.25, 0.5, 1.0], dtype=np.float32)

    encoded = _encode_correctness_input(case, values)

    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(_decode_bfloat16_array(encoded), values)
    np.testing.assert_array_equal(_stats_values(case, encoded), values)


def test_fp8_numpy_decoders_and_finite_round_trips():
    bits = np.arange(256, dtype=np.uint8)
    for fmt in ("e4m3fn", "e4m3fnuz", "e4m3b15"):
        decoded = _decode_fp8_array(fmt, bits)
        expected = np.asarray([_decode_fp8_scalar(fmt, int(byte)) for byte in bits])
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=0, equal_nan=True)
        finite = decoded[~np.isnan(decoded)]
        encoded = _encode_fp8_values(fmt, finite)
        np.testing.assert_array_equal(_decode_fp8_array(fmt, encoded), finite)


def test_selects_dtype_specific_configs_with_duplicate_sizes():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "sku": "MI300X",
                    "scale": 8,
                    "collectives": {
                        "allreduce": [
                            {
                                "message_size": 1024,
                                "algorithm": "bf16-small",
                                "dtype": "bfloat16",
                                "accum": "bfloat16",
                            },
                            {
                                "message_size": 2048,
                                "algorithm": "bf16-large",
                                "dtype": "bfloat16",
                                "accum": "bfloat16",
                            },
                            {
                                "message_size": 1024,
                                "algorithm": "fp16-small",
                                "dtype": "float16",
                                "accum": "float16",
                            },
                        ]
                    },
                }
            ]
        }
    )
    profile = HardwareProfile("MI300X", 8)

    assert store.select(profile, "allreduce", 1536, dtype="bfloat16", accum="bfloat16").algorithm == "bf16-small"
    assert store.select(profile, "allreduce", 1536, dtype="float16", accum="float16").algorithm == "fp16-small"
    assert store.select(profile, "allreduce", 1536, dtype="float32", accum="float32") is None


def test_generic_config_matches_any_dtype():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "collectives": {
                        "allgather": [
                            {
                                "message_size": 1024,
                                "algorithm": "generic-allgather",
                            }
                        ]
                    }
                }
            ]
        }
    )

    config = store.select(HardwareProfile("MI300X", 8), "allgather", 1024, dtype="bfloat16", accum="bfloat16")

    assert config is not None
    assert config.algorithm == "generic-allgather"


def test_upsert_and_write_preserve_dtype_qualifiers(tmp_path):
    store = TunedConfigStore.empty()
    profile = HardwareProfile("MI300X", 8)
    store.upsert(
        profile,
        "allreduce",
        1024,
        TunedConfig("bf16"),
        dtype="bfloat16",
        accum="bfloat16",
    )
    store.upsert(
        profile,
        "allreduce",
        1024,
        TunedConfig("fp16"),
        dtype="float16",
        accum="float16",
    )
    path = tmp_path / "config.json"

    store.write_path(path)
    entries = json.loads(path.read_text())["profiles"][0]["collectives"]["allreduce"]

    assert {(entry["dtype"], entry["accum"], entry["algorithm"]) for entry in entries} == {
        ("bfloat16", "bfloat16", "bf16"),
        ("float16", "float16", "fp16"),
    }


def test_upsert_defaults_accum_to_dtype():
    store = TunedConfigStore.empty()
    profile = HardwareProfile("MI300X", 8)

    store.upsert(profile, "allreduce", 1024, TunedConfig("fp16"), dtype="float16")

    config = store.select(profile, "allreduce", 1024, dtype="float16")
    assert config is not None
    assert config.algorithm == "fp16"


def test_load_defaults_accum_to_dtype():
    store = TunedConfigStore.from_payload(
        {
            "profiles": [
                {
                    "collectives": {
                        "allreduce": [
                            {
                                "message_size": 1024,
                                "algorithm": "fp16",
                                "dtype": "float16",
                            }
                        ]
                    }
                }
            ]
        }
    )

    config = store.select(HardwareProfile(), "allreduce", 1024, dtype="float16")
    assert config is not None
    assert config.algorithm == "fp16"
