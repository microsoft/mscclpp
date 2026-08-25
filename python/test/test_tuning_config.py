# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

from mscclpp_benchmark.tuning_config import HardwareProfile, TunedConfig, TunedConfigStore


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
