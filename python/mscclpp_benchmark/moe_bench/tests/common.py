# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared setup for the GPU tests: package imports and a tiny two-GPU workload."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402


def tiny_config():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs.yaml",
        num_gpus=2,
        model="small",
        input_name="decode",
        routing="balanced",
        overrides={"warmup": 0, "iterations": 3},
    )
    config["model"]["intermediate_size"] = 128
    config["input"]["tokens_per_rank"] = 3
    return config
