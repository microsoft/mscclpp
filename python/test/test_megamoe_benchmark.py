# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.ext.megamoe.benchmark import _latency_summary


def test_latency_summary_uses_slowest_rank_per_sample():
    reports = [
        {"samples": [10.0, 40.0, 30.0]},
        {"samples": [20.0, 30.0, 50.0]},
    ]

    assert _latency_summary(reports, "samples", 3) == {
        "median": 40.0,
        "mean": 110.0 / 3.0,
        "minimum": 20.0,
        "maximum": 50.0,
    }
