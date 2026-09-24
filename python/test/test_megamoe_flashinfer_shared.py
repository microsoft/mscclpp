# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest

from mscclpp.ext.megamoe import benchmark_flashinfer_shared


def test_defaults_match_native_shared_benchmark():
    args = benchmark_flashinfer_shared._parse_args([])
    assert (args.tokens, args.original_hidden, args.hidden, args.intermediate) == (32, 8704, 4096, 4352)
    assert args.experts is None and args.top_k == 7
    assert args.shared_intermediate == 2048 and args.shared_sms == 32
    assert not args.e5m2 and not args.in_kernel_fc2_reduce
    assert args.modes == list(benchmark_flashinfer_shared.MODES)


@pytest.mark.parametrize(
    "argv",
    [
        ["--gate-up-clamp", "nan"],
        ["--route-sm-margin", "30"],
        ["--trace-path", "trace.json"],
        ["--tuned-profile", "profile.json", "--topology-id", "test"],
        ["--cache-dir", "cache"],
        ["--bootstrap-port", "12345"],
        ["--check", "--in-kernel-fc2-reduce"],
        ["--hidden", "33"],
        ["--intermediate", "65"],
        ["--original-hidden", "129"],
        ["--shared-intermediate", "129"],
        ["--modes", "overlap", "overlap"],
    ],
)
def test_argument_rejection(argv):
    with pytest.raises(SystemExit):
        benchmark_flashinfer_shared._parse_args(argv)


def test_mode_selection_matches_native_parser():
    args = benchmark_flashinfer_shared._parse_args(["--modes", "overlap"])
    assert args.modes == ["overlap"]


def test_e5m2_uses_relaxed_default_reference_threshold():
    assert benchmark_flashinfer_shared._parse_args(["--e5m2"]).reference_relative_l2 == 0.12
    explicit = benchmark_flashinfer_shared._parse_args(["--e5m2", "--reference-relative-l2", "0.08"])
    assert explicit.reference_relative_l2 == 0.08


def test_scope_documents_overlap_difference():
    args = benchmark_flashinfer_shared._parse_args([])
    scope = benchmark_flashinfer_shared._scope_report(args)
    assert "routed_enqueue_first_producer_ready_event" in scope["included_by_mode"]["overlap"]
    assert "kernel-entry signal" in scope["overlap_policy"]["limitation"]


def test_routed_config_local_experts():
    config = benchmark_flashinfer_shared._RoutedConfig(1, 4, 32, 4096, 4352, 64, 7, -1.0)
    assert config.local_experts == 16


def test_single_node_environment_helper_is_shared(monkeypatch):
    for name in ("NCCL_IB_DISABLE", "NCCL_NET", "NCCL_MNNVL_ENABLE", "NVSHMEM_REMOTE_TRANSPORT"):
        monkeypatch.delenv(name, raising=False)
    benchmark_flashinfer_shared._configure_single_node_transport(4, 4)
    assert os.environ["NVSHMEM_REMOTE_TRANSPORT"] == "none"
