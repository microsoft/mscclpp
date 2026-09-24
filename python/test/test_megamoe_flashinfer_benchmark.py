# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from types import SimpleNamespace

import pytest

from mscclpp.ext.megamoe import benchmark_flashinfer


def test_defaults_match_native_routed_geometry():
    args = benchmark_flashinfer._parse_args([])
    assert (args.tokens, args.hidden, args.intermediate) == (32, 4096, 4352)
    assert (args.experts, args.top_k) == (64, 7)
    assert (args.graph_batch, args.warmup, args.iterations) == (10, 5, 30)
    assert args.graph and args.fast_math
    assert not args.e5m2 and not args.in_kernel_fc2_reduce


@pytest.mark.parametrize(
    "argv",
    [
        ["--tokens", "0"],
        ["--hidden", "33"],
        ["--intermediate", "65"],
        ["--experts", "0"],
        ["--top-k", "0"],
        ["--top-k", "33"],
        ["--warmup", "0"],
        ["--iterations", "0"],
        ["--graph-batch", "0"],
        ["--gate-up-clamp", "nan"],
        ["--reference-relative-l2", "0"],
        ["--reference-relative-l2", "nan"],
    ],
)
def test_argument_rejection(argv):
    with pytest.raises(SystemExit):
        benchmark_flashinfer._parse_args(argv)


def test_knobs_json_accepts_inline_and_file(tmp_path):
    expected = {"mma_tiler_mnk": (256, 128, 128), "fast_accum": True}
    inline = benchmark_flashinfer._load_knobs(json.dumps({"mma_tiler_mnk": [256, 128, 128], "fast_accum": True}))
    assert inline == expected
    path = tmp_path / "knobs.json"
    path.write_text(json.dumps({"mma_tiler_mnk": [256, 128, 128], "fast_accum": True}))
    assert benchmark_flashinfer._load_knobs(str(path)) == expected
    with pytest.raises(ValueError, match="object"):
        benchmark_flashinfer._load_knobs("[]")


def test_explicit_flashinfer_root(tmp_path):
    root = tmp_path / "flashinfer-checkout"
    package = root / "flashinfer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    assert benchmark_flashinfer._activate_flashinfer_root(root) == root.resolve()
    with pytest.raises(FileNotFoundError, match="source package"):
        benchmark_flashinfer._activate_flashinfer_root(tmp_path / "missing")


def test_source_info_uses_distributed_commit(monkeypatch, tmp_path):
    module = SimpleNamespace(__version__="test", __file__=tmp_path / "flashinfer/__init__.py")
    monkeypatch.setenv("FLASHINFER_SOURCE_COMMIT", "0123456789abcdef")
    info = benchmark_flashinfer._source_info(tmp_path, module)
    assert info["commit"] == "0123456789abcdef"
    assert info["root"] == str(tmp_path)


def test_error_stats():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([[1.0, -2.0]])
    actual = torch.tensor([[1.5, -1.0]])
    stats = benchmark_flashinfer._error_stats(actual, expected)
    assert stats["max_abs_error"] == 1.0
    assert stats["mean_abs_error"] == 0.75
    assert stats["relative_l2"] > 0


def test_runtime_versions_have_expected_keys():
    versions = benchmark_flashinfer._runtime_versions()
    assert set(versions) == {
        "apache-tvm-ffi",
        "nvidia-cutlass-dsl",
        "nvshmem4py-cu13",
        "nvidia-nccl-cu13",
    }


def test_single_node_transport_defaults(monkeypatch):
    for name in ("NCCL_IB_DISABLE", "NCCL_NET", "NCCL_MNNVL_ENABLE", "NVSHMEM_REMOTE_TRANSPORT"):
        monkeypatch.delenv(name, raising=False)
    benchmark_flashinfer._configure_single_node_transport(4, 4)
    assert os.environ["NCCL_IB_DISABLE"] == "1"
    assert os.environ["NCCL_NET"] == "Socket"
    assert os.environ["NCCL_MNNVL_ENABLE"] == "1"
    assert os.environ["NVSHMEM_REMOTE_TRANSPORT"] == "none"


def test_multi_node_transport_is_untouched(monkeypatch):
    monkeypatch.delenv("NVSHMEM_REMOTE_TRANSPORT", raising=False)
    benchmark_flashinfer._configure_single_node_transport(8, 4)
    assert "NVSHMEM_REMOTE_TRANSPORT" not in os.environ
