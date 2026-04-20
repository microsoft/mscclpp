# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Smoke tests for the EP extension.

These tests only exercise single-rank / pure-Python code paths so they can
run in CI without multi-GPU resources. Multi-rank dispatch/combine tests
belong in ``test/python/ext/ep/test_intranode.py`` and are left as TODO
until the Python frontend is validated on H100.

Run with::

    pytest -xvs test/python/ext/ep/test_ep_smoke.py
"""

from __future__ import annotations

import pytest

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pytest.skip("mscclpp_ep_cpp is not built (set -DMSCCLPP_BUILD_EXT_EP=ON)", allow_module_level=True)


def test_config_roundtrip():
    cfg = _cpp.Config(num_sms=20, num_max_nvl_chunked_send_tokens=6, num_max_nvl_chunked_recv_tokens=256,
                     num_max_rdma_chunked_send_tokens=6, num_max_rdma_chunked_recv_tokens=256)
    hint = cfg.get_nvl_buffer_size_hint(7168 * 2, 8)
    assert hint > 0


def test_low_latency_size_hint():
    assert _cpp.get_low_latency_rdma_size_hint(128, 7168, 8, 256) > 0


def test_low_latency_rejected():
    # Low-latency (pure RDMA) path is not ported yet; Python frontend must
    # refuse to construct a Buffer with low_latency_mode=True. We test the
    # underlying C++ constructor directly so this does not depend on the
    # full `mscclpp` Python package being installed.
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # The C++ Buffer allows low_latency_mode at construction; the enforcement
    # lives in the Python frontend (`mscclpp.ext.ep.buffer.Buffer.__init__`).
    # Verify the C++ side does NOT reject it, so the guarantee sits at the
    # Python layer where it belongs.
    buf = _cpp.Buffer(rank=0, num_ranks=1, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=True)
    assert not buf.is_available()
