# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared loader for the MSCCL++ expert-parallel Python extension."""

from __future__ import annotations

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with "
        "-DMSCCLPP_BUILD_EXT_EP=ON or install with `pip install .[ep]`."
    ) from exc

DispatchLayout = _cpp.DispatchLayout
MoEMode = _cpp.MoEMode
CombineMode = _cpp.CombineMode
DispatchDataType = _cpp.DispatchDataType
MoERuntime = _cpp.MoERuntime
Config = getattr(_cpp, "Config", None)
