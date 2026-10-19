# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load the optional native EP extension without hiding dependency failures."""

from importlib import import_module

try:
    _cpp = import_module("mscclpp.mscclpp_ep_cpp")
except ModuleNotFoundError as exc:
    if exc.name != "mscclpp.mscclpp_ep_cpp":
        raise
    raise ImportError(
        "mscclpp.ep requires the mscclpp_ep_cpp extension. "
        "Rebuild MSCCL++ with its expert-parallel Python extension enabled."
    ) from exc

CombineMode = _cpp.CombineMode
DispatchDataType = _cpp.DispatchDataType
DispatchLayout = _cpp.DispatchLayout
MoEMode = _cpp.MoEMode
MoERuntime = _cpp.MoERuntime
PrepareHandle = _cpp.PrepareHandle
DispatchHandle = _cpp.DispatchHandle
create_moe_runtime = _cpp.create_moe_runtime
