# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import os

# Auto-register the MSCCL++ TorchComms backend .so path.
# TorchComms discovers backends via TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP.
# When this package is pip-installed, the .so lives alongside this __init__.py.
if "TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP" not in os.environ:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = glob.glob(os.path.join(_pkg_dir, "_comms_mscclpp*.so"))
    if _candidates:
        os.environ["TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP"] = _candidates[0]


def get_lib_path():
    """Return the path to the _comms_mscclpp shared library, or None if not found."""
    return os.environ.get("TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP")
