# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._mscclpp import *
import os as _os


def get_include():
    """Return the directory that contains the MSCCL++ headers."""
    return _os.path.join(_os.path.dirname(__file__), "include")
