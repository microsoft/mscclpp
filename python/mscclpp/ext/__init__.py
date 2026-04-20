# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .algorithm_collection_builder import *

try:
    from . import ep  # noqa: F401
except ImportError:
    # EP extension not built; leave `mscclpp.ext.ep` undefined.
    pass
