# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum

class GpuModel(Enum):
    # Nvidia GPU Models
    A100 = "a100"
    H100 = "h100"

    # AMD GPU Models
    MI300X = "mi300x"
    ALL = "all"

class BufferMode(Enum):
    INPLACE = 0
    OUTOFPLACE = 1
    ALL = 2