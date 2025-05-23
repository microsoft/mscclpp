# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.src.types import ChannelType
from dataclasses import dataclass

@dataclass
class BaseChannel:
    channel_id: int
    src_rank: int
    dst_rank: int
    channel_type: ChannelType