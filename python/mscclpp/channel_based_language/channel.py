# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.channel_based_language.types import ChannelType
from dataclasses import dataclass

@dataclass
class Channel:
    channel_id: int
    src_rank: int
    dst_rank: int
    channel_type: ChannelType