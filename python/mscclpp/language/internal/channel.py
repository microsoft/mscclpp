# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.types import ChannelType
from dataclasses import dataclass


@dataclass
class BaseChannel:
    channel_id: int
    channel_type: ChannelType
