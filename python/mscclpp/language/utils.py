# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from dataclasses import dataclass, field
from mscclpp.language.collectives import Collective


__all__ = ["AlgoSpec", "ReplicationPolicy"]

class ReplicationPolicy(Enum):
    interleaved = "interleaved"
    none = "none"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    collective: Collective
    nranks_per_node: int
    world_size: int
    in_place: bool
    instances: int
    protocol: str
    instr_fusion: bool = True
    auto_sync: bool = True
    replication_policy: ReplicationPolicy = ReplicationPolicy.interleaved
    reuse_resources: bool = False
    num_threads_per_block: int = 1024
    use_double_scratch_buffer: bool = False
    buffer_alignment: int = 16
    min_message_size: int = 0
    max_message_size: int = 2**64 - 1
    tags: dict = field(default_factory=dict)
