# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Dict


class BarrierInfo:
    def __init__(self, tb_list):
        self.tb_list = tb_list

    def __eq__(self, other):
        return self.tb_list == other.tb_list

    def __hash__(self):
        return hash(tuple(self.tb_list))


@dataclass
class Rank:
    rank_id: int
    current_max_barrier_id: int = 0
    current_barriers: Dict[BarrierInfo, int] = field(default_factory=dict)

    def get_barrier_id(self, tb_list):
        barrier_info = BarrierInfo(tb_list)
        if barrier_info in self.current_barriers:
            return self.current_barriers[barrier_info]
        else:
            self.current_barriers[barrier_info] = self.current_max_barrier_id
            barrier_id = self.current_max_barrier_id
            self.current_max_barrier_id += 1
            return barrier_id
