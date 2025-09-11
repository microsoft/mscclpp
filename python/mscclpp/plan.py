# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Callable, Dict
from ._mscclpp import ExecutionPlan

@dataclass(frozen=True)
class PlanHandle:
    """A handle to a compiled MSCCL++ execution plan."""
    id: str # Unique identifier for the plan
    name: str # Name of the plan
    collective: str # Type of collective operation
    tags: dict # Additional tags or metadata
    constraints: dict # Constraints such as min/max message size, nranks_per_node, world_size
    executionPlan: ExecutionPlan # The actual ExecutionPlan object

class Registry():
    _instance = None
    _plans = {}

    def __new__(cls):
        raise TypeError("Registry cannot be instantiated")

    @classmethod
    def register(cls, plan: PlanHandle):
        cls._plans[plan.id] = plan

    @classmethod
    def get(cls, plan_id: str) -> PlanHandle:
        return cls._plans.get(plan_id)

@dataclass(frozen=True)
class Request:
    collective: str
    msg_bytes: int
    world_size: int
    nranks_per_node: int
    input_buff: int
    output_buff: int
    hints: dict

Selector = Callable[[Dict[str, PlanHandle], Request], PlanHandle | str]
_selector: Selector = None

def set_selector(selector: Selector):
    """Set the global plan selector function.

    Args:
        selector (Selector): A function that takes a dictionary of available plans and a request,
                                and returns a PlanHandle or the ID of a PlanHandle.
    """
    global _selector
    _selector = selector

def clear_selector():
    """Clear the global plan selector function."""
    global _selector
    _selector = None
