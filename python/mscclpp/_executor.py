# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import atexit
from dataclasses import dataclass
from functools import cached_property

from mscclpp._version import __version__, __commit_id__
from ._mscclpp import (
    ExecutionPlan,
    ExecutionPlanConstraint,
    ExecutionPlanHandle as _ExecutionPlanHandle,
    ExecutionPlanRegistry as _ExecutionPlanRegistry,
)


class ExecutionPlanHandle:

    def __init__(self, handle: _ExecutionPlanHandle):
        self._handle = handle

    @cached_property
    def id(self) -> int:
        return self._handle.id

    @cached_property
    def tags(self) -> set:
        return frozenset(self._handle.tags)

    @cached_property
    def plan(self) -> ExecutionPlan:
        return self._handle.plan

    @cached_property
    def constraints(self) -> ExecutionPlanConstraint:
        return self._handle.constraints


@dataclass(frozen=True)
class ExecutionRequest:
    collective: str
    world_size: int
    n_ranks_per_node: int
    send_buffer: int
    recv_buffer: int
    message_size: int
    hints: dict


class ExecutionPlanRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionPlanRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._registry = _ExecutionPlanRegistry.get_instance()
            self._id_map = {}
            self._collective_map = {}
            self._selector = None
            self._initialized = True

    def register_plan(self, plan: ExecutionPlanHandle):
        self._id_map[plan.id] = plan
        if plan.plan.collective not in self._collective_map:
            self._collective_map[plan.plan.collective] = []
        self._collective_map[plan.plan.collective].append(plan)
        return self._instance._registry.register_plan(plan._handle)

    def set_selector(self, selector):
        self._selector = selector
        self._instance._registry.set_selector(selector)

    def set_default_selector(self, selector):
        self._selector = selector
        self._instance._registry.set_default_selector(selector)

    def get(self, id: str) -> ExecutionPlanHandle:
        return self._id_map.get(id, None)

    def select(
        self,
        collective: str,
        world_size: int,
        n_ranks_per_node: int,
        send_buffer: int,
        recv_buffer: int,
        message_size: int,
        hints: dict = {},
    ) -> ExecutionPlanHandle:
        if self._selector is None or collective not in self._collective_map:
            return None
        req = ExecutionRequest(
            collective=collective,
            world_size=world_size,
            n_ranks_per_node=n_ranks_per_node,
            send_buffer=send_buffer,
            recv_buffer=recv_buffer,
            message_size=message_size,
            hints=hints,
        )
        return self._selector(self._collective_map[collective], req)

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._registry.clear()
            cls._instance._id_map = {}
            cls._instance._collective_map = {}
            cls._instance._selector = None
            cls._instance = None


atexit.register(ExecutionPlanRegistry.reset_instance)