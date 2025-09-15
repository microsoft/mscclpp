# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ._mscclpp import ExecutionPlanRegistry as _ExecutionPlanRegistry

import atexit


class ExecutionPlanRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionPlanRegistry, cls).__new__(cls)
            cls._instance._registry = _ExecutionPlanRegistry()
        return cls._instance

    def register_plan(self, plan):
        return self._instance._registry.register_plan(plan)

    def set_selector(self, selector):
        self._instance._registry.set_selector(selector)

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._registry.clear()
            cls._instance = None


# class ExecutionRequest:
#     pass

# class ExecutionPlanHandle:
#     pass

atexit.register(ExecutionPlanRegistry.reset_instance)
