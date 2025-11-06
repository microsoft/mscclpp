# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union, List, Dict

from mscclpp._common import BufferMode, GpuModel
from mscclpp._executor import ExecutionPlanHandle
from mscclpp.comm import CommGroup
from mscclpp.language.collectives import Collective

from ._mscclpp import (
    Algorithm as _Algorithm,
)

class AlgorithmConstraint:
    pass

class Algorithm():
    def __init__(
        self,
        name: str,
        collective: Union[Collective | str],
        min_size: int = 0,
        max_size: int = (1 << 64 - 1),
        buffer_mode: BufferMode = BufferMode.ALL,
        architectures: List[Union[GpuModel, str]] = [GpuModel.ALL],
        constraints: Optional[AlgorithmConstraint] = None,
        execution_plan_handle: Optional[ExecutionPlanHandle] = None,
        native_handle: Optional[int] = None,
        tags: Optional[Dict[str, int]] = None,
    ):
        self.name = name
        self.collective = collective
        self.execution_plan_handle = execution_plan_handle
        self.native_handle = native_handle
        self.min_size = min_size
        self.max_size = max_size
        self.architectures = architectures
        self.buffer_mode = buffer_mode
        self.constraints = constraints
        self.tags = tags

    @classmethod
    def create_from_handle(
        cls,
        name: str,
        collective: Union[Collective, str],
        handle: int,
        min_size: int = 0,
        max_size: int = (1 << 64 - 1),
        buffer_mode: BufferMode = BufferMode.ALL,
        architectures: List[Union[GpuModel, str]] = [GpuModel.ALL],
        constraints: Optional[AlgorithmConstraint] = None,
        tags: Optional[Dict[str, int]] = None,
    ):
        return cls(
            name,
            collective,
            min_size,
            max_size,
            buffer_mode,
            architectures,
            constraints,
            tags=tags,
            native_handle=handle,
        )

    def execute(
        self,
        comm_group: CommGroup,
        rank: int,
        nblocks: int,
        nthreads_per_block: int,
        input: int,
        output: int,
        count: int,
        dtype,
        extra: dict,
        stream: int,
    ):
        if self.is_dsl_based:
            # Launch via execution plan
            pass
        else:
            # Launch via native handle
            pass

    @property
    def is_dsl_based(self) -> bool:
        return self.execution_plan_handle is not None

    @property
    def is_native_based(self) -> bool:
        return self.native_handle is not None
