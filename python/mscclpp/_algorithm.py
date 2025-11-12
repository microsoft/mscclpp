# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, Dict, Union, Any
from functools import cached_property


from ._mscclpp import (
    Algorithm as _Algorithm,
    DslAlgorithm as _DslAlgorithm,
    AlgorithmType as _AlgorithmType,
    ExecutionPlan,
    Communicator,
)


class Algorithm:

    class Constraint:
        def __init__(self, world_size: int = 0, n_ranks_per_node: int = 0):
            self._constraint = _Algorithm.Constraint(world_size, n_ranks_per_node)

        @property
        def world_size(self) -> int:
            return self._constraint.worldSize

        @property
        def n_ranks_per_node(self) -> int:
            return self._constraint.nRanksPerNode

    def __init__(
        self,
        id: Optional[str] = None,
        execution_plan: Optional[ExecutionPlan] = None,
        native_handle: Optional[_Algorithm] = None,
        tags: Optional[Dict[str, int]] = None,
        constraints: Optional[Constraint] = None,
    ):
        if execution_plan is not None:
            self._algorithm = _DslAlgorithm(
                id,
                execution_plan._handle,
                tags=tags if tags is not None else {},
                constraints=constraints._constraint if constraints is not None else _Algorithm.Constraint(),
            )
        elif native_handle is not None:
            self._algorithm = native_handle

    @classmethod
    def create_from_native_handle(cls, handle: _Algorithm):
        return cls(
            native_handle=handle,
        )

    @cached_property
    def name(self) -> str:
        return self._algorithm.name

    @cached_property
    def collective(self) -> str:
        return self._algorithm.collective

    @cached_property
    def message_size_range(self) -> Tuple[int, int]:
        return (self._algorithm.message_range[0], self._algorithm.message_range[1])

    @cached_property
    def tags(self) -> Dict[str, int]:
        return self._algorithm.tags

    def is_dsl_algorithm(self) -> bool:
        if self._algorithm.type == _AlgorithmType.DSL:
            return True
        return False

    def is_native_algorithm(self) -> bool:
        if self._algorithm.type == _AlgorithmType.NATIVE:
            return True
        return False

    def execute(
        self,
        comm: Communicator,
        input_buffer: Union[int, "torch.Tensor", "cupy.ndarray"],
        output_buffer: Union[int, "torch.Tensor", "cupy.ndarray"],
        input_size: int,
        output_size: int,
        dtype: int,
        stream: Union[int, "torch.cuda.Stream", "cupy.cuda.Stream"],
        extras: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Execute the algorithm.

        Args:
            comm: The communicator instance
            input_buffer: Input buffer pointer (as int) or tensor
            output_buffer: Output buffer pointer (as int) or tensor
            input_size: Input size in bytes
            output_size: Output size in bytes
            dtype: Data type (integer value)
            stream: CUDA stream pointer (as int) or stream object
            extras: Optional dictionary with extra parameters (e.g., {'executor': executor_instance})

        Returns:
            Status code (0 for success)
        """
        # Convert tensor/array to pointer if needed
        if hasattr(input_buffer, "data_ptr"):
            # PyTorch tensor
            input_ptr = input_buffer.data_ptr()
        elif hasattr(input_buffer, "data") and hasattr(input_buffer.data, "ptr"):
            # CuPy array
            input_ptr = input_buffer.data.ptr
        else:
            # Assume it's already an integer pointer
            input_ptr = int(input_buffer)

        if hasattr(output_buffer, "data_ptr"):
            # PyTorch tensor
            output_ptr = output_buffer.data_ptr()
        elif hasattr(output_buffer, "data") and hasattr(output_buffer.data, "ptr"):
            # CuPy array
            output_ptr = output_buffer.data.ptr
        else:
            # Assume it's already an integer pointer
            output_ptr = int(output_buffer)

        # Convert stream to pointer if needed
        if hasattr(stream, "cuda_stream"):
            # PyTorch stream
            stream_ptr = stream.cuda_stream
        elif hasattr(stream, "ptr"):
            # CuPy stream
            stream_ptr = stream.ptr
        else:
            # Assume it's already an integer pointer
            stream_ptr = int(stream)

        # Note: The C++ binding currently creates an empty extras dict internally
        # If extras support is needed, the C++ binding needs to be updated
        # For now, extras parameter is accepted but not yet passed through
        return self._algorithm.execute(comm, input_ptr, output_ptr, input_size, output_size, dtype, stream_ptr)
