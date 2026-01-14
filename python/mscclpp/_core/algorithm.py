# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Optional, Tuple, Dict
from functools import cached_property


from mscclpp._mscclpp import (
    Algorithm as _Algorithm,
    DslAlgorithm as _DslAlgorithm,
    AlgorithmType as _AlgorithmType,
    Communicator,
    CollectiveBufferMode,
    DataType,
    Executor,
    ExecutionPlan,
    ReduceOp,
)

__all__ = ["Algorithm", "AlgorithmBuilder", "AlgorithmCollection"]


class Algorithm:
    """A wrapper for collective communication algorithms.

    This class provides a Python interface for collective communication algorithms
    such as allreduce, allgather, and reduce-scatter. Algorithms can be either
    DSL-based (defined using MSCCL++ execution plans) or native (implemented in C++/CUDA).

    Attributes:
        name: Human-readable name of the algorithm.
        collective: The collective operation this algorithm implements (e.g., "allreduce").
        message_size_range: Tuple of (min_size, max_size) in bytes for valid message sizes.
        tags: Dictionary of tag names to tag values for algorithm selection hints.
        buffer_mode: The buffer mode supported by this algorithm (IN_PLACE, OUT_OF_PLACE, or ANY).
    """

    class Constraint:
        """Constraints that define valid execution environments for the algorithm.

        Args:
            world_size: Required world size (number of ranks). 0 means any size.
            n_ranks_per_node: Required number of ranks per node. 0 means any.
        """

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
        constraint: Optional[Constraint] = None,
    ):
        if execution_plan is not None:
            self._algorithm = _DslAlgorithm(
                id,
                execution_plan,
                tags=tags if tags is not None else {},
                constraint=constraint._constraint if constraint is not None else _Algorithm.Constraint(),
            )
        elif native_handle is not None:
            self._algorithm = native_handle

    @classmethod
    def create_from_native_handle(cls, handle: _Algorithm):
        """Create an Algorithm instance from a native C++ algorithm handle.

        Args:
            handle: The native C++ algorithm handle.

        Returns:
            A new Algorithm instance wrapping the native handle.
        """
        return cls(
            native_handle=handle,
        )

    @classmethod
    def create_from_native_capsule(cls, obj):
        """Create an Algorithm instance from a PyCapsule object.

        Args:
            obj: A PyCapsule containing a native algorithm pointer.

        Returns:
            A new Algorithm instance wrapping the algorithm from the capsule.
        """
        handle = _Algorithm.from_native_capsule(obj)
        return cls(native_handle=handle)

    @cached_property
    def name(self) -> str:
        """The human-readable name of the algorithm."""
        return self._algorithm.name

    @cached_property
    def collective(self) -> str:
        """The collective operation this algorithm implements (e.g., "allreduce", "allgather")."""
        return self._algorithm.collective

    @cached_property
    def message_size_range(self) -> Tuple[int, int]:
        """The valid message size range (min_size, max_size) in bytes."""
        return (self._algorithm.message_range[0], self._algorithm.message_range[1])

    @cached_property
    def tags(self) -> Dict[str, int]:
        """Dictionary of tag names to tag values for algorithm selection hints."""
        return self._algorithm.tags

    @cached_property
    def buffer_mode(self) -> CollectiveBufferMode:
        """The buffer mode supported by this algorithm (IN_PLACE, OUT_OF_PLACE, or ANY)."""
        return self._algorithm.buffer_mode

    def is_dsl_algorithm(self) -> bool:
        """Check if this is a DSL-based algorithm.

        Returns:
            True if this algorithm is defined using DSL/execution plan, False otherwise.
        """
        if self._algorithm.type == _AlgorithmType.DSL:
            return True
        return False

    def is_native_algorithm(self) -> bool:
        """Check if this is a native C++/CUDA algorithm.

        Returns:
            True if this algorithm is implemented natively, False otherwise.
        """
        if self._algorithm.type == _AlgorithmType.NATIVE:
            return True
        return False

    def execute(
        self,
        comm: Communicator,
        input_buffer: int,
        output_buffer: int,
        input_size: int,
        output_size: int,
        dtype: DataType,
        op: ReduceOp = ReduceOp.NOP,
        stream: int = 0,
        executor: Optional[Executor] = None,
        nblocks=0,
        nthreads_per_block=0,
        extras: Optional[Dict[str, int]] = None,
    ) -> int:
        """Execute the collective algorithm.

        Args:
            comm: The communicator to use.
            input_buffer: Device pointer to the input buffer.
            output_buffer: Device pointer to the output buffer.
            input_size: Size of the input buffer in bytes.
            output_size: Size of the output buffer in bytes.
            dtype: Data type of the elements.
            op: Reduction operation for reduce-type collectives (default: NOP).
            stream: CUDA stream to execute on (default: 0).
            executor: The executor for DSL algorithms (required for DSL, optional for native).
            nblocks: Number of CUDA blocks (0 for auto-selection).
            nthreads_per_block: Number of threads per block (0 for auto-selection).
            extras: Additional algorithm-specific parameters.

        Returns:
            The result code (0 for success).
        """
        return self._algorithm.execute(
            comm,
            int(input_buffer),
            int(output_buffer),
            input_size,
            output_size,
            dtype,
            op,
            int(stream),
            executor,
            nblocks,
            nthreads_per_block,
            extras if extras is not None else {},
        )

class AlgorithmBuilder:
    def __init__(self, algorithm_builder: _AlgorithmBuilder):
        self._algorithm_builder = algorithm_builder

    def build(self) -> Algorithm:
        return Algorithm.create_from_native_handle(self._algorithm_builder.build())


class AlgorithmCollection:
    def __init__(self, native_collection: _AlgorithmCollection):
        self._native_collection = native_collection
        self._algorithms = [Algorithm.create_from_native_handle(algo) for algo in self._native_collection.to_list()]

    def __iter__(self):
        """Iterate over all algorithms in the collection."""
        return iter(self._algorithms)

    def __len__(self):
        """Return the number of algorithms in the collection."""
        return len(self._algorithms)

    def __getitem__(self, index: int) -> Algorithm:
        """Get an algorithm by index."""
        return self._algorithms[index]

    def get_by_collective(self, collective: str):
        """Get all algorithms for a specific collective operation."""
        return [algo for algo in self._algorithms if algo.collective == collective]

    def register_algorithm(self, collective: str, algo_name: str, algorithm: Algorithm):
        """Register an algorithm for a collective operation."""
        self._native_collection.register_algorithm(collective, algo_name, algorithm._algorithm)
        self._algorithms.append(algorithm)