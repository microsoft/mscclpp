# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, Dict, Union
from functools import cached_property
import atexit


from ._mscclpp import (
    Algorithm as _Algorithm,
    DslAlgorithm as _DslAlgorithm,
    AlgorithmType as _AlgorithmType,
    AlgorithmBuilder as _AlgorithmBuilder,
    AlgorithmCollection as _AlgorithmCollection,
    AlgorithmCollectionBuilder as _AlgorithmCollectionBuilder,
    Communicator,
    CollectiveBufferMode,
    DataType,
    Executor,
    ExecutionPlan,
    ReduceOp,
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
        return cls(
            native_handle=handle,
        )

    @classmethod
    def create_from_native_capsule(cls, obj):
        handle = _Algorithm.from_native_capsule(obj)
        return cls(native_handle=handle)

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

    @cached_property
    def buffer_mode(self) -> CollectiveBufferMode:
        return self._algorithm.buffer_mode

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


class AlgorithmCollectionBuilder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlgorithmCollectionBuilder, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            _AlgorithmCollectionBuilder.reset()
            cls._instance = None

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._builder = _AlgorithmCollectionBuilder.get_instance()
            self._initialized = True

    def add_algorithm_builder(self, algorithm_builder: Union[AlgorithmBuilder, Algorithm]):
        if isinstance(algorithm_builder, AlgorithmBuilder):
            self._builder.add_algorithm_builder(algorithm_builder._algorithm_builder)
            return
        if isinstance(algorithm_builder, Algorithm):
            if algorithm_builder.is_dsl_algorithm():
                self._builder.add_dsl_algorithm_builder(algorithm_builder._algorithm)
                return
        raise ValueError("The 'algorithm_builder' argument must be an instance of AlgorithmBuilder or DSL Algorithm.")

    def set_algorithm_selector(self, selector):
        self._builder.set_algorithm_selector(selector)

    def set_fallback_algorithm_selector(self, selector):
        self._builder.set_fallback_algorithm_selector(selector)

    def build(self) -> AlgorithmCollection:
        collection = self._builder.build()
        return AlgorithmCollection(collection)

    def build_default_algorithms(self, scratch_buffer: int, scratch_buffer_size: int, rank: int) -> AlgorithmCollection:
        native_collection = self._builder.build_default_algorithms(int(scratch_buffer), scratch_buffer_size, rank)
        return AlgorithmCollection(native_collection)


atexit.register(AlgorithmCollectionBuilder.reset)
