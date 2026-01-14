# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Union
from mscclpp._core.utils import Algorithm
import atexit

from mscclpp._mscclpp import (
    AlgorithmBuilder as _AlgorithmBuilder,
    AlgorithmCollection as _AlgorithmCollection,
    AlgorithmCollectionBuilder as _AlgorithmCollectionBuilder,
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
