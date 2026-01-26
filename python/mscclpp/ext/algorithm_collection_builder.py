# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Union
from mscclpp._core.algorithm import Algorithm, AlgorithmBuilder, AlgorithmCollection
import atexit

from mscclpp._mscclpp import CppAlgorithmCollectionBuilder

__all__ = ["AlgorithmCollectionBuilder"]


class AlgorithmCollectionBuilder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlgorithmCollectionBuilder, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            CppAlgorithmCollectionBuilder.reset()
            cls._instance = None

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._builder = CppAlgorithmCollectionBuilder.get_instance()
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
