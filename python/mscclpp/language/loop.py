# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.globals import *
from typing import Dict


class LoopIterationContext:
    """A context manager for defining pipelined loop operations in MSCCL++ programs.

    LoopIterationContext provides a way to define operations that will be executed
    in a pipelined manner across multiple iterations, where each pipeline iteration
    processes a specific chunk size (unit) of data. The pipeline allows overlapping
    execution of operations enabling efficient data processing.

    Attributes:
        unit (int): The unit size for pipeline operations.
        num_chunks (int): The number of chunks to process in the pipeline.
        operations (list): List of operations to be pipelined.
    """

    def __init__(self, unit, num_chunks):
        """Initialize a new LoopIterationContext.

        Args:
            unit (int): The unit size for pipeline operations, typically representing
                the granularity of data processing.
            num_chunks (int): The number of chunks to process in the pipeline,
                determining the pipeline depth.

        Example:
            >>> with LoopIterationContext(unit=1024, num_chunks=4):
            ...     # Define operations to be pipelined
            ...     pass
        """
        self.unit = unit
        self.num_chunks = num_chunks
        self.pre_operations = dict()

    def __enter__(self):
        """Enter the context and set this as the active loop context.

        This method is called when entering the 'with' statement and registers
        this context as the active pipeline context in the current program.
        """
        get_program().set_loop_context(self)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context and process collected operations into pipeline operations.

        This method is called when exiting the 'with' statement. It processes all
        operations that were collected during the context execution, groups them
        by rank and thread block, and converts them into pipeline operations.
        """
        get_program().set_loop_context(None)

    def process_operation(self, operations):
        """Add an operation to be included in the pipeline.

        This method is called internally to collect operations that should be
        pipelined together. Operations added here will be grouped and converted
        into pipeline operations when the context exits.

        Args:
            rank (int): The rank where this operation will be executed.
            tb (int): The thread block ID that will execute this operation.
            operation: The operation object to be added to the pipeline.
        """
        for operation in operations:
            operation.set_pipeline_context(self)
