from mscclpp.language.internal.globals import *
from mscclpp.language.internal.operations import PipelineOperation


class LoopIterationContext:
    def __init__(self, unit, num_chunks):
        self.unit = unit
        self.num_chunks = num_chunks
        self.operations = []

    def __enter__(self):
        get_program().set_loop_context(self)

    def __exit__(self, exc_type, exc_value, traceback):
        get_program().set_loop_context(None)

        pipeline_operation = {}
        for rank, tb, operation in self.operations:
            key = (rank, tb)
            if key not in pipeline_operation:
                pipeline_operation[key] = PipelineOperation(self.unit, self.num_chunks)
            pipeline_operation[key].operations.append(operation)

        for (rank, tb), pipeline in pipeline_operation.items():
            get_program().add_operation(rank, tb, pipeline)

    def add_operation(self, rank, tb, operation):
        self.operations.append((rank, tb, operation))
