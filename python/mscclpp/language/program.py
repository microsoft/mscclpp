# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.collectives import Collective
from mscclpp.language.internal.globals import set_program
from mscclpp.language.internal.types import BufferType, RemoteBuffer, ChannelType
from mscclpp.language.internal.gpu import Gpu
from mscclpp.language.channel import *
from mscclpp.language.rank import Semaphore
from mscclpp.language.collectives import *
from mscclpp.language.utils import AlgoSpec, ReplicationPolicy
from typing import List
import json


class CollectiveProgram:
    """A program definition for MSCCL++ collective communication operations.

    CollectiveProgram serves as the main container for defining and executing
    collective communication programs using the MSCCL++ DSL. It manages
    GPU resources, channels, operations, and provides serialization to JSON
    format for execution.

    Attributes:
        name (str): The name of the program.
        collective (Collective): The collective operation this program implements.
        num_ranks (int): The number of ranks participating in the program.
        instances (int): The number of instances to replicate.
        protocol (str): The communication protocol ("Simple" or "LL").
        instr_fusion (bool): Whether to enable instruction fusion optimization.
        replication_policy (ReplicationPolicy): The policy for replicating operations.
        reuse_resources (bool): Whether to reuse resources across instances.
        num_threads_per_block (int): Number of threads per GPU thread block.
        use_double_scratch_buffer (bool): Whether to use double scratch buffering.
        buffer_alignment (int): Buffer alignment in bytes.
        min_message_size (int): Minimum message size for this program.
        max_message_size (int): Maximum message size for this program.
        buffers (list): Buffer configurations for each rank.
        gpus (List[Gpu]): List of GPU objects representing each rank.
        loop_context: Current pipeline loop context, if any.
    """

    def __init__(
        self,
        name: str,
        collective: Collective,
        num_ranks: int,
        instances: int = 1,
        protocol: str = "Simple",
        instr_fusion: bool = True,
        auto_sync: bool = True,
        replication_policy: ReplicationPolicy = ReplicationPolicy.interleaved,
        reuse_resources: bool = False,
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
        buffer_alignment: int = 16,
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
    ):
        """Initialize a new CollectiveProgram.

        Args:
            name (str): The name identifier for this program.
            collective (Collective): The collective operation to implement.
            num_ranks (int): The number of participating ranks.
            instances (int, optional): Number of instances to replicate. Defaults to 1.
            protocol (str, optional): Communication protocol ("Simple" or "LL").
                Defaults to "Simple".
            instr_fusion (bool, optional): Enable instruction fusion optimization.
                Defaults to True.
            replication_policy (ReplicationPolicy, optional): Policy for operation replication.
                Defaults to ReplicationPolicy.interleaved.
            reuse_resources (bool, optional): Whether to reuse resources. Defaults to False.
            num_threads_per_block (int, optional): Threads per GPU thread block. Defaults to 1024.
            use_double_scratch_buffer (bool, optional): Use double scratch buffering.
                Defaults to False.
            buffer_alignment (int, optional): Buffer alignment in bytes. Defaults to 16.
            min_message_size (int, optional): Minimum message size. Defaults to 0.
            max_message_size (int, optional): Maximum message size. Defaults to 2^64-1.

        Raises:
            AssertionError: If protocol is not "Simple" or "LL".

        Example:
            >>> from mscclpp.language.collectives import AllReduce
            >>> collective = AllReduce(num_ranks=4, chunk_factor=1, inplace=False)
            >>> with CollectiveProgram("allreduce_4", collective, 4) as prog:
            ...     # Define communication operations
            ...     pass
        """
        self.name = name
        self.collective = collective
        self.num_ranks = num_ranks
        self.instances = instances
        self.protocol = protocol
        self.instr_fusion = instr_fusion
        self.auto_sync = auto_sync
        self.replication_policy = replication_policy
        self.reuse_resources = reuse_resources
        self.num_threads_per_block = num_threads_per_block
        self.use_double_scratch_buffer = use_double_scratch_buffer
        self.buffer_alignment = buffer_alignment
        self.min_message_size = min_message_size
        self.max_message_size = max_message_size
        assert protocol == "Simple" or protocol == "LL", f"Given protocol: {protocol}. Must be either Simple, LL"
        self.buffers = collective.init_buffers()
        self.gpus: List[Gpu] = []
        for rank in range(self.num_ranks):
            self.gpus.append(
                Gpu(rank, self.buffers[rank][BufferType.input].size, self.buffers[rank][BufferType.output].size, 0)
            )

        self.loop_context = None

    @classmethod
    def from_spec(cls, spec: AlgoSpec):
        """Initialize a new CollectiveProgram from an algorithm specification.

        This constructor provides an alternative way to create a CollectiveProgram
        using an AlgoSpec object, which contains the complete algorithm specification
        including collective instance, protocol parameters, and optimization settings.
        The collective operation is directly provided through the spec's collective attribute.

        Args:
            spec (AlgoSpec): Algorithm specification containing all program parameters
                and configuration settings, including a Collective instance.

        Raises:
            AssertionError: If protocol is not "Simple" or "LL".

        Example:
            >>> from mscclpp.language.utils import AlgoSpec
            >>> from mscclpp.language.collectives import AllReduce
            >>> collective = AllReduce(num_ranks=4, chunk_factor=1, inplace=False)
            >>> spec = AlgoSpec(
            ...     name="my_allreduce",
            ...     collective=collective,
            ...     world_size=4,
            ...     instances=1,
            ...     protocol="Simple",
            ...     in_place=False
            ... )
            >>> with CollectiveProgram.from_spec(spec) as prog:
            ...     # Define communication operations
            ...     pass
        """
        return cls(
            spec.name,
            spec.collective,
            spec.world_size,
            instances=spec.instances,
            protocol=spec.protocol,
            instr_fusion=spec.instr_fusion,
            auto_sync=spec.auto_sync,
            replication_policy=spec.replication_policy,
            reuse_resources=spec.reuse_resources,
            num_threads_per_block=spec.num_threads_per_block,
            use_double_scratch_buffer=spec.use_double_scratch_buffer,
            buffer_alignment=spec.buffer_alignment,
            min_message_size=spec.min_message_size,
            max_message_size=spec.max_message_size,
        )

    def __enter__(self):
        """Enter the program context and set this as the active program.

        This method is called when entering the 'with' statement and registers
        this program as the active program in the global context.
        """
        set_program(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the program context and clear the active program.

        This method is called when exiting the 'with' statement and removes
        this program from the global context.
        """
        MemoryChannel.reset()
        PortChannel.reset()
        SwitchChannel.reset()
        Semaphore.reset()
        set_program(None)

    def add_channel(self, channel):
        if channel.channel_type == ChannelType.switch:
            for gpu in channel.rank_group.ranks:
                self.gpus[gpu].add_channel(channel)
        else:
            self.gpus[channel.src_rank].add_channel(channel)

    def setup_channel(self, tb, channel):
        tb_channel_ids = []
        tb_channel_ids.append(self.gpus[channel.src_rank].setup_channel(tb, channel))
        return tb_channel_ids

    def setup_remote_chunk(self, rank, tb, remote_chunk: RemoteBuffer, channel_access: ChannelType):
        return self.gpus[rank].add_remote_buffer(tb, remote_chunk, channel_access)

    def add_semaphore(self, semaphore):
        self.gpus[semaphore.rank].add_semaphore(semaphore)

    def add_operation(self, rank, tb, operation):
        if self.loop_context != None:
            self.loop_context.add_operation(rank, tb, operation)
        else:
            self.gpus[rank].add_operation(tb, operation)

    def post_process_operations(self):
        for gpu in self.gpus:
            if self.instr_fusion:
                gpu.optimize_operations()
            gpu.adding_data_sync()
            if self.auto_sync:
                gpu.resolve_data_dependency()
            gpu.replicate_instances(
                self.instances,
                self.get_default_replication_policy_function(),
                self.get_buffer_replication_policy_function(),
            )

    def get_default_replication_policy_function(self):
        return lambda value, instance, num_instances: value * num_instances + instance

    def get_buffer_replication_policy_function(self):
        if self.replication_policy == ReplicationPolicy.interleaved:
            return lambda value, size, instance, num_instances: value * num_instances + instance * size
        else:
            return lambda value, instance, num_instances: value

    def set_loop_context(self, loop_context):
        if self.loop_context is not None and loop_context is not None:
            raise RuntimeError("Nested Pipelines are not Supported.")
        self.loop_context = loop_context

    def to_json(self, indent=2, **kwargs):
        self.post_process_operations()
        json_obj = {
            "name": self.name,
            "collective": self.collective.name,
            "protocol": self.protocol,
            "inplace": self.collective.inplace,
            "reuse_resources": self.reuse_resources,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "num_threads_per_block": self.num_threads_per_block,
            "use_double_scratch_buffer": self.use_double_scratch_buffer,
            "buffer_alignment": self.buffer_alignment,
            "min_message_size": self.min_message_size,
            "max_message_size": self.max_message_size,
        }

        return json.dumps(json_obj, indent=indent, **kwargs)
