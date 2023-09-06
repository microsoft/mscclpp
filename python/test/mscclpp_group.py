# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
from typing import Type

import cupy as cp
from mscclpp import (
    Communicator,
    Connection,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    ProxyService,
    RegisteredMemory,
    SimpleProxyChannel,
    SmChannel,
    SmDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    TransportFlags,
)
import numpy as np

from .mscclpp_mpi import MpiGroup

logger = logging.getLogger(__name__)


class MscclppGroup:
    def __init__(self, mpi_group: MpiGroup, interfaceIpPortTrio=""):
        self.bootstrap = TcpBootstrap.create(mpi_group.comm.rank, mpi_group.comm.size)
        if interfaceIpPortTrio == "":
            uniq_id = None
            if mpi_group.comm.rank == 0:
                # similar to NCCL's unique id
                uniq_id = self.bootstrap.create_unique_id()
            uniq_id_global = mpi_group.comm.bcast(uniq_id, 0)
            self.bootstrap.initialize(uniq_id_global)
        else:
            # use this instead
            self.bootstrap.initialize(interfaceIpPortTrio)
        self.communicator = Communicator(self.bootstrap)
        self.my_rank = self.bootstrap.get_rank()
        self.nranks = self.bootstrap.get_n_ranks()

    def barrier(self):
        self.bootstrap.barrier()

    def send(self, tensor: np.ndarray, peer: int, tag: int):
        self.bootstrap.send(tensor.ctypes.data, tensor.size * tensor.itemsize, peer, tag)

    def recv(self, tensor: np.ndarray, peer: int, tag: int):
        self.bootstrap.recv(tensor.ctypes.data, tensor.size * tensor.itemsize, peer, tag)

    def my_ib_device(self, local_rank: int) -> Transport:
        if local_rank == 0:
            return Transport.IB0
        if local_rank == 1:
            return Transport.IB1
        if local_rank == 2:
            return Transport.IB2
        if local_rank == 3:
            return Transport.IB3
        if local_rank == 4:
            return Transport.IB4
        if local_rank == 5:
            return Transport.IB5
        if local_rank == 6:
            return Transport.IB6
        if local_rank == 7:
            return Transport.IB7
        else:
            assert False  # only 8 IBs are supported

    def make_connection(self, remote_ranks: list[int], transport: Transport) -> dict[int, Connection]:
        connections = {}
        for rank in remote_ranks:
            connections[rank] = self.communicator.connect_on_setup(rank, 0, transport)
        self.communicator.setup()
        connections = {rank: connections[rank].get() for rank in connections}
        return connections

    def register_tensor_with_connections(
        self, tensor: Type[cp.ndarray] or Type[np.ndarray], connections: dict[int, Connection]
    ) -> dict[int, RegisteredMemory]:
        transport_flags = TransportFlags()
        for rank in connections:
            transport_flags |= connections[rank].transport()
        data_ptr = tensor.data.ptr if isinstance(tensor, cp.ndarray) else tensor.ctypes.data
        local_reg_memory = self.communicator.register_memory(data_ptr, tensor.size * tensor.itemsize, transport_flags)
        all_registered_memories = {}
        all_registered_memories[self.my_rank] = local_reg_memory
        future_memories = {}
        for rank in connections:
            self.communicator.send_memory_on_setup(local_reg_memory, rank, 0)
            future_memories[rank] = self.communicator.recv_memory_on_setup(rank, 0)
        self.communicator.setup()
        for rank in connections:
            all_registered_memories[rank] = future_memories[rank].get()
        return all_registered_memories

    def make_semaphore(
        self,
        connections: dict[int, Connection],
        semaphore_type: Type[Host2HostSemaphore] or Type[Host2DeviceSemaphore] or Type[SmDevice2DeviceSemaphore],
    ) -> dict[int, Host2HostSemaphore]:
        semaphores = {}
        for rank in connections:
            semaphores[rank] = semaphore_type(self.communicator, connections[rank])
        self.communicator.setup()
        return semaphores

    def make_sm_channels(self, tensor: cp.ndarray, connections: dict[int, Connection]) -> dict[int, SmChannel]:
        semaphores = self.make_semaphore(connections, SmDevice2DeviceSemaphore)
        registered_memories = self.register_tensor_with_connections(tensor, connections)
        channels = {}
        for rank in connections:
            channels[rank] = SmChannel(semaphores[rank], registered_memories[rank], tensor.data.ptr)
        return channels

    def make_sm_channels_with_packet(
        self, tensor: cp.ndarray, packetTensor: cp.ndarray, connections: dict[int, Connection]
    ) -> dict[int, SmChannel]:
        semaphores = self.make_semaphore(connections, SmDevice2DeviceSemaphore)
        registered_memories = self.register_tensor_with_connections(packetTensor, connections)
        channels = {}
        for rank in connections:
            channels[rank] = SmChannel(
                semaphores[rank],
                registered_memories[rank],
                tensor.data.ptr,
                packetTensor.data.ptr,
            )
        return channels

    def make_proxy_channels_with_packet(
        self, proxy_service: ProxyService, tensor: cp.ndarray, connections: dict[int, Connection]
    ) -> dict[int, SmChannel]:
        semaphores = self.make_semaphore(connections, Host2DeviceSemaphore)
        registered_memories = self.register_tensor_with_connections(tensor, connections)
        memory_ids = {}
        semaphore_ids = {}
        for rank in registered_memories:
            memory_ids[rank] = proxy_service.add_memory(registered_memories[rank])
        for rank in semaphores:
            semaphore_ids[rank] = proxy_service.add_semaphore(semaphores[rank])
        channels = {}
        for rank in semaphores:
            channels[rank] = SimpleProxyChannel(
                proxy_service.proxy_channel(semaphore_ids[rank]),
                memory_ids[rank],
                memory_ids[self.my_rank],
            )
        return channels
