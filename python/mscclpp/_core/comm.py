# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import Type

import cupy as cp
from mscclpp._mscclpp import (
    CppCommunicator,
    CppConnection,
    connect_nvls_collective,
    CppEndpointConfig,
    CppSemaphore,
    CppProxyService,
    CppRegisteredMemory,
    CppPortChannel,
    CppMemoryChannel,
    CppTcpBootstrap,
    CppTransport,
    CppTransportFlags,
)
import numpy as np
import pickle

from mscclpp.utils import is_torch_tensor

__all__ = ["CommGroup"]


class CommGroup:
    def __init__(
        self,
        mpi_comm: "mpi4py.MPI.Comm" = None,
        torch_group: "dist.ProcessGroup" = None,
        interfaceIpPortTrio: str = "",
        rank: int = None,
        size: int = None,
    ):
        if interfaceIpPortTrio == "" and (mpi_comm is not None or torch_group is not None):
            uniq_id = None
            rank, size = (
                (mpi_comm.Get_rank(), mpi_comm.Get_size())
                if mpi_comm is not None
                else (torch_group.rank(), torch_group.size())
            )
            self.bootstrap = CppTcpBootstrap.create(rank, size)
            if rank == 0:
                uniq_id = self.bootstrap.create_unique_id()
            if mpi_comm is not None:
                import mpi4py

                uniq_id_global = mpi_comm.bcast(uniq_id, 0)
            else:
                import torch
                import torch.distributed as dist

                if rank == 0:
                    uniq_id_global = uniq_id
                    pickled_data = pickle.dumps(uniq_id)
                    data_tensor = torch.frombuffer(bytearray(pickled_data), dtype=torch.uint8).clone()
                else:
                    data_tensor = torch.zeros(256, dtype=torch.uint8)
                dist.broadcast(data_tensor, src=0, group=torch_group)
                uniq_id_global = pickle.loads(data_tensor.numpy().tobytes())
            self.bootstrap.initialize(uniq_id_global)
        elif not interfaceIpPortTrio == "":
            assert rank >= 0 and size >= 1
            self.bootstrap = CppTcpBootstrap.create(rank, size)
            self.bootstrap.initialize(interfaceIpPortTrio)
        else:
            raise RuntimeError("Either the interface or mpi_group need to be specified")
        self.communicator = CppCommunicator(self.bootstrap)
        self.my_rank = self.bootstrap.get_rank()
        self.nranks = self.bootstrap.get_n_ranks()
        self.nranks_per_node = self.bootstrap.get_n_ranks_per_node()

    def barrier(self):
        self.bootstrap.barrier()

    def send(self, tensor: np.ndarray, peer: int, tag: int):
        self.bootstrap.send(tensor.ctypes.data, tensor.size * tensor.itemsize, peer, tag)

    def recv(self, tensor: np.ndarray, peer: int, tag: int):
        self.bootstrap.recv(tensor.ctypes.data, tensor.size * tensor.itemsize, peer, tag)

    def my_ib_device(self, local_rank: int) -> CppTransport:
        if local_rank == 0:
            return CppTransport.IB0
        if local_rank == 1:
            return CppTransport.IB1
        if local_rank == 2:
            return CppTransport.IB2
        if local_rank == 3:
            return CppTransport.IB3
        if local_rank == 4:
            return CppTransport.IB4
        if local_rank == 5:
            return CppTransport.IB5
        if local_rank == 6:
            return CppTransport.IB6
        if local_rank == 7:
            return CppTransport.IB7
        else:
            assert False  # only 8 IBs are supported

    def make_connection(
        self,
        all_ranks: list[int],
        endpoints: CppEndpointConfig | CppTransport | dict[int, CppEndpointConfig] | dict[int, CppTransport],
        use_switch: bool = False,
    ) -> dict[int, CppConnection]:
        if type(endpoints) is CppTransport:
            endpoints = CppEndpointConfig(endpoints)
        elif type(endpoints) is dict:
            endpoints = {k: CppEndpointConfig(v) if type(v) is CppTransport else v for k, v in endpoints.items()}
        connections = {}
        for rank in all_ranks:
            if type(endpoints) is dict:
                endpoint = endpoints[rank]
            else:
                endpoint = endpoints
            if endpoint.transport == CppTransport.CudaIpc and use_switch:
                return connect_nvls_collective(self.communicator, all_ranks, 2**30)
            else:
                connections[rank] = self.communicator.connect(endpoint, rank)
        connections = {rank: connections[rank].get() for rank in connections}
        return connections

    def register_tensor_with_connections(
        self, tensor: Type[cp.ndarray] | Type[np.ndarray], connections: dict[int, CppConnection]
    ) -> dict[int, CppRegisteredMemory]:
        local_reg_memory = self.register_local_memory(tensor, connections)
        all_registered_memories = {}
        all_registered_memories[self.my_rank] = local_reg_memory
        future_memories = {}
        for rank in connections:
            self.communicator.send_memory(local_reg_memory, rank)
            future_memories[rank] = self.communicator.recv_memory(rank)
        for rank in connections:
            all_registered_memories[rank] = future_memories[rank].get()
        return all_registered_memories

    def _register_memory_with_connections(
        self, memory: CppRegisteredMemory, connections: dict[int, CppConnection]
    ) -> dict[int, CppRegisteredMemory]:
        all_registered_memories = {}
        all_registered_memories[self.my_rank] = memory
        future_memories = {}
        for rank in connections:
            self.communicator.send_memory(memory, rank)
            future_memories[rank] = self.communicator.recv_memory(rank)
        for rank in connections:
            all_registered_memories[rank] = future_memories[rank].get()
        return all_registered_memories

    def make_semaphores(self, connections: dict[int, CppConnection]) -> dict[int, CppSemaphore]:
        future_semaphores = {}
        for rank in connections:
            future_semaphores[rank] = self.communicator.build_semaphore(connections[rank], rank)
        return {rank: future.get() for rank, future in future_semaphores.items()}

    def make_memory_channels(
        self, tensor: cp.ndarray, connections: dict[int, CppConnection]
    ) -> dict[int, CppMemoryChannel]:
        semaphores = self.make_semaphores(connections)
        registered_memories = self.register_tensor_with_connections(tensor, connections)
        channels = {}
        for rank in connections:
            channels[rank] = CppMemoryChannel(
                semaphores[rank], registered_memories[rank], registered_memories[self.my_rank]
            )
        return channels

    def make_memory_channels_with_scratch(
        self,
        tensor: cp.ndarray,
        registeredScratchBuffer: CppRegisteredMemory,
        connections: dict[int, CppConnection],
    ) -> dict[int, CppMemoryChannel]:
        semaphores = self.make_semaphores(connections)
        registered_memories = self._register_memory_with_connections(registeredScratchBuffer, connections)
        channels = {}
        tensor_data_ptr = tensor.data_ptr() if is_torch_tensor(tensor) else tensor.data.ptr
        tensor_size = (
            tensor.numel() * tensor.element_size() if is_torch_tensor(tensor) else tensor.size * tensor.itemsize
        )
        local_registered_memory = self.communicator.register_memory(tensor_data_ptr, tensor_size, CppTransportFlags())
        scratch_data_ptr = registeredScratchBuffer.data()
        for rank in connections:
            channels[rank] = CppMemoryChannel(
                semaphores[rank], registered_memories[rank], local_registered_memory, scratch_data_ptr
            )
        return channels

    def make_port_channels(
        self, proxy_service: CppProxyService, tensor: cp.ndarray, connections: dict[int, CppConnection]
    ) -> dict[int, CppPortChannel]:
        semaphores = self.make_semaphores(connections)
        registered_memories = self.register_tensor_with_connections(tensor, connections)
        memory_ids = {}
        semaphore_ids = {}
        for rank in registered_memories:
            memory_ids[rank] = proxy_service.add_memory(registered_memories[rank])
        for rank in semaphores:
            semaphore_ids[rank] = proxy_service.add_semaphore(semaphores[rank])
        channels = {}
        for rank in semaphores:
            channels[rank] = proxy_service.port_channel(semaphore_ids[rank], memory_ids[rank], memory_ids[self.my_rank])
        return channels

    def make_port_channels_with_scratch(
        self,
        proxy_service: CppProxyService,
        tensor: cp.ndarray,
        registeredScratchBuffer: CppRegisteredMemory,
        connections: dict[int, CppConnection],
    ) -> dict[int, CppPortChannel]:
        transport_flags = CppTransportFlags()
        for rank in connections:
            transport_flags |= connections[rank].transport()
        data_ptr = (
            tensor.data.ptr
            if isinstance(tensor, cp.ndarray)
            else tensor.data_ptr() if is_torch_tensor(tensor) else tensor.ctypes.data
        )
        tensor_size = (
            tensor.numel() * tensor.element_size() if is_torch_tensor(tensor) else tensor.size * tensor.itemsize
        )
        local_reg_memory = self.communicator.register_memory(data_ptr, tensor_size, transport_flags)

        semaphores = self.make_semaphores(connections)
        registered_memories = self._register_memory_with_connections(registeredScratchBuffer, connections)
        memory_ids = {}
        semaphore_ids = {}
        for rank in registered_memories:
            if rank == self.my_rank:
                memory_ids[self.my_rank] = proxy_service.add_memory(local_reg_memory)
            else:
                memory_ids[rank] = proxy_service.add_memory(registered_memories[rank])
        for rank in semaphores:
            semaphore_ids[rank] = proxy_service.add_semaphore(semaphores[rank])
        channels = {}
        for rank in semaphores:
            channels[rank] = proxy_service.port_channel(semaphore_ids[rank], memory_ids[rank], memory_ids[self.my_rank])
        return channels

    def register_semaphore_with_proxy(
        self, proxy_service: CppProxyService, connections: dict[int, CppConnection]
    ) -> dict[int, CppPortChannel]:
        semaphores = self.make_semaphores(connections)
        semaphore_ids = {}
        for rank in semaphores:
            semaphore_ids[rank] = proxy_service.add_semaphore(semaphores[rank])
        channels = {}
        for rank in semaphores:
            channels[rank] = proxy_service.base_port_channel(semaphore_ids[rank])
        return channels

    def register_memory_with_proxy(
        self, proxy_service: CppProxyService, tensor: cp.ndarray, connections: dict[int, CppConnection]
    ) -> dict[int, int]:
        registered_memories = self.register_tensor_with_connections(tensor, connections)
        memory_ids = {}
        for rank in registered_memories:
            memory_ids[rank] = proxy_service.add_memory(registered_memories[rank])
        return memory_ids

    def register_local_memory(self, tensor: cp.ndarray, connections: dict[int, CppConnection]) -> CppRegisteredMemory:
        transport_flags = CppTransportFlags()
        for rank in connections:
            transport_flags |= connections[rank].transport()
        data_ptr = (
            tensor.data.ptr
            if isinstance(tensor, cp.ndarray)
            else tensor.data_ptr() if is_torch_tensor(tensor) else tensor.ctypes.data
        )
        tensor_size = (
            tensor.numel() * tensor.element_size() if is_torch_tensor(tensor) else tensor.size * tensor.itemsize
        )
        return self.communicator.register_memory(data_ptr, tensor_size, transport_flags)
