import time

import netifaces as ni
import pytest
import torch
from applied_comms._cpp import _ext
from applied_comms._utils import KernelBase, pack
from mscclpp import (
    Fifo,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    ProxyService,
    SmDevice2DeviceSemaphore,
    Transport,
)
from mscclpp_group import MscclppGroup
from mpi4py import MPI

from applied_models.tests import PipeLayout, parametrize_layouts

ethernet_interface_name = "eth0"


def all_ranks_on_the_same_node(layout: PipeLayout):
    if (ethernet_interface_name in ni.interfaces()) is False:
        pytest.skip(f"{ethernet_interface_name} is not an interface to use on this node")
    my_ip = ni.ifaddresses(ethernet_interface_name)[ni.AF_INET][0]["addr"]
    root_ip = layout.comm.bcast(my_ip, 0)
    last_rank_ip = layout.comm.bcast(my_ip, layout.comm.size - 1)
    return last_rank_ip == root_ip


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("ifIpPortTrio", ["eth0:localhost:50000", ethernet_interface_name, ""])
def test_group_with_ip(layout: PipeLayout, ifIpPortTrio: str):
    if (ethernet_interface_name in ni.interfaces()) is False:
        pytest.skip(f"{ethernet_interface_name} is not an interface to use on this node")
    my_ip = ni.ifaddresses(ethernet_interface_name)[ni.AF_INET][0]["addr"]
    root_ip = layout.comm.bcast(my_ip, 0)
    if ifIpPortTrio == ethernet_interface_name:
        ifIpPortTrio += ":" + root_ip + ":50000"  # some random port

    if all_ranks_on_the_same_node(layout) is False and "localhost" in ifIpPortTrio:
        # ranks are on different nodes
        pytest.skip("this case is not supported as localhost will be different for different nodes")

    group = MscclppGroup(layout, ifIpPortTrio)

    nelem = 1024
    memory = torch.zeros(nelem, dtype=torch.int32, device="cpu")
    nelemPerRank = nelem // group.nranks
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = (
        group.my_rank + 1
    )
    memory_expected = torch.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1

    for rank in range(group.nranks):
        if rank == group.my_rank:
            continue
        group.send(
            memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))],
            rank,
            0,
        )
    for rank in range(group.nranks):
        if rank == group.my_rank:
            continue
        group.recv(memory[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))], rank, 0)

    assert torch.equal(memory, memory_expected)


def create_and_connect(comm: MPI.Comm, transport: str):
    if transport == "NVLink" and all_ranks_on_the_same_node(layout) is False:
        pytest.skip("cannot use nvlink for cross node")
    group = MscclppGroup(comm)

    remote_nghrs = list(range(comm.size))
    remote_nghrs.remove(comm.rank)
    if transport == "NVLink":
        tran = Transport.CudaIpc
    elif transport == "IB":
        tran = group.my_ib_device(group.my_rank % 8)
    else:
        assert False
    connections = group.make_connection(remote_nghrs, tran)
    return group, connections


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
def test_group_with_connections(layout: PipeLayout, transport: str):
    create_and_connect(layout, transport)


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
def test_connection_write(layout: PipeLayout, transport: Transport, nelem: int):
    group, connections = create_and_connect(layout, transport)
    memory = torch.zeros(nelem, dtype=torch.int32, device="cuda")
    nelemPerRank = nelem // group.nranks
    sizePerRank = nelemPerRank * memory.element_size()
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = (
        group.my_rank + 1
    )
    memory_expected = torch.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    torch.cuda.synchronize()
    group.barrier()
    all_reg_memories = group.register_tensor_with_connections(memory, connections)
    for rank in connections:
        connections[rank].write(
            all_reg_memories[rank],
            sizePerRank * group.my_rank,
            all_reg_memories[group.my_rank],
            sizePerRank * group.my_rank,
            sizePerRank,
        )
    poll_for = 100
    for i in range(poll_for):
        all_correct = torch.equal(memory, memory_expected)
        if all_correct:
            break
        time.sleep(0.1)
    for conn in connections:
        connections[conn].flush()
    torch.cuda.synchronize()
    group.barrier()
    assert all_correct


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_connection_write_and_signal(
    layout: PipeLayout, transport: Transport, nelem: int, device: str
):
    # this test starts with a random tensor on rank 0 and rotates it all the way through all ranks
    # and finally, comes back to rank 0 to make sure it matches all the original values

    if device == "cpu" and transport == "NVLink":
        pytest.skip("nvlink doesn't work with host allocated memory")
    group, connections = create_and_connect(layout, transport)
    if group.my_rank == 0:
        memory = torch.randn(nelem, dtype=torch.float32, device=device)
        memory_expected = memory.clone()
    else:
        memory = torch.zeros(nelem, dtype=torch.float32, device=device)

    signal_memory = torch.zeros(1, dtype=torch.int64, device=device)
    all_reg_memories = group.register_tensor_with_connections(memory, connections)
    all_signal_memories = group.register_tensor_with_connections(signal_memory, connections)

    next_rank = (group.my_rank + 1) % group.nranks
    bufferSize = nelem * memory.element_size()
    dummy_memory_on_cpu = torch.zeros(1, dtype=torch.int64, device="cpu")
    torch.cuda.synchronize()

    signal_val = 123
    if group.my_rank != 0:
        while signal_memory[0] != signal_val:
            time.sleep(0.1)
    connections[next_rank].write(
        all_reg_memories[next_rank], 0, all_reg_memories[group.my_rank], 0, bufferSize
    )
    connections[next_rank].flush()
    if group.my_rank == 0:
        memory[:] = 0
        torch.cuda.synchronize()
    connections[next_rank].update_and_sync(
        all_signal_memories[next_rank], 0, dummy_memory_on_cpu.data_ptr(), signal_val
    )
    all_correct = False
    if group.my_rank == 0:
        while signal_memory[0] != signal_val:
            time.sleep(0.1)
        all_correct = torch.equal(memory, memory_expected)
    torch.cuda.synchronize()
    group.barrier()
    all_correct = layout.comm.bcast(all_correct, 0)
    assert all_correct


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
def test_h2h_semaphores(layout: PipeLayout):
    group, connections = create_and_connect(layout, "IB")

    semaphores = group.make_semaphore(connections, Host2HostSemaphore)
    for rank in connections:
        semaphores[rank].signal()

    for rank in connections:
        semaphores[rank].wait()
    group.barrier()


class MscclppKernel(KernelBase):
    def __init__(
        self,
        test_name,
        my_rank=None,
        nranks=None,
        semaphore_or_channels=None,
        tensor=None,
        use_packet=False,
        scratch=None,
        fifo=None,
    ):
        if test_name == "h2d_semaphore":
            super().__init__(
                file="h2d_semaphore_test.ci",
                args=dict(KERNEL="h2d_semaphore", N_SHARDS=nranks),
            )
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "d2d_semaphore":
            super().__init__(
                file="d2d_semaphore_test.ci",
                args=dict(KERNEL="d2d_semaphore", N_SHARDS=nranks),
            )
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "sm_channel":
            super().__init__(
                file="sm_channel_test.ci",
                args=dict(
                    KERNEL="sm_channel",
                    N_SHARDS=nranks,
                    TD="int",
                    USE_PACKET="true" if use_packet else "false",
                ),
            )
            self.nblocks = nranks
            self.nthreads = 1024
        elif test_name == "fifo":
            super().__init__(
                file="fifo_test.ci",
                args=dict(KERNEL="fifo"),
            )
            self.nblocks = 1
            self.nthreads = 1
        elif test_name == "proxy":
            super().__init__(
                file="proxy_test.ci",
                args=dict(KERNEL="proxy", N_SHARDS=nranks),
            )
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "simple_proxy_channel":
            super().__init__(
                file="simple_proxy_channel_test.ci",
                args=dict(
                    KERNEL="simple_proxy_channel",
                    N_SHARDS=nranks,
                    TD="int",
                    USE_PACKET="true" if use_packet else "false",
                ),
            )
            self.nblocks = 1
            self.nthreads = 1024
        else:
            assert False

        self.params = b""
        if test_name in ["h2d_semaphore", "d2d_semaphore", "sm_channel", "simple_proxy_channel"]:
            first_arg = next(iter(semaphore_or_channels.values()))
            size_of_semaphore_or_channels = len(first_arg.device_handle().raw)
            device_handles = []
            for rank in range(nranks):
                if rank == my_rank:
                    device_handles.append(
                        bytes(size_of_semaphore_or_channels)
                    )  # just zeros for semaphores that do not exist
                else:
                    device_handles.append(semaphore_or_channels[rank].device_handle().raw)
            self.params += b"".join(device_handles) + pack(my_rank, nranks)
            if test_name == "sm_channel":
                self.params += pack(tensor.numel())
            if test_name == "simple_proxy_channel":
                self.params += pack(tensor, scratch, tensor.numel())
        elif test_name == "fifo":
            self.params = fifo.device_handle().raw
        elif test_name == "proxy":
            semaphore_device_handles = [
                semaphore.device_handle().raw for semaphore in semaphore_or_channels
            ]
            self.params = pack(my_rank, nranks) + fifo.raw + b"".join(semaphore_device_handles)

    def __call__(self):
        return self._kernel.launch_kernel(self.params, self.nblocks, self.nthreads, 0, None)


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
def test_h2d_semaphores(layout: PipeLayout, transport: str):
    group, connections = create_and_connect(layout, transport)

    semaphores = group.make_semaphore(connections, Host2DeviceSemaphore)
    kernel = MscclppKernel("h2d_semaphore", group.my_rank, group.nranks, semaphores)
    kernel()
    for rank in semaphores:
        semaphores[rank].signal()
    torch.cuda.synchronize()
    group.barrier()


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
def test_d2d_semaphores(layout: PipeLayout):
    group, connections = create_and_connect(layout, "NVLink")

    semaphores = group.make_semaphore(connections, SmDevice2DeviceSemaphore)
    kernel = MscclppKernel("d2d_semaphore", group.my_rank, group.nranks, semaphores)
    kernel()
    torch.cuda.synchronize()
    group.barrier()


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("use_packet", [False, True])
def test_sm_channels(layout: PipeLayout, nelem: int, use_packet: bool):
    group, connections = create_and_connect(layout, "NVLink")

    memory = torch.zeros(nelem, dtype=torch.int32, device="cuda")
    if use_packet:
        scratch = torch.zeros(nelem * 2, dtype=torch.int32, device="cuda")
    else:
        scratch = None
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.element_size()
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = (
        group.my_rank + 1
    )
    memory_expected = torch.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1

    if use_packet:
        channels = group.make_sm_channels_with_packet(memory, scratch, connections)
    else:
        channels = group.make_sm_channels(memory, connections)
    kernel = MscclppKernel(
        "sm_channel", group.my_rank, group.nranks, channels, memory, use_packet, scratch
    )

    kernel()
    torch.cuda.synchronize()
    group.barrier()
    assert torch.equal(memory, memory_expected)


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
def test_fifo(
    layout: PipeLayout,
):
    fifo = Fifo()
    kernel = MscclppKernel("fifo", fifo=fifo)

    kernel()
    poll_for = 100
    for i in range(poll_for):
        trigger = fifo.poll()
        if trigger.fst == 123:
            return
        time.sleep(0.1)
    assert False


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
def test_proxy(
    layout: PipeLayout,
    nelem: int,
    transport: str,
):
    group, connections = create_and_connect(layout, transport)

    memory = torch.zeros(nelem, dtype=torch.int32, device="cuda")
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.element_size()
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = (
        group.my_rank + 1
    )
    memory_expected = torch.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    torch.cuda.synchronize()
    group.barrier()
    all_reg_memories = group.register_tensor_with_connections(memory, connections)

    semaphores = group.make_semaphore(connections, Host2DeviceSemaphore)

    list_conn = []
    list_sem = []
    list_reg_mem = []
    first_conn = next(iter(connections.values()))
    first_sem = next(iter(semaphores.values()))
    for rank in range(group.nranks):
        if rank in connections:
            list_conn.append(connections[rank])
            list_sem.append(semaphores[rank])
        else:
            list_conn.append(first_conn)  # just for simplicity of indexing
            list_sem.append(first_sem)

        list_reg_mem.append(all_reg_memories[rank])

    proxy = _ext.MyProxyService(
        group.my_rank,
        group.nranks,
        nelem * memory.element_size(),
        list_conn,
        list_reg_mem,
        list_sem,
    )

    fifo_device_handle = proxy.fifo_device_handle()

    kernel = MscclppKernel(
        "proxy",
        my_rank=group.my_rank,
        nranks=group.nranks,
        semaphore_or_channels=list_sem,
        fifo=fifo_device_handle,
    )
    proxy.start()
    kernel()
    torch.cuda.synchronize()
    proxy.stop()
    group.barrier()
    assert torch.equal(memory, memory_expected)


@parametrize_layouts((1, 2), (1, 4), (1, 8), (1, 16))
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("transport", ["NVLink", "IB"])
@pytest.mark.parametrize("use_packet", [False, True])
def test_simple_proxy_channel(
    layout: PipeLayout,
    nelem: int,
    transport: str,
    use_packet: bool,
):
    group, connections = create_and_connect(layout, transport)

    memory = torch.zeros(nelem, dtype=torch.int32, device="cuda")
    if use_packet:
        scratch = torch.zeros(nelem * 2, dtype=torch.int32, device="cuda")
    else:
        scratch = torch.zeros(
            1, dtype=torch.int32, device="cuda"
        )  # just so that we can pass a valid ptr
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.element_size()
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = (
        group.my_rank + 1
    )
    memory_expected = torch.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    torch.cuda.synchronize()
    group.barrier()

    proxy_service = ProxyService()
    if use_packet:
        memory_to_register = scratch
    else:
        memory_to_register = memory
    simple_channels = group.make_proxy_channels_with_packet(
        proxy_service, memory_to_register, connections
    )

    kernel = MscclppKernel(
        "simple_proxy_channel",
        my_rank=group.my_rank,
        nranks=group.nranks,
        semaphore_or_channels=simple_channels,
        tensor=memory,
        use_packet=use_packet,
        scratch=scratch,
    )
    proxy_service.start_proxy()
    kernel()
    torch.cuda.synchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert torch.equal(memory, memory_expected)