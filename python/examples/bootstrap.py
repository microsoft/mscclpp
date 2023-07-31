# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import mscclpp
import argparse
import multiprocessing as mp
import logging
import torch
import sys

IB_TRANSPORTS = [
    mscclpp.Transport.IB0,
    mscclpp.Transport.IB1,
    mscclpp.Transport.IB2,
    mscclpp.Transport.IB3,
    mscclpp.Transport.IB4,
    mscclpp.Transport.IB5,
    mscclpp.Transport.IB6,
    mscclpp.Transport.IB7,
]

# Use to hold the sm channels so they don't get garbage collected
sm_channels = []


def setup_connections(comm, rank, world_size, element_size, proxy_service):
    simple_proxy_channels = []
    sm_semaphores = []
    connections = []
    remote_memories = []
    memory = torch.zeros(element_size, dtype=torch.int32)
    memory = memory.to("cuda")

    transport_flag = mscclpp.TransportFlags(IB_TRANSPORTS[rank]) | mscclpp.Transport.CudaIpc
    ptr = memory.data_ptr()
    size = memory.numel() * memory.element_size()
    reg_mem = comm.register_memory(ptr, size, transport_flag)

    for r in range(world_size):
        if r == rank:
            continue
        conn = comm.connect_on_setup(r, 0, mscclpp.Transport.CudaIpc)
        connections.append(conn)
        comm.send_memory_on_setup(reg_mem, r, 0)
        remote_mem = comm.recv_memory_on_setup(r, 0)
        remote_memories.append(remote_mem)
    comm.setup()

    # Create simple proxy channels
    for i, conn in enumerate(connections):
        proxy_channel = mscclpp.SimpleProxyChannel(
            proxy_service.device_channel(proxy_service.add_semaphore(conn)),
            proxy_service.add_memory(remote_memories[i].get()),
            proxy_service.add_memory(reg_mem),
        )
        simple_proxy_channels.append(mscclpp.device_handle(proxy_channel))
    comm.setup()

    # Create sm channels
    for i, conn in enumerate(connections):
        sm_chan = mscclpp.SmDevice2DeviceSemaphore.create(comm, conn)
        sm_semaphores.append(sm_chan)
    comm.setup()

    for i, conn in enumerate(sm_semaphores):
        sm_chan = mscclpp.SmChannel(sm_semaphores[i], remote_memories[i].get(), ptr)
        sm_channels.append(sm_chan)
    return simple_proxy_channels, [mscclpp.device_handle(sm_chan) for sm_chan in sm_channels]


def run(rank, args):
    world_size = args.gpu_number
    torch.cuda.set_device(rank)

    boot = mscclpp.TcpBootstrap.create(rank, world_size)
    boot.initialize(args.if_ip_port_trio)
    comm = mscclpp.Communicator(boot)
    proxy_service = mscclpp.ProxyService(comm)

    logging.info("Rank: %d, setting up connections", rank)
    setup_connections(comm, rank, world_size, args.num_elements, proxy_service)

    logging.info("Rank: %d, starting proxy service", rank)
    proxy_service.start_proxy()


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("if_ip_port_trio", type=str)
    parser.add_argument("-n", "--num-elements", type=int, default=10)
    parser.add_argument("-g", "--gpu_number", type=int, default=2)
    args = parser.parse_args()
    processes = []

    for rank in range(args.gpu_number):
        p = mp.Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
