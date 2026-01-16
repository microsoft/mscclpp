# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_allgather.py
# For AMD: MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> GPU_MAX_HW_QUEUES=7 torchrun --nnodes=1 --nproc_per_node=8 customized_allgather.py

import mscclpp

import mscclpp.utils as mscclpp_utils
import torch
import os
import netifaces as ni
import ipaddress

_abs_path = os.path.dirname(os.path.abspath(__file__))


def interfaces_for_ip_netifaces(ip: str):
    target = ipaddress.ip_address(ip)
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if "addr" in link:
                    addr = ipaddress.ip_address(link["addr"])
                    if addr == target:
                        return interface
    return None


class CustomizedComm:
    def __init__(self, comm: mscclpp.CommGroup):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        self.executor = mscclpp.Executor(comm.communicator)
        mscclpp_native = mscclpp.compile_native(
            name="mscclpp_native", file=os.path.join(_abs_path, "customized_allgather.cu")
        )
        capsule = mscclpp_native.create_allgather_algorithm()
        self.algorithm = mscclpp.Algorithm.create_from_native_capsule(capsule)

    def all_gather(self, tensor: torch.Tensor, out_tensor: torch.Tensor, stream: torch.cuda.Stream = None):
        self.algorithm.execute(
            self.comm.communicator,
            tensor.data_ptr(),
            out_tensor.data_ptr(),
            tensor.nbytes,
            out_tensor.nbytes,
            mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            stream=stream.cuda_stream if stream is not None else 0,
        )

    def barrier_cpu(self):
        self.comm.barrier()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    torch.cuda.set_device(local_rank)
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    mscclpp_group = mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world_size)
    local_tensor_size = 1 << 20
    out_tensor = torch.randn(local_tensor_size * world_size, device="cuda", dtype=torch.float32)
    tensor = out_tensor[rank * local_tensor_size : (rank + 1) * local_tensor_size]
    comm = CustomizedComm(mscclpp_group)
    comm.barrier_cpu()
    comm.all_gather(tensor, out_tensor, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    comm = None
    print(f"Rank {rank} allgather completed successfully.")


if __name__ == "__main__":
    main()
