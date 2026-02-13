# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm_with_default_algo.py

import os
import torch
import mscclpp.utils as mscclpp_utils
import mscclpp
import mscclpp.ext
import netifaces as ni
import ipaddress


def load_algorithms(scratch_buffer: torch.tensor, rank: int) -> mscclpp.AlgorithmCollection:
    collection_builder = mscclpp.ext.AlgorithmCollectionBuilder()
    return collection_builder.build_default_algorithms(
        scratch_buffer=scratch_buffer.data_ptr(),
        scratch_buffer_size=scratch_buffer.nbytes,
        rank=rank,
    )


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


def to_mscclpp_reduce_op(op: torch.distributed.ReduceOp) -> mscclpp.ReduceOp:
    if op == torch.distributed.ReduceOp.SUM:
        return mscclpp.ReduceOp.SUM
    elif op == torch.distributed.ReduceOp.MIN:
        return mscclpp.ReduceOp.MIN
    else:
        raise ValueError(f"unsupported op: {op}")


class CustomizedComm:
    def __init__(self, comm: mscclpp.CommGroup):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        dlpack = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        flag_dlpack = mscclpp.RawGpuBuffer(128 * 4).to_dlpack(data_type=str(torch.uint8))
        self.flag_buffer = torch.utils.dlpack.from_dlpack(flag_dlpack)
        algorithms = load_algorithms(scratch_buffer=self.scratch_buffer, rank=self.rank)
        self._algorithm_nvls_packet = [
            algo
            for algo in algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_nvls_packet"
        ][0]
        self._algorithm_nvls_nonzero_copy = [
            algo
            for algo in algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_nvls_with_copy"
        ][0]

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream: torch.cuda.Stream = None):
        assert op == torch.distributed.ReduceOp.SUM
        algo = None
        if tensor.nbytes < 1 << 20:
            algo = self._algorithm_nvls_packet
        else:
            algo = self._algorithm_nvls_nonzero_copy
        algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=to_mscclpp_reduce_op(op),
            stream=stream.cuda_stream if stream is not None else 0,
        )

    def barrier(self):
        tensor = torch.empty(1, dtype=torch.float, device=torch.device("cuda"))
        self.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, stream=torch.cuda.current_stream())

    def destroy(self):
        self._algorithm_nvls_nonzero_copy = None
        self._algorithm_nvls_packet = None
        self.scratch_buffer = None
        self.comm = None


def init_dist() -> CustomizedComm:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    mscclpp_group = mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world)
    return CustomizedComm(mscclpp_group)


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    comm = init_dist()
    comm.barrier()
    input_data = torch.randn(1 << 22, dtype=torch.float16, device=torch.device("cuda"))
    comm.all_reduce(input_data, op=torch.distributed.ReduceOp.SUM, stream=torch.cuda.current_stream())
    comm.barrier()
    comm.destroy()
    print(f"rank {local} All-reduce operation completed successfully.")


if __name__ == "__main__":
    main()
