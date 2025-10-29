import mscclpp
import mscclpp.comm as mscclpp_comm
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
    def __init__(self, comm: mscclpp_comm.CommGroup):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        self.registry = mscclpp.ExecutionPlanRegistry()
        self.executor = mscclpp.Executor(comm.communicator)
        mscclpp_native = mscclpp.compile(file = os.path.join(_abs_path, "customized_allgather.cu"))
        self.algo = mscclpp_native.create_allgather_algorithm()

    def all_gather(self, tensor: torch.Tensor, stream: torch.cuda.Stream = None):
        self.algo.launch()


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
    mscclpp_group = mscclpp_comm.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world_size)
    return CustomizedComm(mscclpp_group)



if __name__ == "__main__":
    main()