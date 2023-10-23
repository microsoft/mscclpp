import os
import cupy as cp
from test.mscclpp_group import MscclppGroup
from test.utils import KernelBuilder, pack
from mscclpp import Transport
from mpi4py import MPI
import netifaces as ni

class MscclppOp():
    def __init__(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        network_interface = "eth0"
        my_ip = ni.ifaddresses(network_interface)[ni.AF_INET][0]["addr"]
        root_ip = comm.bcast(my_ip, root=0)
        ifIpPortTrio = network_interface + ":" + root_ip + ":50000"  # some random port

        self.group = MscclppGroup(interfaceIpPortTrio=ifIpPortTrio, rank=rank, size=size)
        self.group.barrier()


    def make_callback1(self, memory):
        self.memory = memory
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = ""
        if memory.dtype == cp.float16:
            type_str = "__half"
        elif memory.dtype == cp.float32:
            type_str = "float"
        elif memory.dtype == cp.int32:
            type_str = "int"
        else:
            raise RuntimeError("Unknown data type")

        # create a sm_channel for each remote neighbor
        self.sm_channels = self.group.make_sm_channels(self.memory, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(file="allreduce1.cu", kernel_name="allreduce1", file_dir=file_dir, macro_dict={"TYPE": type_str}).get_compiled_kernel()
        self.params = b""
        self.device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.device_handles.append(self.sm_channels[rank].device_handle().raw)
        self.params += pack(cp.asarray(memoryview(b"".join(self.device_handles)), dtype=cp.uint8), self.memory, self.group.my_rank, self.group.nranks, self.memory.size)

        def _make_call(stream_ptr):
            self.kernel.launch_kernel(self.params, 24, 1024, 0, stream_ptr)

        return _make_call
