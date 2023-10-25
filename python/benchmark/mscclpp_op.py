import os
import cupy as cp
from test.mscclpp_group import MscclppGroup
from test.utils import KernelBuilder, pack
from mscclpp import Transport, ProxyService
from mpi4py import MPI

def type_to_str(dtype):
    if dtype == cp.float16:
        return "__half"
    elif dtype == cp.float32:
        return "float"
    elif dtype == cp.int32:
        return "int"
    else:
        raise RuntimeError("Unknown data type")


class MscclppAllReduce1():
    def __init__(self, group: MscclppGroup, memory: cp.ndarray):
        self.group = group
        self.memory = memory
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

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

    def __call__(self, stream_ptr):
        self.kernel.launch_kernel(self.params, 24, 1024, 0, stream_ptr)
        return self.memory

class MscclppAllReduce2:
    def __init__(self, group: MscclppGroup, memory: cp.ndarray, memory_out: cp.ndarray):
        self.group = group
        self.memory = memory
        self.memory_out = memory_out
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

        self.scratch = cp.zeros(self.memory.size*8, dtype=self.memory.dtype)
        # create a sm_channel for each remote neighbor
        self.sm_channels = self.group.make_sm_channels_with_scratch(self.memory, self.scratch, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(file="allreduce1.cu", kernel_name="allreduce2", file_dir=file_dir, macro_dict={"TYPE": type_str}).get_compiled_kernel()
        self.params = b""
        self.device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.device_handles.append(self.sm_channels[rank].device_handle().raw)
        self.params += pack(cp.asarray(memoryview(b"".join(self.device_handles)), dtype=cp.uint8), self.memory, self.scratch, self.memory_out, self.group.my_rank, self.group.nranks, self.memory.size)

    def __call__(self, stream_ptr):
        self.kernel.launch_kernel(self.params, 21, 512, 0, stream_ptr)
        return self.memory_out

class MscclppAllReduce3:
    def __init__(self, group: MscclppGroup, memory: cp.ndarray, proxy_service: ProxyService):
        self.group = group
        self.memory = memory
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

        self.proxy_service = proxy_service
        self.scratch = cp.zeros(self.memory.size, dtype=self.memory.dtype)

        # create a sm_channel for each remote neighbor
        self.fst_round_proxy_chans = self.group.make_proxy_channels_with_scratch(self.proxy_service, self.memory, self.scratch, self.connections)
        self.snd_round_proxy_chans = self.group.make_proxy_channels(self.proxy_service, self.memory, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(file="allreduce1.cu", kernel_name="allreduce3", file_dir=file_dir, macro_dict={"TYPE": type_str}).get_compiled_kernel()
        self.params = b""
        self.fst_device_handles = []
        self.snd_device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.fst_device_handles.append(self.fst_round_proxy_chans[rank].device_handle().raw)
                self.snd_device_handles.append(self.snd_round_proxy_chans[rank].device_handle().raw)
        self.params += pack(cp.asarray(memoryview(b"".join(self.fst_device_handles)), dtype=cp.uint8), cp.asarray(memoryview(b"".join(self.snd_device_handles)), dtype=cp.uint8), self.memory, self.scratch, self.group.my_rank, self.group.nranks, self.memory.size)

    def __call__(self, stream_ptr):
        self.kernel.launch_kernel(self.params, 24, 1024, 0, stream_ptr)
        return self.memory
