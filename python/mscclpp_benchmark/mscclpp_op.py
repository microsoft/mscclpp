import os
import cupy as cp
import ctypes
from mscclpp import Transport, ProxyService, MemoryDevice2DeviceSemaphore
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder, GpuBuffer, pack


IB_TRANSPORTS = [
    Transport.IB0,
    Transport.IB1,
    Transport.IB2,
    Transport.IB3,
    Transport.IB4,
    Transport.IB5,
    Transport.IB6,
    Transport.IB7,
]


def type_to_str(dtype):
    if dtype == cp.float16:
        return "__half"
    elif dtype == cp.float32:
        return "float"
    elif dtype == cp.int32:
        return "int"
    else:
        raise RuntimeError("Unknown data type")


class MscclppAllReduce1:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        memory: cp.ndarray,
        read_only: int = 1,
        block_size: int = 1024,
        nblocks: int = 24,
    ):
        self.group = group
        self.memory = memory
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

        # create a memory_channel for each remote neighbor
        self.memory_channels = self.group.make_memory_channels(self.memory, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu",
            kernel_name="allreduce1",
            file_dir=file_dir,
            macro_dict={"TYPE": type_str},
        ).get_compiled_kernel()
        self.device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.device_handles.append(self.memory_channels[rank].device_handle().raw)

        self.device_handles_cp = cp.asarray(memoryview(b"".join(self.device_handles)), dtype=cp.uint8)

        self.set_params(nblocks, block_size, read_only)

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory

    def set_params(self, nblocks, block_size, read_only):
        self.nblocks = nblocks
        self.block_size = block_size
        self.read_only = read_only
        self.params = b""
        self.params += pack(
            self.device_handles_cp,
            self.memory,
            self.group.my_rank,
            self.group.nranks,
            ctypes.c_size_t(self.memory.size),
            self.read_only,
        )

    def auto_tune(self):
        nblocks_to_try = [8, 12, 16, 24, 32, 48, 64, 72, 96, 108]
        block_size_to_try = [256, 512, 1024]
        read_only_to_try = [0, 1]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                for read_only in read_only_to_try:
                    self.set_params(nblocks, block_size, read_only)
                    yield nblocks, block_size, read_only


class MscclppAllReduce2:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        memory: cp.ndarray,
        memory_out: cp.ndarray,
        block_size: int = 512,
        nblocks: int = 21,
    ):
        self.group = group
        self.memory = memory
        self.memory_out = memory_out
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

        self.scratch = GpuBuffer(self.memory.size * 8, dtype=self.memory.dtype)
        # create a memory_channel for each remote neighbor
        self.memory_channels = self.group.make_memory_channels_with_scratch(self.memory, self.scratch, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu", kernel_name="allreduce2", file_dir=file_dir, macro_dict={"TYPE": type_str}
        ).get_compiled_kernel()
        self.device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.device_handles.append(self.memory_channels[rank].device_handle().raw)

        self.device_handles_cp = cp.asarray(memoryview(b"".join(self.device_handles)), dtype=cp.uint8)

        self.set_params(nblocks, block_size)

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory_out

    def set_params(self, nblocks, block_size):
        self.nblocks = nblocks
        self.block_size = block_size

        self.params = b""
        self.params += pack(
            self.device_handles_cp,
            self.memory,
            self.scratch,
            self.memory_out,
            self.group.my_rank,
            self.group.nranks,
            ctypes.c_size_t(self.memory.size),
        )

    def auto_tune(self):
        nblocks_to_try = [21, 42, 63, 84, 105]
        block_size_to_try = [256, 512, 1024]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                self.set_params(nblocks, block_size)
                yield nblocks, block_size


class MscclppAllReduce3:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        memory: cp.ndarray,
        proxy_service: ProxyService,
        block_size: int = 1024,
        nblocks: int = 24,
    ):
        self.group = group
        self.memory = memory
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        type_str = type_to_str(memory.dtype)

        self.proxy_service = proxy_service
        self.scratch = GpuBuffer(self.memory.size, dtype=self.memory.dtype)

        # create a memory_channel for each remote neighbor
        self.fst_round_port_chans = self.group.make_port_channels_with_scratch(
            self.proxy_service, self.memory, self.scratch, self.connections
        )
        self.snd_round_port_chans = self.group.make_port_channels(self.proxy_service, self.memory, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu", kernel_name="allreduce3", file_dir=file_dir, macro_dict={"TYPE": type_str}
        ).get_compiled_kernel()
        self.fst_device_handles = []
        self.snd_device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.fst_device_handles.append(self.fst_round_port_chans[rank].device_handle().raw)
                self.snd_device_handles.append(self.snd_round_port_chans[rank].device_handle().raw)
        self.fst_device_handles_cp = cp.asarray(memoryview(b"".join(self.fst_device_handles)), dtype=cp.uint8)
        self.snd_device_handles_cp = cp.asarray(memoryview(b"".join(self.snd_device_handles)), dtype=cp.uint8)

        self.set_params(nblocks, block_size)

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, 24, 1024, 0, stream)
        return self.memory

    def set_params(self, nblocks, block_size):
        self.nblocks = nblocks
        self.block_size = block_size
        self.params = b""
        self.params += pack(
            self.fst_device_handles_cp,
            self.snd_device_handles_cp,
            self.memory,
            self.scratch,
            self.group.my_rank,
            self.group.nranks,
            ctypes.c_size_t(self.memory.size),
        )

    def auto_tune(self):
        nblocks_to_try = [8, 12, 16, 24, 32, 48, 64, 72, 96, 108]
        block_size_to_try = [256, 512, 1024]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                self.set_params(nblocks, block_size)
                yield nblocks, block_size


class MscclppAllReduce4:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        memory: cp.ndarray,
        nranks_per_node: int,
        proxy_service: ProxyService,
        nblocks: int = 45,
        block_size: int = 512,
        pipeline_depth: int = 3,
    ):
        self.group = group
        self.memory = memory

        self.nranks_per_node = nranks_per_node
        in_same_node = lambda rank: rank // nranks_per_node == self.group.my_rank // nranks_per_node
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)
        transports = {}
        for rank in remote_nghrs:
            if in_same_node(rank):
                transports[rank] = Transport.CudaIpc
            else:
                transports[rank] = IB_TRANSPORTS[rank % nranks_per_node]

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, transports)
        type_str = type_to_str(memory.dtype)

        self.proxy_service = proxy_service
        self.scratch = GpuBuffer(self.memory.size, dtype=self.memory.dtype)
        same_node_connections = {rank: conn for rank, conn in self.connections.items() if in_same_node(rank)}
        # create a memory_channel for each remote neighbor
        self.memory_channels = self.group.make_memory_channels(self.memory, same_node_connections)
        self.reduce_scatter_port_channels = self.group.make_port_channels_with_scratch(
            self.proxy_service, self.memory, self.scratch, self.connections
        )
        self.all_gather_port_channels = self.group.make_port_channels(self.proxy_service, self.memory, self.connections)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu", kernel_name="allreduce4", file_dir=file_dir, macro_dict={"TYPE": type_str}
        ).get_compiled_kernel()
        self.mem_device_handles = []
        self.reduce_sactter_proxy_device_handles = []
        self.all_gather_proxy_device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank and in_same_node(rank):
                self.mem_device_handles.append(self.memory_channels[rank].device_handle().raw)
            if rank != self.group.my_rank:
                self.reduce_sactter_proxy_device_handles.append(
                    self.reduce_scatter_port_channels[rank].device_handle().raw
                )
                self.all_gather_proxy_device_handles.append(self.all_gather_port_channels[rank].device_handle().raw)

        self.mem_device_handles_cp = cp.asarray(memoryview(b"".join(self.mem_device_handles)), dtype=cp.uint8)
        self.reduce_sactter_proxy_device_handles_cp = cp.asarray(
            memoryview(b"".join(self.reduce_sactter_proxy_device_handles)), dtype=cp.uint8
        )
        self.all_gather_proxy_device_handles_cp = cp.asarray(
            memoryview(b"".join(self.all_gather_proxy_device_handles)), dtype=cp.uint8
        )

        self.set_params(nblocks, block_size, pipeline_depth)

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory

    def set_params(self, nblocks, block_size, pipeline_depth):
        self.nblocks = nblocks
        self.block_size = block_size
        self.pipeline_depth = pipeline_depth

        self.params = b""
        self.params += pack(
            self.mem_device_handles_cp,
            self.reduce_sactter_proxy_device_handles_cp,
            self.all_gather_proxy_device_handles_cp,
            self.memory,
            self.scratch,
            self.group.my_rank,
            self.nranks_per_node,
            self.group.nranks,
            bytes(4),  # padding for memory alignment
            ctypes.c_size_t(self.memory.size),
            self.pipeline_depth,
        )

    def auto_tune(self):
        nblocks_to_try = [24, 32, 40, 45, 48, 64, 72, 90, 96, 108]
        block_size_to_try = [256, 512, 1024]
        pipeline_depth_to_try = [1, 2, 3, 4]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                for pipeline_depth in pipeline_depth_to_try:
                    self.set_params(nblocks, block_size, pipeline_depth)
                    yield nblocks, block_size, pipeline_depth


class MscclppAllReduce5:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        memory: cp.ndarray,
        memory_out: cp.ndarray,
        nranks_per_node: int,
        proxy_service: ProxyService,
        nblocks: int = 21,
        block_size: int = 512,
    ):
        self.group = group
        self.memory = memory
        self.memory_out = memory_out

        self.nranks_per_node = nranks_per_node
        in_same_node = lambda rank: rank // nranks_per_node == self.group.my_rank // nranks_per_node
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)
        transports = {}
        for rank in remote_nghrs:
            if in_same_node(rank):
                transports[rank] = Transport.CudaIpc
            else:
                transports[rank] = IB_TRANSPORTS[rank % nranks_per_node]

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, transports)
        type_str = type_to_str(memory.dtype)

        self.proxy_service = proxy_service
        self.scratch = GpuBuffer(self.memory.size * 8, dtype=self.memory.dtype)
        self.put_buff = GpuBuffer(self.memory.size * 8 // nranks_per_node, dtype=self.memory.dtype)
        same_node_connections = {rank: conn for rank, conn in self.connections.items() if in_same_node(rank)}
        across_node_connections = {rank: conn for rank, conn in self.connections.items() if not in_same_node(rank)}
        # create a memory_channel for each remote neighbor
        self.memory_channels = self.group.make_memory_channels_with_scratch(
            self.memory, self.scratch, same_node_connections
        )
        self.port_channels = self.group.make_port_channels_with_scratch(
            self.proxy_service, self.put_buff, self.scratch, across_node_connections
        )
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu", kernel_name="allreduce5", file_dir=file_dir, macro_dict={"TYPE": type_str}
        ).get_compiled_kernel()
        self.mem_device_handles = []
        self.proxy_device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank and in_same_node(rank):
                self.mem_device_handles.append(self.memory_channels[rank].device_handle().raw)
            if rank != self.group.my_rank and not in_same_node(rank):
                self.proxy_device_handles.append(self.port_channels[rank].device_handle().raw)

        self.mem_device_handles_cp = cp.asarray(memoryview(b"".join(self.mem_device_handles)), dtype=cp.uint8)
        self.proxy_device_handles_cp = cp.asarray(memoryview(b"".join(self.proxy_device_handles)), dtype=cp.uint8)

        self.set_params(nblocks, block_size)

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory_out

    def set_params(self, nblocks, block_size):
        self.nblocks = nblocks
        self.block_size = block_size

        self.params = b""
        self.params += pack(
            self.mem_device_handles_cp,
            self.proxy_device_handles_cp,
            self.memory,
            self.scratch,
            self.put_buff,
            self.memory_out,
            self.group.my_rank,
            self.nranks_per_node,
            self.group.nranks,
            bytes(4),  # padding for memory alignment
            ctypes.c_size_t(self.memory.size),
        )

    def auto_tune(self):
        nblocks_to_try = [21, 42, 84]
        block_size_to_try = [256, 512, 1024]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                self.set_params(nblocks, block_size)
                yield nblocks, block_size


class MscclppAllReduce6:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        nelem: int,
        memory_dtype: cp.dtype,
        block_size: int = 1024,
        nblocks: int = 32,
    ):
        self.group = group
        datatype_size = memory_dtype().itemsize
        buffer_size = nelem * datatype_size
        type_str = type_to_str(memory_dtype)
        all_ranks = list(range(group.nranks))
        remote_nghrs = all_ranks.copy()
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.nvlink_connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        self.nvls_connection = group.make_connection(all_ranks, Transport.Nvls)
        self.memory = GpuBuffer(nelem, memory_dtype)
        self.nvls_mem_handle = self.nvls_connection.bind_allocated_memory(
            self.memory.data.ptr, self.memory.data.mem.size
        )

        # create a memory_channel for each remote neighbor
        self.semaphores = group.make_semaphore(self.nvlink_connections, MemoryDevice2DeviceSemaphore)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel = KernelBuilder(
            file="allreduce.cu",
            kernel_name="allreduce6",
            file_dir=file_dir,
            macro_dict={"TYPE": type_str},
        ).get_compiled_kernel()
        self.device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank:
                self.device_handles.append(self.semaphores[rank].device_handle().raw)

        self.device_handles_cp = cp.asarray(memoryview(b"".join(self.device_handles)), dtype=cp.uint8)
        self.nvls_handle = self.nvls_mem_handle.device_handle().raw

        if self.memory.dtype != cp.float16 and self.memory.dtype != cp.float32:
            raise RuntimeError("Unsupported data type")

        if self.memory.dtype == cp.float16:
            vector_size = 8
        elif self.memory.dtype == cp.float32:
            vector_size = 4
        else:
            vector_size = 1
        self.set_params(nblocks, block_size, vector_size)

    def get_memory(self):
        return self.memory

    def __call__(self, stream_ptr):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream_ptr)
        return self.memory

    def set_params(self, nblocks, block_size, vector_size):
        self.nblocks = nblocks
        self.block_size = block_size
        self.vector_size = vector_size
        self.params = b""
        self.params += pack(
            self.device_handles_cp,
            self.nvls_handle,
            self.group.my_rank,
            self.group.nranks,
            ctypes.c_size_t(self.memory.size),
            self.vector_size,
        )

    def auto_tune(self):
        nblocks_to_try = [8, 12, 16, 24, 32, 48, 64, 72, 96, 108]
        block_size_to_try = [256, 512, 1024]
        if self.memory.dtype == cp.float16:
            vector_size_to_try = [8, 4, 2]
        elif self.memory.dtype == cp.float32:
            vector_size_to_try = [4, 2, 1]
        else:
            vector_size_to_try = [1]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                for vector_size in vector_size_to_try:
                    self.set_params(nblocks, block_size, vector_size)
                    yield nblocks, block_size, vector_size
