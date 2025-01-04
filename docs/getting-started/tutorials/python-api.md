# Working with Python API

We provide Python API which help to initialze and setup the channel easily.
In this tutorial, you will write a simple program to initialize communication between eight GPUs using MSCCL++ Python API.

## Setup Channel with Python API

We will setup a mesh topology with eight GPUs. Each GPU will be connected to its neighbors. The following code shows how to initialize communication with MSCCL++ Python API.
```python
from mpi4py import MPI
import cupy as cp

from mscclpp import (
    ProxyService,
    Transport,
)
from mscclpp.utils import GpuBuffer
import mscclpp.comm as mscclpp_comm

def create_connection(group: mscclpp_comm.CommGroup, transport: str):
    remote_nghrs = list(range(group.nranks))
    remote_nghrs.remove(group.my_rank)
    if transport == "NVLink":
        tran = Transport.CudaIpc
    elif transport == "IB":
        tran = group.my_ib_device(group.my_rank % 8)
    else:
        assert False
    connections = group.make_connection(remote_nghrs, tran)
    return connections

if __name__ == "__main__":
    mscclpp_group = mscclpp_comm.CommGroup(MPI.COMM_WORLD)
    connections = create_connection(mscclpp_group, "NVLink")
    nelems = 1024
    memory = GpuBuffer(nelem, dtype=cp.int32)
    proxy_service = ProxyService()
    simple_channels = group.make_proxy_channels(proxy_service, memory, connections)
    proxy_service.start_proxy()
    mscclpp_group.barrier()
    launch_kernel(mscclpp_group.my_rank, mscclpp_group.nranks, simple_channels, memory)
    cp.cuda.runtime.deviceSynchronize()
    mscclpp_group.barrier()
```

### Launch Kernel with Python API
We provide some Python utils to help you launch kernel via python. Here is a exampl.
```python
from mscclpp.utils import KernelBuilder, pack

def launch_kernel(my_rank: int, nranks: int, simple_channels: List[ProxyChannel], memory: cp.ndarray):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(file="test.cu", kernel_name="test", file_dir=file_dir).get_compiled_kernel()
    params = b""
    first_arg = next(iter(simple_channels.values()))
    size_of_channels = len(first_arg.device_handle().raw)
    device_handles = []
    for rank in range(nranks):
        if rank == my_rank:
            device_handles.append(
                bytes(size_of_channels)
            )  # just zeros for semaphores that do not exist
        else:
            device_handles.append(simple_channels[rank].device_handle().raw)
    # keep a reference to the device handles so that they don't get garbage collected
    d_channels = cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)
    params = pack(d_channels, my_rank, nranks, memory.size)

    nblocks = 1
    nthreads = 512
    kernel.launch_kernel(params, nblocks, nthreads, 0, None)
```

The test kernel is defined in `test.cu` as follows:
```cuda
#include <mscclpp/packet_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    proxy_channel(mscclpp::ProxyChannelDeviceHandle* channels, int my_rank, int nranks,
                         int num_elements) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    uint64_t size_per_rank = (num_elements * sizeof(int)) / nranks;
    uint64_t my_offset = size_per_rank * my_rank;
    __syncthreads();
    if (tid < nranks && tid != my_rank) {
      channels[tid].putWithSignalAndFlush(my_offset, my_offset, size_per_rank);
      channels[tid].wait();
    }
}
```
