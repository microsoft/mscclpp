# Working with Python API

We provide Python API which help to initialze and setup the channel easily.
In this tutorial, you will write a simple program to initialize communication between eight GPUs using MSCCL++ Python API.

Internal backend checks use `mscclpp._mscclpp.is_hip`, a boolean set when the native
extension is built: `True` for HIP/ROCm and `False` for CUDA. Reading this flag does
not initialize a GPU or require a visible device.

## Setup Channel with Python API

We will setup a mesh topology with eight GPUs. Each GPU will be connected to its neighbors. The following code shows how to initialize communication with MSCCL++ Python API.
```python
from mpi4py import MPI
import numpy as np

from mscclpp import (
    ProxyService,
    Transport,
    CommGroup,
    GpuBuffer,
)
from mscclpp_benchmark.gpu import device_synchronize, set_device


def create_connection(group: CommGroup, transport: str):
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
    mscclpp_group = CommGroup(MPI.COMM_WORLD)
    set_device(mscclpp_group.my_rank % mscclpp_group.nranks_per_node)
    connections = create_connection(mscclpp_group, "NVLink")
    nelems = 1024
    memory = GpuBuffer(nelems, dtype=np.int32)
    proxy_service = ProxyService()
    simple_channels = mscclpp_group.make_port_channels(proxy_service, memory, connections)
    proxy_service.start_proxy()
    mscclpp_group.barrier()
    kernel, d_channels = launch_kernel(mscclpp_group.my_rank, mscclpp_group.nranks, simple_channels, memory)
    device_synchronize()
    mscclpp_group.barrier()
    proxy_service.stop_proxy()
```

### Launch Kernel with Python API
We provide some Python utils to help you launch kernel via python. Here is a exampl.
```python
from mscclpp.utils import KernelBuilder, pack
from mscclpp import GpuBuffer, PortChannel
import os

def launch_kernel(my_rank: int, nranks: int, simple_channels: dict[int, PortChannel], memory: GpuBuffer):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(file="test.cu", kernel_name="port_channel", file_dir=file_dir).get_compiled_kernel()
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
    d_channels = GpuBuffer.from_numpy(memoryview(b"".join(device_handles)), dtype=np.uint8)
    params = pack(d_channels, my_rank, nranks, memory.size)

    nblocks = 1
    nthreads = 512
    kernel.launch_kernel(params, nblocks, nthreads, 0, None)
    return kernel, d_channels  # Keep both alive until device_synchronize() completes.
```

The final argument to `Kernel.launch_kernel()` accepts an integer CUDA/HIP stream
pointer, a PyTorch stream, or `None` for the default stream. CuPy stream objects
must be passed as their integer `.ptr` value instead.

Kernel loading and launching use `cuda-bindings` on CUDA and `hip-python` on ROCm,
selected by `mscclpp._mscclpp.is_hip`. CUDA initializes the current device's runtime
context before loading a module. Failed driver calls raise `RuntimeError` with the
API name and error code. Neither kernel launching nor `GpuBuffer` requires CuPy.

`pack()` extracts pointers from NumPy arrays, PyTorch tensors, and GPU arrays that
expose an integer `array.data.ptr`, including `GpuBuffer`. The packing implementation
does not copy array contents. Keep the buffers alive until the GPU has finished using
the packed pointers. A NumPy pointer refers to host memory; use `GpuBuffer.from_numpy()`
to upload data before passing it to a kernel that expects device memory.

`CommGroup` memory registration uses the same pointer extraction. NumPy and
GPU arrays must also expose `size` and `itemsize` for calculating the byte count;
PyTorch tensors use `numel()` and `element_size()`. The communication wrapper's
implementation does not import CuPy.

### Native GPU buffers

`GpuBuffer` owns a native MSCCL++ allocation rather than a CuPy array. It exposes
`shape`, `dtype`, `size`, `itemsize`, `nbytes`, `strides`, `device_id`, and `data.ptr`.
`allocation_size` is the actual allocation extent (which may exceed `nbytes` for
NVLS alignment); bind an unsliced buffer's pointer and allocation size to NVLS.

```python
host = np.arange(1024, dtype=np.float32)
buffer = GpuBuffer.from_numpy(host)
buffer.copy_from_numpy(host * 2)
buffer[256:512].fill(0)
result = buffer.to_numpy()
```

Transfers and `fill()` synchronize the allocation's device and restore the previous
current device. Run them outside graph capture and timed benchmark operations.
Use NumPy on downloaded arrays for arithmetic, comparisons, or advanced indexing.
Buffers support C- or F-contiguous layouts with native-endian numeric/boolean dtypes;
explicit strides must match that layout. Dimensions may be zero. One-dimensional
unit-step slices alias GPU storage and retain the native allocation even after the
parent wrapper is released. Multidimensional slicing, strided views, implicit array
conversion, element assignment, and array arithmetic are not supported.

The test kernel is defined in `test.cu` as follows:
```cuda
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel_device.hpp>

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    port_channel(mscclpp::PortChannelDeviceHandle* channels, int my_rank, int nranks,
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
