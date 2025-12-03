# MSCCL++ Torch Integration Guide

This guide demonstrates how to integrate MSCCL++ with PyTorch for customized collective communication operations.

## Overview

MSCCL++ provides three ways to implement custom collective algorithms:

1. **Native C++/CUDA Kernel Algorithms**: Define custom CUDA kernels using MSCCL++ C++ API
2. **DSL-based Algorithms**: Use MSCCL++ Python DSL to define collective operations
3. **Default Built-in Algorithms**: Leverage pre-built algorithms from MSCCL++

All approaches work with PyTorch through the MSCCL++ communicator interface.

```{figure} ../figs/customize_algo.png
:name: MSCCL++ Customization Algorithm Selection
:alt: MSCCL++ Customization Algorithm Selection
:align: center
:width: 400px

MSCCL++ Customization Algorithm Selection Overview
```

## Algorithm API

The `Algorithm` class provides a unified interface for executing collective operations:

```python
# Full Algorithm execution interface
algorithm.execute(
    comm=communicator,                # MSCCL++ communicator (required)
    input_buffer=ptr,                 # Input buffer pointer (required)
    output_buffer=ptr,                # Output buffer pointer (required)
    input_size=nbytes,                # Input size in bytes (required)
    output_size=nbytes,               # Output size in bytes (required)
    dtype=mscclpp.DataType,           # Data type (required)
    op=mscclpp.ReduceOp.NOP,          # Reduction operation (default: NOP)
    stream=0,                         # CUDA stream handle (default: 0)
    executor=None,                    # MSCCL++ Executor for DSL algorithms (default: None)
    nblocks=0,                        # Number of thread blocks (default: 0, auto-select)
    nthreads_per_block=0,             # Number of threads per block (default: 0, auto-select)
    extras=None                       # Additional parameters as dict (default: None)
)
```

**Parameters:**

- **comm**: The MSCCL++ communicator object
- **input_buffer**: Pointer to input buffer (use `tensor.data_ptr()` for PyTorch tensors)
- **output_buffer**: Pointer to output buffer  
- **input_size**: Size of input data in bytes (use `tensor.nbytes`)
- **output_size**: Size of output data in bytes
- **dtype**: Data type (e.g., `mscclpp.DataType.float16`, `mscclpp.DataType.float32`)
- **op**: Reduction operation for reduce collectives
- **stream**: CUDA stream handle (use `torch.cuda.current_stream().cuda_stream`)
- **executor**: Required for DSL-based algorithms (`mscclpp.Executor` instance)
- **nblocks**: Number of CUDA thread blocks to launch (0 = auto-select by algorithm)
- **nthreads_per_block**: Threads per block (0 = auto-select by algorithm)
- **extras**: Dictionary of additional algorithm-specific parameters. Values must be pointer addresses (int/uintptr_t), e.g., `{"buffer_ptr": tensor.data_ptr()}`

**Example Usage:**

```python
# Native algorithm (kernel-based)
algorithm.execute(
    comm=comm_group.communicator,
    input_buffer=input_tensor.data_ptr(),
    output_buffer=output_tensor.data_ptr(),
    input_size=input_tensor.nbytes,
    output_size=output_tensor.nbytes,
    dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(input_tensor.dtype),
    stream=torch.cuda.current_stream().cuda_stream
)

# DSL-based algorithm (requires executor)
algorithm.execute(
    comm=comm_group.communicator,
    executor=executor,
    input_buffer=tensor.data_ptr(),
    output_buffer=tensor.data_ptr(),
    input_size=tensor.nbytes,
    output_size=tensor.nbytes,
    dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
    stream=torch.cuda.current_stream().cuda_stream
)

# With custom launch configuration
algorithm.execute(
    comm=comm_group.communicator,
    input_buffer=tensor.data_ptr(),
    output_buffer=tensor.data_ptr(),
    input_size=tensor.nbytes,
    output_size=tensor.nbytes,
    dtype=mscclpp.DataType.float32,
    op=mscclpp.ReduceOp.SUM,
    stream=torch.cuda.current_stream().cuda_stream,
    nblocks=64,
    nthreads_per_block=512,
    extras={"scratch_buffer": scratch_buffer.data_ptr()}  # Pass pointer addresses
)
```

## Implementation Approaches

### 1. Native C++/CUDA Kernel Algorithm

Define custom CUDA kernels using MSCCL++ C++ API. See `examples/torch-integration/customized_allgather.py` and `customized_allgather.cu`.

**Step 1: Implement AlgorithmBuilder in C++**

```cpp
// In your .cu file
class AllgatherAlgoBuilder : public mscclpp::AlgorithmBuilder {
 public:
  std::shared_ptr<mscclpp::Algorithm> build() override {
    auto self = std::make_shared<AllgatherAlgoBuilder>();
    std::shared_ptr<mscclpp::Algorithm> algo = std::make_shared<mscclpp::NativeAlgorithm>(
        "allgather",                    // Algorithm name
        "allgather",                    // Collective name
        [self](std::shared_ptr<mscclpp::Communicator> comm) { 
            self->initialize(comm); 
        },
        [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, 
               void* output, size_t inputSize, size_t outputSize, 
               mscclpp::DataType dtype, mscclpp::ReduceOp op, 
               cudaStream_t stream, int nBlocks, int nThreadsPerBlock, 
               const std::unordered_map<std::string, uintptr_t>& extras) {
            return self->kernelFunc(ctx, input, output, inputSize, dtype, stream);
        },
        [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, 
               void* output, size_t inputSize, size_t outputSize, 
               mscclpp::DataType dtype) { 
            return self->initContext(comm, input, output, inputSize, dtype); 
        },
        [self](const void* input, void* output, size_t inputSize, 
               size_t outputSize, mscclpp::DataType dtype) {
            return self->generateContextKey(input, output, inputSize, outputSize, dtype);
        }
    );
    return algo;
  }
  
 private:
  void initialize(std::shared_ptr<mscclpp::Communicator> comm) { /* ... */ }
  mscclpp::CommResult kernelFunc(/* ... */) { /* ... */ }
  std::shared_ptr<mscclpp::AlgorithmCtx> initContext(/* ... */) { /* ... */ }
  std::string generateContextKey(/* ... */) { /* ... */ }
};

// Expose to Python using pybind11
PYBIND11_MODULE(mscclpp_native, m) {
    m.def("create_allgather_algorithm", []() -> py::capsule {
        AllgatherAlgoBuilder builder;
        auto algo = builder.build();
        return py::capsule(new std::shared_ptr<mscclpp::Algorithm>(algo), 
                          "mscclpp::Algorithm");
    });
}
```

**Step 2: Compile and load in Python**

```python
import mscclpp
import os

# Compile the native algorithm
mscclpp_native = mscclpp.compile_native(
    name="mscclpp_native", 
    file=os.path.join(path, "customized_allgather.cu")
)

# Create algorithm from capsule
capsule = mscclpp_native.create_allgather_algorithm()
algorithm = mscclpp.Algorithm.create_from_native_capsule(capsule)
```

**Step 3: Execute in your application**

```python
import torch
import mscclpp.comm as mscclpp_comm
import mscclpp.utils as mscclpp_utils

# Initialize communicator
comm_group = mscclpp_comm.CommGroup(
    interfaceIpPortTrio=f"{interface}:{master_addr}:{master_port}",
    rank=rank, 
    size=world_size
)

# Prepare tensors
tensor = torch.randn(local_size, device="cuda", dtype=torch.float32)
output = torch.randn(local_size * world_size, device="cuda", dtype=torch.float32)

# Execute algorithm
algorithm.execute(
    comm_group.communicator,
    tensor.data_ptr(),
    output.data_ptr(),
    tensor.nbytes,
    output.nbytes,
    mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
    stream=torch.cuda.current_stream().cuda_stream
)
```

### 2. DSL-based Algorithm

Use MSCCL++ Python DSL to define collective operations. See `examples/torch-integration/customized_comm_with_dsl.py`.

**Step 1: Define the collective program**

```python
from mscclpp.language.collectives import AllReduce
from mscclpp.language.channel import SwitchChannel, MemoryChannel, BufferType
from mscclpp.language.program import CollectiveProgram

def allreduce_nvls(spec: mscclpp.AlgoSpec) -> CollectiveProgram:
    gpu_size = spec.world_size
    with CollectiveProgram(
        spec.name,
        spec.collective,
        gpu_size,
        instances=spec.instances,
        protocol=spec.protocol,
        num_threads_per_block=spec.num_threads_per_block,
        min_message_size=spec.min_message_size,
        max_message_size=spec.max_message_size,
    ) as program:
        # Create channels
        nvls_chan = SwitchChannel(
            rank_list=[gpu for gpu in range(gpu_size)], 
            buffer_type=BufferType.input
        )
        
        # Define communication pattern
        for gpu in range(gpu_size):
            rank = Rank(gpu)
            input_buffer = rank.get_input_buffer()
            nvls_chan.at_rank(gpu).reduce(
                buffer_offset=gpu, 
                size=1, 
                dst_chunk=input_buffer[gpu:gpu+1], 
                tb=0
            )
            nvls_chan.at_rank(gpu).broadcast(
                src_chunk=input_buffer[gpu:gpu+1], 
                buffer_offset=gpu, 
                size=1, 
                tb=0
            )
    return program
```

**Step 2: Compile the algorithm**

```python
# Define algorithm specification
spec = mscclpp.AlgoSpec(
    name="allreduce_nvls",
    collective=AllReduce(world_size, 1, True),
    nranks_per_node=nranks_per_node,
    world_size=world_size,
    in_place=True,
    instances=nranks_per_node,
    protocol="Simple",
    num_threads_per_block=1024,
    min_message_size=1 << 20,
    max_message_size=48 << 30,
    tags={"nvls": 1}
)

# Compile the algorithm
algorithm = mscclpp.compile(algo=allreduce_nvls, algo_spec=spec, rank=rank)
```

**Step 3: Execute in your application**

```python
class CustomizedComm:
    def __init__(self, comm: mscclpp_comm.CommGroup, algorithms=[]):
        self.comm = comm
        self.executor = mscclpp.Executor(comm.communicator)
        self.algorithms = algorithms

    def all_reduce(self, tensor: torch.Tensor, stream: torch.cuda.Stream = None):
        algo = self.algorithms[0]
        algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            stream=stream.cuda_stream if stream is not None else 0
        )

# Usage
comm = CustomizedComm(comm_group, algorithms=[algorithm])
tensor = torch.randn(size, dtype=torch.bfloat16, device="cuda")
comm.all_reduce(tensor, stream=torch.cuda.current_stream())
```

### 3. Default Built-in Algorithms

Use pre-built algorithms from MSCCL++. See `examples/torch-integration/customized_comm_with_default_algo.py`.

**Step 1: Load default algorithms**

```python
def load_algorithms(scratch_buffer: torch.Tensor, rank: int):
    collection_builder = mscclpp.AlgorithmCollectionBuilder()
    return collection_builder.build_default_algorithms(
        scratch_buffer=scratch_buffer.data_ptr(),
        scratch_buffer_size=scratch_buffer.nbytes,
        rank=rank
    )
```

**Step 2: Select and execute algorithms**

```python
class CustomizedComm:
    def __init__(self, comm: mscclpp_comm.CommGroup):
        self.comm = comm
        # Allocate scratch buffer
        dlpack = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        
        # Load default algorithms
        algorithms = load_algorithms(self.scratch_buffer, comm.my_rank)
        
        # Select specific algorithms by name
        self._algo_nvls_packet = [
            algo for algo in algorithms 
            if algo.collective == "allreduce" 
            and algo.name == "default_allreduce_nvls_packet"
        ][0]
        
        self._algo_nvls_copy = [
            algo for algo in algorithms 
            if algo.collective == "allreduce" 
            and algo.name == "default_allreduce_nvls_with_copy"
        ][0]
    
    def all_reduce(self, tensor: torch.Tensor, op, stream=None):
        # Select algorithm based on message size
        if tensor.nbytes < 1 << 20:
            algo = self._algo_nvls_packet
        else:
            algo = self._algo_nvls_copy
        
        algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=mscclpp.ReduceOp.SUM,
            stream=stream.cuda_stream if stream is not None else 0
        )
```

## Complete Example

Here's a complete example showing initialization and execution:

```python
import os
import torch
import mscclpp.comm as mscclpp_comm
import mscclpp.utils as mscclpp_utils
import mscclpp

def init_communicator():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = get_network_interface(master_addr)
    
    interface_trio = f"{interface}:{master_addr}:{master_port}"
    comm_group = mscclpp_comm.CommGroup(
        interfaceIpPortTrio=interface_trio,
        rank=rank,
        size=world_size
    )
    
    return comm_group

def main():
    # Initialize communicator
    comm_group = init_communicator()
    
    # Load or compile your algorithm
    # Option 1: Native algorithm
    mscclpp_native = mscclpp.compile_native(
        name="my_algo", 
        file="customized_allgather.cu"
    )
    capsule = mscclpp_native.create_allgather_algorithm()
    algorithm = mscclpp.Algorithm.create_from_native_capsule(capsule)
    
    # Option 2: DSL algorithm
    # algorithm = mscclpp.compile(algo=allreduce_nvls, algo_spec=spec, rank=rank)
    
    # Option 3: Default algorithm
    # algorithms = load_default_algorithms(...)
    # algorithm = select_algorithm(algorithms, message_size)
    
    # Prepare tensors
    local_size = 1 << 20
    tensor = torch.randn(local_size, device="cuda", dtype=torch.float32)
    output = torch.randn(local_size * comm_group.nranks, device="cuda")
    
    # Execute collective operation
    algorithm.execute(
        comm_group.communicator,
        tensor.data_ptr(),
        output.data_ptr(),
        tensor.nbytes,
        output.nbytes,
        mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
        stream=torch.cuda.current_stream().cuda_stream
    )
    
    torch.cuda.synchronize()
    print(f"Rank {comm_group.my_rank}: operation completed")

if __name__ == "__main__":
    main()
```

## Running the Examples

To run the torch-integration examples:

```bash
# Set environment variables
export MSCCLPP_MASTER_ADDR=<master_node_ip>
export MSCCLPP_MASTER_PORT=<port>

# Run with torchrun
# Example 1: Native kernel-based allgather
torchrun --nnodes=1 --nproc_per_node=8 examples/torch-integration/customized_allgather.py

# Example 2: DSL-based allreduce
torchrun --nnodes=1 --nproc_per_node=8 examples/torch-integration/customized_comm_with_dsl.py

# Example 3: Default algorithms
torchrun --nnodes=1 --nproc_per_node=8 examples/torch-integration/customized_comm_with_default_algo.py
```

## See Also

- Example code: `examples/torch-integration/`
- API reference: {doc}`../py_api`
- DSL guide: Check DSL documentation for defining custom collectives