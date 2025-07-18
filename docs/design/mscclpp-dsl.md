# MSCCL++ DSL
## Introduction

The MSCCL++ Domain-Specific Language (DSL) provides a Python-native API for defining and executing GPU-based communication collective. With a few high-level calls, users can construct complex data movement and synchronization workflows without dealing with low-level CUDA code.

Here is the highlights of the MSCCL++ DSL:
- **Fine-grained Python-native API**: MSCCL++ DSL provides a Pythonic, fine-grained API for defining and executing GPU-based communication collectives. Users can construct complex data movement and synchronization workflows without writing low-level CUDA code, while still achieving performance comparable to hand-tuned CUDA implementations.

- **Effortless performance tuning**:  The MSCCL++ DSL analyzes data dependencies and synchronization patterns to automatically fuse operations to eliminate data movement overhead, while instance counts can be manually configured to boost performance.

- **Flexible execution model**: The MSCCL++ DSL allows users to load different execution plans at runtime, enabling dynamic optimization based on the current workload and hardware configuration.


## MSCCL++ DSL Concepts

### Buffer/Chunk
Buffer is a data structure that holds the data to be sent or received. The input/output buffer is predefined based on communication patterns. User can allocate scratch buffer for intermediate data movement. Chunk is a slice of the buffer.

```python
rank = Rank(rank_id)
input_buffer = rank.get_input_buffer()
dst_chunk = input_buffer[0:1]
src_chunk = input_buffer[1:2]
rank.copy(dst_chunk, src_chunk, tb=0)
rank.reduce(dst_chunk, src_chunk, op=ReduceOperationType.sum, tb=0)
```

### Channel
User need to use channel to communicate between ranks. Now we have three types of channels: memoryChannel, portChannel and switchChannel.
- **MemoryChannel**: Uses peer-to-peer memory access to communicate between GPUs.
- **PortChannel**: Uses interconnection ports to communicate between GPUs.
- **SwitchChannel**: Uses interconnection-switch-enabled multimem memory access to communicate between GPUs.

**Note:** Each time call channel will created a new one. If user want to reuse the channel, please keep the channel object.

Here is the example for two ranks synchronization with each others.
```python
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    channel = MemoryChannel(dst_rank, src_rank)
    channel.signal(tb=0, data_sync=SyncType.none)
    channel.wait(tb=0, data_sync=SyncType.after)
```

#### For switch channel
Switch channel associates a group of buffers from a specified set of ranks. All operations invoked on the channel will be applied to those buffers.

Example for two ranks allreduce via switch channel.
```python
# Creating Channels
switch_chan = SwitchChannel(rank_list=[gpu for gpu in range(gpu_size)], buffer_type=BufferType.input)
for gpu in range(gpu_size):
    buffer_offset = gpu
    rank = Rank(gpu)
    input_buffer = rank.get_input_buffer()
    switch_chan.at_rank(gpu).group_load_reduce(buffer_offset, size=1, dst_chunk=input_buffer[gpu : gpu + 1], tb=0)
    switch_chan.at_rank(gpu).group_store(input_buffer[gpu : gpu + 1], buffer_offset, size=1, tb=0)
```

### Synchronization
We provide some synchronization primitives to sync threadblocks inside a rank. The synchronization is done through a barrier or semaphore. The barrier is used to synchronize a set of thread blocks in the rank, while the semaphore allows asynchronous signaling and waiting between thread blocks.

```python
rank = Rank(0)
rank.barrier([0, 1])
sem = Semaphore(rank=0, initial_value=1)
sem.acquire(tb=0, data_sync=SyncType.after)
sem.release(tb=0, data_sync=SyncType.before)
```

The synchronization inside the thread-block can be inferred by MSCCL++ DSL automatically. Which mean if we have data dependence between two operations, we will insert a synchronization point between them. 

But for multi-thread-blocks synchronization and cross ranks synchronization, we need to insert the synchronization point manually.


## Kernel fusion
MSCCL++ DSL performs kernel fusion by analyzing all operations scheduled within the same thread‐block. For each thread‐block, the DSL builds a directed acyclic graph (DAG) of chunk‐level operations and tracks data dependencies and usage patterns. When two or more operations meet fusion criteria—such as contiguous chunk access, no intervening dependencies, and compatible resource requirements—the DSL merges them into a single GPU kernel function. This fusion strategy reduces launch overhead and memory traffic, resulting in more efficient execution.  


## Pipeline Loop
Pipeline enables overlapping operations across thread blocks. Using Semaphore for cross-block synchronization, it overlaps stages—such as copying data from the input buffer to a scratch buffer—with subsequent peer transfers. A pipelined loop orchestrates these stages to run concurrently, maximizing overall throughput.

This example demonstrates a pipelined loop that copies data from an input buffer to a scratch buffer, then transfers it to other peers. The `Semaphore` is used to synchronize the two stages. `unit` specifies the size of each chunk, and `num_chunks` indicates how many chunks will be processed in the loop.

```python
sem = Semaphore(rank=rank, initial_value=1)
rank = Rank(rank)
channel = MemoryChannel(dst_rank, src_rank)
with LoopIterationContext(unit=2**20, num_chunks=1):
    # The dst_chunk and src_chunk sizes should match the num_chunks parameter in the loop context.
    rank.copy(dst_chunk, src_chunk, tb=0)
    sem.release(tb=0)
    sem.acquire(tb=1)
    channel.put(other_peer_chunk, dst_chunk, tb=1)
``` 


Here is the example for two ranks allreduce. Which achieve non-zero copy and use nvls. We use 3 thread-blocks to do the allreduce.
The first thread-block is used to copy data from input buffer to scratch buffer, the second thread-block is used to do allreduce in scratch buffer, and the third thread-block is used to copy data from scratch buffer to output buffer.  The thread-blocks are synchronized by semaphores.
```python
nvls_chan = SwitchChannel(rank_list=[0, 1], buffer=Buffer.scratch)
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    chan = MemoryChannel(dst_rank, src_rank)
    chan1 = MemoryChannel(dst_rank, src_rank)
    rank = Rank(i)
    sem0 = Semaphore(rank=i, initial_value=1)
    sem1 = Semaphore(rank=i, initial_value=1)
    input_buffer = rank.get_input_buffer()
    output_buffer = rank.get_output_buffer()
    scratch_buffer = Buffer(i, scratch_buffer_size)
    with Loop.iteration(unit=2**20, num_chunks=1):
        # copy data to scratch buffer
        for offset in range(nranks):
            dst_chunk = scratch_buffer[offset:offset+1]
            src_chunk = input_buffer[offset:offset+1]
            rank.copy(dst_chunk, src_chunk, tb=0)
        chan.signal(tb=0, SyncType.before)
        chan.wait(tb=0, SyncType.after)
        sem0.release(tb=0)

        # do allreduce in scratch buffer
        sem0.acquire(tb=1, SyncType.after)
        nvls_chan.group_load_reduce(buffer_offset=i, size=1, dst_chunk=scratch_buffer[i:i+1], tb=1)
        nvls_chan.group_store(src_chunk=input_buffer[i:i+1], buffer_offset=i, size=1, tb=1)
        chan1.signal(tb=1, SyncType.before)
        sem1.release(tb=1)

        # copy data back to output buffer
        sem1.acquire(tb=2)
        chan1.wait(tb=2, SyncType.after)
        for index in range(nranks):
            dst_chunk = output_buffer[index:index+1]
            src_chunk = scratch_buffer[index:index+1]
            rank.copy(dst_chunk, src_chunk, tb=2)
```

## Generate execution plan
The MSCCL++ DSL generates an execution plan in JSON format, which describes the operations to be executed on each rank. The execution plan includes information about the buffers, channels, and synchronization points. This plan is then used by the MSCCL++ runtime to execute the operations on the GPU.

For the details of the execution plan, please refer to the [MSCCL++ Execution Plan](./mscclpp-execution-plan.md).

## All2All support
For now, DSL only support static all2all algorithm. For all2allv support, we need to get the send/recv size at the runtime. It may require some placeholder at the Json execution plan and relace to the real size at the runtime. If we could make chunk size be variable, we could use the same way to support all2allv.
