# MSCCL++ DSL Concepts
### Chunk/Tensor
Chunk is a data structure that holds the data to be sent or received. For some local operations, such as copy/reduce we provide some functions to manipulate the chunk.

```python
rank = Rank(rank_id)
input_buffer = rank.get_input_buffer()
dst_chunk = input_buffer[0:1]
scr_chunk = intput_buffer[1:2]
rank.copy(dst_chunk, src_chunk, tb=0)
rank.reduce(dst_chunk, src_chunk, op="sum", tb=0)
```

### Channel
All cross ranks related op is done through channels. A channel is a communication medium between ranks. It can be a network socket, shared memory, or any other form of IPC. The channel is responsible for sending and receiving messages between ranks.

**Note:** Each time call channel will created a new one. If user want to reuse the channel, please keep the channel object.


Examples:
```python
channel = Channel(dst_rank, src_rank, channel_type)
channel.put(dst_chunk, src_chunk, tb=0)
channel.signal(tb=0, sync="before")
channel.wait(tb=0, sync="after")
```

Here is the example for two ranks synchonization with each others.
```python
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    channel = Channel(dst_rank, src_rank, channel_type=Channel.memory)
    channel.relaxedSignal(tb=0, sync=None)
    channel.relaxedWait(tb=0, sync="after")
```

#### For nvls based channel
NVLS channel need to bind with a group of buffers from a set of ranks. Each operation on the channel will be performed on the buffers in the group.

```python
nvls_chan = SwitchChannel(rank_list=[], buffer=Buffer.input)
nvls_chan.group_load_reduce(offset1, size1, op="sum", tb=0)
nvls_chan.group_store(offset, size, tb=0)
```

Example for two ranks allreduce.
```python
# allreduce for 2 ranks.
# 1. copy data from input buffer to scratch buffer
# 2. allreduce the data in scratch buffer
# 3. copy data from scratch buffer to output buffer
nvls_chan = SwitchChannel(rank_list=[0, 1], buffer=Buffer.scratch)
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    chan = Channel(dst_rank, src_rank, channel_type=Channel.memory)
    rank = Rank(i)
    input_buffer = rank.get_input_buffer()
    output_buffer = rank.get_output_buffer()
    scratch_buffer = rank.Buffer(nranks)
    for offset in range(nranks):
        # copy data to scratch buffer
        dst_chunk = scratch_buffer[offset:offset+1]
        src_chunk = input_buffer[offset:offset+1]
        rank.copy(dst_chunk, src_chunk, tb=0)
    chan.signal(tb=0, sync="before")
    chan.wait(tb=0, sync="after")

    # do allreduce in scratch buffer
    buffer_offset = src_rank
    nvls_chan.group_load_reduce(buffer_offset, 1, op="sum", tb=0)
    nvls_chan.group_store(buffer_offset, 1, tb=0)

    # copy data back to output buffer
    chan.signal(tb=0, sync="before")
    chan.wait(tb=0, sync="after")
    for offset in range(nranks):
        dst_buffer = output_buffer[offset:offset+1]
        src_buffer = scratch_buffer[offset:offset+1]
        rank.copy(dst_buffer, src_buffer, tb=0)
```

### Synchronization
We provide some synchronization primitives to sync threadblocks inside a rank. The synchronization is done through a barrier or semaphore. The barrier is used to synchronize a set of thread blocks in the rank, while the semaphore allows asynchronous signaling and waiting between thread blocks.

```python
rank.barrier(tb_list=[])
sem = Rank.Semaphore(rank=0, size=1)
sem.acquire(tb=0, sync="after")
sem.release(tb=0, sync="before")
```

The synchronization inside the thread-block can be inferred by MSCCL++ DSL automatically. Which mean if we have data dependence between two operations, we will insert a synchronization point between them. 

But for multi-thread-blocks synchronization and cross ranks synchronization, we need to insert the synchronization point manually.

We could use atomic operation to implement the semaphore machanism.

## For kernel fusion
We only fuse the kernel that in the same thread-block. We still need to construct the DAG for each thread-block. Track the chunk usage and see if we can fuse the kernel.


## For Pipeline Loop
For some cases, we need to pipeline the kernel to overlap some operations. For example, the first stage is copy data from input buffer to scratch buffer, the second stage is transfer data from scratch buffer to other peers. We could use `Rank.semphore` to synchronize the two stages. 
```python
sem = Rank.Semaphore(rank=0, size=1)
rank = Rank(0)
rank.copy(dst_chunk, src_chunk, tb=0)
sem.release(tb=0)
channel = Channel(dst_rank, src_rank, channel_type)
sem.acquire(tb=1)
channel.put(dst_chunk, src_chunk, tb=1)
```

Also we could provide some gramar sugar to make the pipeline more readable. For example, we could use `Loop` to construct the pipeline. 
```python
sem = Rank.Semaphore(rank=rank, size=1)
rank = Rank(src_rank)
with Loop.iteration(unit=2**20, num_chunks=1) as iter:
    # the dst_chunk and src_chunk size but same as loop context
    rank.copy(dst_chunk, src_chunk, tb=0, iter_context=iter)
    sem.release(tb=0)
    channel = Channel(dst_rank, src_rank, channel_type)
    sem.acquire(tb=1)
    channel.put(dst_chunk, src_chunk, tb=1, iter_context=iter)
``` 


Here is the example for two ranks allreduce. Which achieve non-zero copy and use nvls. We use 3 thread-blocks to do the allreduce.
The first thread-block is used to copy data from input buffer to scratch buffer, the second thread-block is used to do allreduce in scratch buffer, and the third thread-block is used to copy data from scratch buffer to output buffer.  The thread-blocks are synchronized by semaphores.
```python
nvls_chan = SwitchChannel(rank_list=[0, 1], buffer=Buffer.scratch)
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    chan = Channel(dst_rank, src_rank, channel_type=Channel.memory)
    chan1 = Channel(dst_rank, src_rank, channel_type=Channel.memory)
    rank = Rank(i)
    sem0 = Rank.Semaphore(rank=i, size=1)
    sem1 = Rank.Semaphore(rank=i, size=1)
    input_buffer = rank.get_input_buffer()
    output_buffer = rank.get_output_buffer()
    scratch_buffer = rank.Buffer(scratch_buffer_size)
    with Loop.iteration(unit=2**20, num_chunks=1) as iter:
        # copy data to scratch buffer
        for offset in range(nranks):
            dst_chunk = scratch_buffer[offset:offset+1]
            src_chunk = input_buffer[offset:offset+1]
            rank.copy(dst_chunk, src_chunk, tb=0, iter_context=iter)
        chan.signal(tb=0, sync="before")
        chan.wait(tb=0, sync="after")
        sem0.release(tb=0)

        # do allreduce in scratch buffer
        sem0.acquire(tb=1, sync="after")
        nvls_chan.group_load_reduce(offset, size=1, op="sum", tb=0, iter_context=iter)
        nvls_chan.group_store(offset, size=1, tb=0, iter_context=iter)
        chan1.signal(tb=1, sync="before")
        sem1.release(tb=1)

        # copy data back to output buffer
        sem1.acquire(tb=2)
        chan1.wait(tb=2, sync="after")
        for index in range(nranks):
            dst_chunk = output_buffer[index:index+1]
            src_chunk = scratch_buffer[index:index+1]
            rank.copy(dst_chunk, src_chunk, tb=2, iter_context=iter)
```

## All2All support
For now, DSL only support static all2all algorithm. For all2allv support, we need to get the send/recv size at the runtime. It may require some placeholder at the Json execution plan and relace to the real size at the runtime. If we could make chunk size be variable, we could use the same way to support all2allv.
