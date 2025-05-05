# MSCCL++ DSL Concepts
### Channel
All cross ranks related op is done through channels. A channel is a communication medium between ranks. It can be a network socket, shared memory, or any other form of IPC. The channel is responsible for sending and receiving messages between ranks.

Exmpales:
```python
channel = Channel(dst_rank, src_rank, channel_type, tag)
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
    channel = Channel(dst_rank, src_rank, channel_type=Channel.memory, tag=0)
    channel.relaxSignal(tb=0, sync=None)
    channel.wait(tb=0, sync="after")
```

#### For nvls based channel
NVLS channel need to bind with a group of buffers from a set of ranks. Each operation on the channel will be performed on the buffers in the group.

```python
nvls_chan = SwitchChannel(rank_list=[], buffer=Buffer.input, tag=0) # this interface may need to refine
nvls_chan.group_load_reduce(index1, size1, op="sum", tb=0)
nvls_chan.group_store(index, size, tb=0)
```

Example for two ranks allreduce.
```python
# allreduce for 2 ranks.
# 1. copy data from input buffer to scratch buffer
# 2. allreduce the data in scratch buffer
# 3. copy data from scratch buffer to output buffer
nvls_chan = SwitchChannel(rank_list=[0, 1], buffer=Buffer.scratch, tag=0)
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    chan = Channel(dst_rank, src_rank, channel_type=Channel.memory, tag=0)
    rank = Rank(i)
    chunk_index = src_rank
    # copy data to scratch buffer
    dst_chunk = Chunk(src_rank, Buffer.scatch, chunk_index, 1)
    src_chunk = Chunk(src_rank, Buffer.input, chunk_index, 1)
    rank.copy(dst_chunk, src_chunk, tb=0)
    chan.signal(tb=0, sync="before")
    chan.wait(tb=0, sync="after")

    # do allreduce in scratch buffer
    nvls_chan.group_load_reduce(chunk_index, 1, op="sum", tb=0)
    nvls_chan.group_store(chunk_index, 1, tb=0)

    # copy data back to output buffer
    chan.signal(tb=0, sync="before")
    chan.wait(tb=0, sync="after")
    dst_chunk = Chunk(src_rank, Buffer.output, chunk_index, 1)
    src_chunk = Chunk(src_rank, Buffer.scratch, chunk_index, 1)
    rank.copy(dst_chunk, src_chunk, tb=0)
```

### Chunk/Tensor
Chunk is a data structure that holds the data to be sent or received. For some local operations, such as copy/reduce we provide some functions to manipulate the chunk.

```python
dst = Chunk(rank, index, size)
scr = Chunk(rank, index + 1, size)
Rank.copy(dst_chunk, src_chunk, tb=0)
Rank.reduce(dst_chunk, src_chunk, op="sum", tb=0)
```

### Synchronization
We provide some synchronization primitives to sync threadblocks inside a rank. The synchronization is done through a barrier or semaphore. The barrier is used to synchronize all threadblocks in a rank, while the semaphore is used to synchronize a specific threadblock.
```python
rank.barrier(tb_list=[])
sem = Rank.Semaphore(size=1, tag=0)
sem.acquire(tb=0)
sem.release(tb=0)
```

The synchronization inside the thread-block can be inferred by MSCCL++ DSL auotmaticly. Which mean if we have data dependence between two operations, we will insert a synchronization point between them. 

But for multi-thread-blocks synchronization and cross ranks synchronization, we need to insert the synchronization point manually.

## For kernel fusion
We only fuse the kernel that in the same thread-block. We still need to construct the DAG for each thread-block. Track the chunk usage and see if we can fuse the kernel.


## For Pipeline Loop
For some cases, we need to pipeline the kernel to overlap some operations. For example, the first stage is copy data from input buffer to scratch buffer, the second stage is transfer data from scratch buffer to other peers. We could use `Rank.semphore` to synchronize the two stages. 
```python
sem = Rank.Semaphore(size=1, tag=0)
Rank.copy(dst_chunk, src_chunk, tb=0)
sem.release(tb=0)
channel = Channel(dst_rank, src_rank, channel_type, tag)
sme.acquire(tb=1)
channel.put(dst_chunk, src_chunk, tb=1)
```

Also we could provide some gramar sugar to make the pipeline more readable. For example, we could use `Loop` to construct the pipeline. 
```python
sem = Rank.Semaphore(size=1, tag=0)
with Loop.iteration(unit=2**20, num_chunks=1) as iter:
    # the dst_chunk and src_chunk size but same as loop context
    Rank.copy(dst_chunk, src_chunk, tb=0, iter_context=iter)
    sem.release(tb=0)
    channel = Channel(dst_rank, src_rank, channel_type, tag)
    sme.acquire(tb=1)
    channel.put(dst_chunk, src_chunk, tb=1, iter_context=iter)
``` 


Here is the example for two ranks allreduce. Which achieve non-zero copy and use nvls. We use 3 thread-blocks to do the allreduce.
The first thread-block is used to copy data from input buffer to scratch buffer, the second thread-block is used to do allreduce in scratch buffer, and the third thread-block is used to copy data from scratch buffer to output buffer.  The thread-blocks are synchronized by semaphores.
```python
nvls_chan = SwitchChannel(rank_list=[0, 1], buffer=Buffer.scratch, tag=0)
nranks = 2
for i in range(nranks):
    src_rank = i
    dst_rank = (i + 1) % nranks
    chan = Channel(dst_rank, src_rank, channel_type=Channel.memory, tag=0)
    chan1 = Channel(dst_rank, src_rank, channel_type=Channel.memory, tag=1)
    rank = Rank(i)
    chunk_index = src_rank
    sem0 = Rank.Semaphore(size=1, tag=0)
    sem1 = Rank.Semaphore(size=1, tag=1)
    with Loop.iteration(unit=2**20, num_chunks=1) as iter:
        # copy data to scratch buffer
        dst_chunk = Chunk(src_rank, Buffer.scatch, chunk_index, 1)
        src_chunk = Chunk(src_rank, Buffer.input, chunk_index, 1)
        rank.copy(dst_chunk, src_chunk, tb=0, iter_context=iter)
        chan.signal(tb=0, sync="before")
        chan.wait(tb=0, sync="after")
        sem0.release(tb=0)

        # do allreduce in scratch buffer
        sem0.acquire(tb=1, sync="after")
        nvls_chan.group_load_reduce(chunk_index, 1, op="sum", tb=0, iter_context=iter)
        nvls_chan.group_store(chunk_index, 1, tb=0, iter_context=iter)
        chan1.signal(tb=1, sync="before")
        sem1.release(tb=1)

        # copy data back to output buffer
        sem1.acquire(tb=2)
        chan1.wait(tb=2, sync="after")
        dst_chunk = Chunk(src_rank, Buffer.output, chunk_index, 1)
        src_chunk = Chunk(src_rank, Buffer.scratch, chunk_index, 1)
        rank.copy(dst_chunk, src_chunk, tb=2, iter_context=iter)
```