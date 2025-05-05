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

#### For nvls based channel
NVLS channel need to bind with a group of buffers from a set of ranks. Each operation on the channel will be performed on the buffers in the group.

```python
nvls_chan = SwitchChannel(rank_list=[], buffer_type=Buffer.input, tag=0)
nvls_chan.group_reduce(index1, size1)
nvls_chan.group_broadcast(index, size)
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

The synchronization inside the thread-block can be infered by MSCCL++ DSL auotmaticly. Which mean if we have data dependence between two operations, we will insert a synchronization point between them. 

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
