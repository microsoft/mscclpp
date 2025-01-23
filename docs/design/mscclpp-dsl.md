# MSCCL++ DSL
## MSCCLPPLang Introduction
MSCCLPPLang is a Python moudule for writing high-performance commnunication algorithms. It is designed to be easy to use and efficient, while providing a high-level interface for writing communication algorithms. MSCCLPPLang program will be compiled to json based execution plan, which can be executed by MSCCL++ executor.

## How to use MSCCLPPLang
### Install mscclpp package
```bash
git clone https://github.com/microsoft/mscclpp.git
cd mscclpp
pip install .
```

### Import mscclpp language module
```python
import mscclpp.language *
from mscclpp.language.types import ChannelType, ReplicationPolicy
from mscclpp.language.collectives import AllGather

instances = 1
size = gpus
collective = AllGather(size, chunk_factor=1, inplace=True)
with MSCCLPPProgram(
    "allgather",
    collective,
    size,
    instances,
    protocol="Simple",
    replication_policy=ReplicationPolicy.interleaved,
):
    pass
```

## How MSCCLPPLang Works
MSCCLPPLang provides a high-level interface for writing communication algorithms. We treat the communication algorithm as a graph, where the nodes are the data and the edges are the communication operations. The graph is represented as a Python program, which is compiled to a json based execution plan.

### Core Concepts

#### MSCCLPPProgram
A MSCCLPPProgram provides the context to write MSCCLPPLang program, which can be initialized with `with` statement in Python. Its parameters include:

- `name`: Name of this program.
- `collective`: Collective type of this program, should be from `mscclpp.language.collectives`.
- `instances`: Number of parallel instances of this program. Please see the [Instance](#instance) section for more details.
- `protocol`: Data transmission protocol used in this program, can be `LL` or `Simple`. Optional, default is `Simple`.
- `instr_fusion`: Whether low-level instruction fusion is enabled. Optional, default is `True`.
- `replication_policy`: Data replication policy, should be from `mscclpp.language.types.ReplicationPolicy`. Optional, default is `duplicated`. Please see the [Instance](#instance) section for more details.
- `num_threads_per_block`: Thread block size. Optional, default is `1024`.
- `use_double_scratch_buffer`: Whether requires double scratch buffer during execution. Optional, default is `False`.

### Collective:
A collective is a communication operation that involves multiple GPUs. We provide a set of collective operations for users to utilize. For example, the `AllGather` operation gathers data from all GPUs to all GPUs. To instantiate a collective, the user needs to specify the number of ranks, the chunk factor (how many chunks the input buffer will be split into), and whether the operation is in-place.

#### Chunk
A chunk is a piece of data that is sent between GPUs. It is the basic unit of data in MSCCLPPLang. Chunk can be a piece of data from input buffer, output buffer or intermediate buffer.
Example of creating a chunk:
```python
c = chunk(rank, Buffer.input, index, size)
```
- rank: the rank of the GPU that the chunk belongs to.
- buffer: the buffer that the chunk belongs to. It can be Buffer.input, Buffer.output or Buffer.scratch.
- index: the index of the chunk in the buffer.
- size: the number of unit chunks.

Assume we split the input data in the buffer into 4 chunks. On GPU rank 0, we can retrieve the chunks from indices 0 to 2 using the following command:
```python
c = chunk(0, Buffer.input, 0, 2)
```

#### Operation
The operation can only be applied to the chunks. We provide a set of communications operations for the users to use. For example, the `put` operation is used to send the data from one GPU to another GPU. The `get` operation is used to receive the data from another GPU.

***Please notice***: MSCCLPPLang only provides one-sided communication operations. The user needs to make sure that the data is ready to be sent or received before calling the communication operations. Also we provides `wait/signal` operations to synchronize the communication across GPUs.

#### Channel
A channel is a communication channel between two GPUs. It is used to send and receive data between GPUs. We supports three types of channel: `ChannelType.memory`, `ChannelType.port` and `ChannelType.nvls`.

`ChannelType.memory` is used for communication between GPUs on the same node. This channel uses GPU processors to transfer data.

`ChannelType.port` is used for communication between GPUs, whether they are on different nodes or the same node. This channel will offload the data transfer to CPU processors, which can provide better throughput compared to `ChannelType.memory`. However, this comes at the cost of higher latency compared to `ChannelType.memory`.

`ChannelType.nvls` is used for communication between GPUs on the same node. This feature offloads the data processing task to the switch, requiring specific hardware support. Refer [nvdia documentation](https://www.nvidia.com/en-us/data-center/nvlink/) for more details.

#### Thread Block
We can assign operations to a thread block. The thread block is a group of threads that are executed together on the GPU. In the operation function, we can specify the thread block that the operation belongs to via `sendtb` or `recvtb` parameter.

#### Instance
An instance is a parallel execution of the program. For example, if a collective algorithm is designed to run on `n` chunks with `m` thread blocks, setting the instance to 2 will run the algorithm on `2n` chunks with `2m` thread blocks. Serveral replication policies are supported, including `duplicated` and `interleaved`.
- `duplicated`: Each chunk is split into smaller parts based on the number of instances, duplicating the same instructions for all parts. For example, ChunkA is split into ChunkA0 and ChunkA1, while ChunkB is split into ChunkB0 and ChunkB1. Both ChunkA0 and ChunkA1 belong to Instance 0, and both ChunkB0 and ChunkB1 belong to Instance 1.
- `interleaved`: Assign chunks to instances in an interleaved manner. For example, ChunkA and ChunkB are split into to ChunkA0, ChunkA1, ChunkB0, and ChunkB1. ChunkA0 and ChunkB0 belong to Instance 0, while ChunkA1 and ChunkB1 belong to Instance 1.

#### Instruction Fusion
MSCCLPPLang provides the instruction fusion mechanism to fuse multiple operations into a single kernel. This can reduce the overhead of launching multiple instructions. When users create the MSCCLPPLang program, they can specify the `instr_fusion` parameter to enable the instruction fusion. By default, the instruction fusion is enabled.

## MSCCLPPLang APIs

### Basic APIs
- `chunk(rank, buffer, index, size)`: create a chunk.
- `put(self, dst, buffer, index, sendtb, chan_type)`: send the data from one GPU to another GPU. User can specify the index of the chunk in the destination buffer, the sendtb and the channel type.
- `get(self, src, buffer, index, recvtb, chan_type)`: receive the data from another GPU. User can specify the index of the chunk in the destination buffer, the recvtb and the channel type.
- `signal(self, dst, buffer, index, sendtb, chan_type)`: send a signal to another GPU.
- `wait(self, src, buffer, index, recvtb, chan_type)`: wait for a signal from another GPU.
- `flush(self, dst, buffer, index, sendtb, chan_type)`: flush the data in the buffer to the destination GPU. This is used to make sure the data is sent to the destination GPU.
- `copy(self, dst, buffer, index, sendtb)`: copy the data from one buffer to another buffer in the same GPU.
- `reduce(self, other_chunkref, recvtb, channel_type)`: Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref

### Packet APIs
Packet APIs are used when user wants to use LL algorithm. The packet APIs are similar to the basic APIs, it will packet the data and flags into a packet and send the packet to the destination GPU. The destination GPU will unpack the packet and get the data and flags. So no synchronization is needed when using packet APIs. (`ChannelType.nvls` does not support packet APIs)
- `packet_put(self, dst, buffer, index, sendtb, chan_type)`: send the data from one GPU to another GPU using packet.
- `copy_packet(self, dst, buffer, index, sendtb)`: copy the data from one buffer to another buffer in the same GPU using packet.
- `reduce_packet(self, other_chunkref, recvtb)`: Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref using packet.


### Examples
We provide several examples demonstrating how to use the MSCCL++ DSL to write communication collective algorithms. For more details, please refer to the [examples](https://github.com/microsoft/mscclpp/tree/main/mscclpp-lang/python/examples) folder.