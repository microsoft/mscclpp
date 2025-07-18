# MSCCL++ DSL Execution plan
The MSCCL++ DSL is used to generate a JSON file that describes the structure and behavior of a collective communication algorithm. It contains a detailed specification of the operations involved and must be distributed to all participating machines. Once distributed, the MSCCL++ executor can be triggered to use this JSON file to generate an execution plan and run the algorithm.

![alt text](../figs/mscclpp_dsl_json_schema.png)

The generic structure of the JSON file consist of the following fields:

- ```name```: The name of the algorithm. This is useful when you want to define multiple algorithms for the same collective operation (e.g., two different allreduce algorithms). The name helps distinguish and store two different execution plan and context for them.
- ```collective```: Specifies the type of collective operation. Supported values include: ```allreduce```, ```allgather```, ```reducescatter```, ```broadcast```, and ```alltoall```.
- ```protocol```: Defines how the data is stored and transferred. It can be either Simple or LL (Low Latency). In the LL case, data is stored with flags to avoid signal-based synchronization and uses a scratch buffer as an intermediate storage format.
- ```inplace```: A boolean flag indicating whether the algorithm operates in-place or out-of-place.
- ```gpus```: Describes the GPU-specific configuration and the operations executed on each GPU. This will be detailed in the following sections.
- ```num_threads_per_block```: Specifies the number of threads per thread block. Typical values include 256, 512, 768, or 1024, depending on performance requirements.
- ```use_double_scratch_buffer```: A boolean flag that enables to double the scratch buffer. When set to true, the size of the scratch buffer is doubled and it is logically divided into two halves. This allows alternating executions of the same algorithm to use different halves of the buffer, for instance, odd-numbered executions use the first half, and even-numbered executions use the second. This strategy helps avoid memory conflicts when the same algorithm is invoked multiple times.
- ```buffer_alignment```: The requirement that a buffer's memory address must be a multiple of a specific number of bytes.
- ```min_message_size```: The minimum message size supported by this algorithm. If the message is smaller than this value, the algorithm will not be selected for execution.
- ```max_message_size```: The maximum message size supported by this algorithm. If the message exceeds this value, the algorithm will not be selected for execution.
- ```reuse_resources```: A boolean flag indicating whether the algorithm can reuse resources from previous executions. This is useful for algorithms that can be executed multiple times without needing to reallocate resources, such as buffers, channels and semaphores.

Example:

```json
{
  "name": "hierarchical_ring_reducescatter",
  "collective": "reducescatter",
  "protocol": "Simple",
  "inplace": true,
  "gpus": [...],
  "num_threads_per_block": 1024,
  "use_double_scratch_buffer": false,
  "buffer_alignment": 16,
  "min_message_size": 0,
  "max_message_size": 18446744073709551615,
  "reuse_resources": false,
}
```

## GPU
The gpus field is the core of the JSON file, containing the detailed configuration of the collective algorithm for each GPU. It is defined as a list, where each element describes the setup for a specific GPU. Each GPU entry includes the following fields:

- ```id```: The identifier of the GPU.
- ```input_chunks```: The number of chunks in the input buffer.
- ```output_chunks```: The number of chunks in the output buffer.
- ```scratch_chunks```: The number of chunks in the scratch buffer.
- ```threadblocks```: A list describing all operations assigned to each thread block. Each entry defines how a thread block participates in the collective operation.
- ```channels```: A list of communication channels, where each element describes a channel.
- ```remote_buffers```: A list with all the remote buffers used for that GPU on the algorithm.

Example:

```json
"gpus": [
    {
      "id": 0,
      "input_chunks": 1,
      "output_chunks": 2,
      "scratch_chunks": 0,
      "threadblocks": [...],
      "channels": [...],
      "remote_buffers": [...],
    }
]
```

### Channels
The channel field describes the characteristics of the channels. Basically, we have three types of channels:

- Memory Channel: A MemoryChannel wraps data transfer methods that use thread-copy mechanism, ie., directly use GPU threads for writing to peer GPUs memory.
- Port Channel: A PortChannel implements primitives when data transfer is done over ports connected to GPU memory, such as cudaMemcpy for intra-node DMA or ibv_post_send for RDMA. 
- Switch Channel: A SwitchChannel provides primitives for performing collective operations among GPUs. These operations usually require specialized hardware sup-
port.

The Memory Channel has the following fields:
- ```type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```memory```.
- ```connected_to```: Specifies the connections between channels. For example, if we have a list like: [1, 2, 3], it indicates that there are three channels: the first is connected to rank 1, the second to rank 2, and the third to rank 3.

Example:

```json
"channels": [
  {
    "channel_type": "memory",
    "connected_to": [
      1,
      2,
      3
    ]
  }
]
```

The Port Channel has the following fields:
- ```channel_type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```port```.
- ```connected_to```: Specifies the connections between channels. For example, if we have a list like: [1, 2, 3], it indicates that there are three channels: the first is connected to rank 1, the second to rank 2, and the third to rank 3.

Example:

```json
"channels": [
  {
    "channel_type": "port",
    "connected_to": [
      1,
      2,
      3
    ]
  }
]
```

The Switch Channel has the following fields:
- ```channel_type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```switch```.
- ```buffer_type```: Consist of the buffer type which the Switch Channel will be binded, this could have the following values: "i" for the input buffer, "o" for the output buffer, "s" for the scratch buffer.
- ```rank_groups```: Consist of a group of ranks connected via the Switch Channel, including the number of ranks (size) and the list of connected ranks.

Example:

```json
"channels": [
  {
    "channel_type": "switch",
    "buffer_type": "i",
    "rank_groups": [
      {
        "size": 8,
        "ranks": [0, 1, 2, 3, 4, 5, 6, 7]
      }
    ]
  }
]
```

### Remote Buffers
The ```remoteBuffers``` field describes all the remote buffers that a given GPU needs to access. Each entry in this list contains the following fields:

- ```rank```:  Indicates the rank that owns the remote buffer.
- ```type```: Consist of the buffer type, this could have the following values: "i" for the input buffer, "o" for the output buffer, "s" for the scratch buffer.
- ```access_channel_types```: A list specifying what types of channels we should use to manage this buffer.

Example

```json
"remoteBuffers": [
  {
    "rank": 1,
    "type": "i",
    "access_channel_types": ["memory"]
  }
]
```

### Thread Blocks
The thread block field describes the operation inside each thread block, we have the following fields:

- ```id```: The thread block id.
- ```ops```: The list of all operations in the order they will be executed by this thread block.
- ```channels```: The channels the thread block will use, referenced by the channel id. The channel id is based on the global channel description in the gpu, for example if the channel type is memory and the channel id is 0, it refers to the first channel of memory id type descriptioned in the gpu channels field.
- ```remote_buffer_ids```: A list with all the remote buffer ids(related to the remote buffer field on the GPU) used by the thread block.

For Example:

```json
"threadblocks": [
  {
    "id": 0,
    "ops": [...],
    "channels":[
      {
        "channel_type": "memory",
        "channel_ids": [0]
      }
    ],
    "remote_buffer_refs": [
      {
        "access_channel_type": "memory",
        "remote_buffer_ids": [0]
      }
    ]
  }
]
```