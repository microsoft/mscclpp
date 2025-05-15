# MSCCL++ DSL Json File
The MSCCL++ DSL is used to define a JSON file that describes the structure and behavior of a collective communication algorithm. It contains a detailed specification of the operations involved and must be distributed to all participating machines. Once distributed, the MSCCL++ executor can be triggered to use this JSON file to generate an execution plan and run the algorithm.

![alt text](../figs/mscclpp_dsl_json_schema.png)

The generic structure of the JSON file consist of the following fields:

- ```name```: The name of the algorithm. This is useful when you want to define multiple algorithms for the same collective operation (e.g., two different allreduce algorithms). The name helps distinguish and store two different execution plan and context for them.
- ```collective```: Specifies the type of collective operation. Supported values include: ```allreduce```, ```allgather```, ```reducescatter```, ```broadcast```, and ```alltoall```.
- ```protocol```: Defines how the data is stored and transferred. It can be either Simple or LL (Low Latency). In the LL case, data is stored with flags to avoid signal-based synchronization and uses a scratch buffer as an intermediate storage format.
- ```inplace```: A boolean flag indicating whether the algorithm operates in-place or out-of-place.
- ```gpus```: Describes the GPU-specific configuration and the operations executed on each GPU. This will be detailed in the following sections.
- ```num_threads_per_block```: Specifies the number of threads per thread block. Typical values include 256, 512, 768, or 1024, depending on performance requirements.
- ```use_double_scratch_buffer```: A boolean flag that enables to double the scratch buffer. When set to true, the scratch buffer double the size and is logically divided into two halves. This allows alternating executions of the same algorithm to use different halves of the buffer, for instance, odd-numbered executions use the first half, and even-numbered executions use the second. This strategy helps avoid memory conflicts when the same algorithm is invoked multiple times.
- ```min_message_size```: The minimum message size supported by this algorithm. If the message is smaller than this value, the algorithm will not be selected for execution.
- ```max_message_size```: The maximum message size supported by this algorithm. If the message exceeds this value, the algorithm will not be selected for execution.

Example:

```json
{
  "name": "reducescatter",
  "collective": "reducescatter",
  "protocol": "Simple",
  "inplace": true,
  "gpus": [...],
  "num_threads_per_block": 1024,
  "use_double_scratch_buffer": false,
  "min_message_size": 0,
  "max_message_size": 18446744073709551615
}
```

## GPU
The gpus field is the core of the JSON file, containing the detailed configuration of the collective algorithm for each GPU. It is defined as a list, where each element describes the setup for a specific GPU. Each GPU entry includes the following fields:

- ```id```: The identifier of the GPU.
- ```inputChunks```: The number of chunks in the input buffer.
- ```outputChunks```: The number of chunks in the output buffer.
- ```scratchChunks```: The number of chunks in the scratch buffer.
- ```chunkGroups```: The number of chunk groups used in the algorithm.
- ```threadblocks```: A list describing all operations assigned to each thread block. Each entry defines how a thread block participates in the collective operation.
- ```channels```: A list of communication channels, where each element describes a channel.
- ```bufferCollection```: A list with all the remote buffers used on the algorithm.

### Channels
The channel field describes the characteristics of the channels. Basically, we have three types of channels:

- Memory Channel: A MemoryChannel wraps data transfer methods that use thread-copy mechanism, ie., directly use GPU threads for writing to peer GPUs memory.
- Port Channel: A PortChannel implements primitives when data transfer is done over ports connected to GPU memory, such as cudaMemcpy for intra-node DMA or ibv_post_send for RDMA. 
- Switch Channel: A SwitchChannel provides primitives for performing collective operations among GPUs. These operations usually require specialized hardware sup-
port.

The Memory Channel has the following fields:
- ```type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```memory```.
- ```connectedTo```: Specifies the connections between channels. For example, if we have a list like: [1, 2, 3], it indicates that there are three channels: the first is connected to rank 1, the second to rank 2, and the third to rank 3.

Example:

```json
"channels": [
  {
    "type": "memory",
    "connectedTo": [
      1,
      2,
      3
    ]
  }
]
```

The Port Channel has the following fields:
- ```type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```port```.
- ```connectedTo```: Specifies the connections between channels. For example, if we have a list like: [1, 2, 3], it indicates that there are three channels: the first is connected to rank 1, the second to rank 2, and the third to rank 3.

Example:

```json
"channels": [
  {
    "type": "port",
    "connectedTo": [
      1,
      2,
      3
    ]
  }
]
```

The Switch Channel has the following fields:
- ```type```: Specifies the type of the channel, which in the case of the Memory Channel, will be ```switch```.
- ```buff```: Consist of the buffer type which the Switch Channel will be binded, this could have the following values: "i" for the input buffer, "o" for the output buffer, "s" for the scratch buffer.
- ```rankGroups```: Consist of the group of the ranks connected by the Switch Channel, this field contains the size and the list of the ranks connected.

Example:

```json
"channels": [
  {
    "type": "switch",
    "buff": "i",
    "rankGroups": [
      {
        "size": 8,
        "ranks": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ]
      }
    ]
  }
]
```

### Buffer Collection
The buffer collection field will contain the description of all the remote buffers that his GPU need to access, it will contain the following fields:

-
-
-

Example

```json
"bufferCollection": [
  {
    "rank": 1,
    "buff": "i"
  }
]
```

### Thread Block
The thread block field describe the operation inside each thread block, we have the following fields:

- ```id```: The thread block id.
- ```ops```: The list with all the operation in order they will be executed by this tread block.
- ```channels```: The channels the thread block will use, referenced by the channel id. The channel id is based on the global channel description in the gpu, for example if the channel type is memory and the channel id is 0, it refers to the first channel of memory id type descriptioned in the gpu channels field.
- ```bufferCollection```: A list with all the remote buffers used by the thread block.

For Example:

```json
"threadblocks": [
  {
    "id": 0,
    "ops": [...],
    "channels":[
      {
        "ctype": "memory",
        "cids": [
          0
        ]
      }
    ],
    "bufferCollection": [
      {
        "buffid": 0
      }
    ]
  }
]
```

The most important field here is the ops fields, where there is the description for all the operations, each operation has specific fields, let's given an overview in all the possible fields and after this go throught each of them:

- ```name```: Operation name, it include: signal, wait, put, get.
- ```ctype```: Channel type, it include: memory, port, switch.
- ```i_cids```: The channel id of the channels used in the receive operations.
- ```i_buff```: Buffer information in the receive operations.
- ```srcs```: Thinking in remove this field.
- ```o_cids```: The channel id of the channels used in inter machine the send operations.
- ```o_buff```: Buffer information in the send operations.
- ```dsts```: Thinking in remove this field.
- ```srcbuff```: Buffer type in the source in intra machine operations.
- ```srcoff```: Source offset in the source buffer in intra machine operations.
- ```dstbuff```: Buffer type in the destine in intra machine operations.
- ```dstoff```: Destine offset in the destine buffer in intra machine operations.
- ```cnt```: Data size in intra machine operations.
- ```barrier_id```:
- ```nthread_blocks```:

#### Signal Operation
The signal operation is composed by the field ```name```, ```o_cids```, ```ctype```.

Example

```json
"ops": [
  {
    "name": "signal",
    "o_cids": [
      {
        "id": 0
      }
    ],
    "ctype": "memory",
  }
]
```

### Wait Operation
The signal operation is composed by the field ```name```, ```i_cids```, ```ctype```.

```json
"ops": [
  {
    "name": "wait",
    "i_cids": [
      {
        "id": 0
      }
    ],
    "ctype": "memory",
  }
]
```

### Put Operation
The put operation is composed by the field ```name```, ```o_buff```, ```o_cids```, ```ctype```.

```json
"ops": [
  {
    "name": "put",
    "o_buff": [
      {
        "srcbuff": "o",
        "srcoff": 0,
        "dstbuff": "o",
        "dstoff": 0,
        "cnt": 1
      }
    ],
    "o_cids": [
      {
        "id": 0
      }
    ],
    "ctype": "memory"
  }
]
```

## Examples

in place AllGather two Nodes:

```python
size = 2

for src_rank in range(size):
  rank = Rank(src_rank)
  src_input_buffer = rank.get_input_buffer()
  src_chunk = input_buffer[src_rank:src_rank + 1] 
  for dst_rank in range(size):
    rank = Rank(dst_rank)
    dst_input_buffer = rank.get_input_buffer()
    dst_chunk = input_buffer[src_rank:src_rank + 1] 
    if src_rank != dst_rank:
      channel = Channel(dst_rank, src_rank, channel_type)
      channel.relaxedSignal(tb=0, sync=None)
      channel.relaxedWait(tb=0, sync="after")
      channel.put(dst_chunk, src_chunk, tb=0)
      channel.signal(tb=0, sync="before")
      channel.wait(tb=0, sync="after")
```

For this example we will have the following JSON file:

```json
{
  "name": "allgather",
  "collective": "allgather",
  "protocol": "Simple",
  "inplace": true,
  "gpus": [
    {
      "id": 0,
      "inputChunks": 1,
      "outputChunks": 2,
      "scratchChunks": 0,
      "chunkGroups": 1,
      "threadblocks": [
        {
          "id": 0,
          "ops": [
            {
              "name": "rsignal",
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "rwait",
              "i_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "put",
              "o_buff": [
                {
                "srcbuff": "o",
                "srcoff": 0,
                "dstbuffid": 0,
                "dstoff": 0,
                "cnt": 1
                }
              ],
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory"
            },
            {
              "name": "signal",
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "wait",
              "i_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            }
          ],
          "channels":[
            {
              "ctype": "memory",
              "cids": [
                0
              ]
            }
          ],
          "bufferCollection": [
            {
              "buffid": 0
            }
          ]
        }
      ],
      "channels": [
        {
          "type": "memory",
          "connectedTo": [
            1
          ]
        }
      ],
      "bufferCollection": [
        {
          "rank": 1,
          "buff": "i"
        }
      ]
    },
    {
      "id": 1,
      "inputChunks": 1,
      "outputChunks": 2,
      "scratchChunks": 0,
      "chunkGroups": 1,
      "threadblocks": [
        {
          "id": 0,
          "ops": [
            {
              "name": "rsignal",
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "rwait",
              "i_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "put",
              "o_buff": [
                {
                "srcbuff": "o",
                "srcoff": 1,
                "dstbuffid": 0,
                "dstoff": 1,
                "cnt": 1
                }
              ],
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory"
            },
            {
              "name": "signal",
              "o_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            },
            {
              "name": "wait",
              "i_cids": [
                {
                  "id": 0
                }
              ],
              "ctype": "memory",
            }
          ],
          "channels":[
            {
              "ctype": "memory",
              "cids": [
                0
              ]
            }
          ],
          "bufferCollection": [
            {
              "buffid": 0
            }
          ]
        }
      ],
      "channels": [
        {
          "type": "memory",
          "connectedTo": [
            0
          ]
        }
      ],
      "bufferCollection": [
        {
          "rank": 0,
          "buff": "i"
        }
      ]
    }
  ]
}
```