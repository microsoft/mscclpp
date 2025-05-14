# MSCCL++ DSL Json File
The case of use of the MSCCL++ DSL consist of creating a JSON file wich will generate of the python script algorithm, in this JSON file will have all the detailed description of the operation defined in the algorithm. This JSON file needs to be spread for all the machines after this the MSCCL++ executor can be trigger to use 
this JSON file to generate the execution plan to execute the algorithm.

![alt text](../figs/mscclpp_dsl_json_schema.png)

The generic structure of the JSON file consist of the following fields:

- ```name```: Consist of the name of the algorithm, this is usefull if you want to use different algorithm for the same collective, for instance two allreduce algorithm, the name can differ them and create/store to setups one for each case.
- ```collective```: Consist in the collective algorotihm, for instance it can assume the values: allreduce, allgather, reducescatter, broadcast, alltoall.
- ```protocol```: Consist of the how the data will be storage, it can be Simple or LL for low latency protocol. The LL case the data will be storage with a flag to avoid signal and await sinchronization and use the scratch buffer as a intermediary buffer to store the data in this different format.
- ```inplace```: Consist in the flag to setup the envinronment for inplace or outplace algorithm.
- ```gpus```: This will describe all the details of the gpu and the operation execute on it. We will go deep on this in the next sections
- ```num_threads_per_block```: Consiste in the number of threads inside each thread block, for instance it can be: 256, 512, 768, 1024. Depending on your necessity.
- ```use_double_scratch_buffer```: Consist in a bollean value flag to give the option to duplicate the buffer, each executor call will use one side of the buffer, allowing avoid conflicts in some algorithms.
- ```min_message_size```: Consist in the minimum message size that this algorithm support, so if the message sizes is smaller than this this JSON file will not be choose to run.
- ```max_message_size```: Consist in the maximum message size that this algorithm support, so if the message sizes is bigger than this this JSON file will not be choose to run.

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
The gpu field is the core of the JSON file, where we have the detailed description of the collective algorithm fior each gpu, the gpu consist in a list describing each gpu, for each element in the list we have the following fields:

- ```id```: The GPU id
- ```inputChunks```: The amount of chunks in the input buffer.
- ```outputChunks```: The amount of chunks in the output buffer.
- ```scratchChunks```: The amount of chunks in the scratch buffer.
- ```chunkGroups```: The amount of chunk groups.
- ```threadblocks```: Consist in a list with the description of all the operation that will happen in each thread block.
- ```channels```: Consist in the list of all channel, which describe each each channel.

### Channels
The channel field describe the caratcterist of the channels, basically we have 3 type of channels:

- Memory Channel: 
- Port Channel:
- Switch Channel:

The Memory Channel has the following fields:
- ```type```: Specify the type of the channel, in the Memory Channel case it values will be ```memory```.
- ```connectedTo```: Specify which channel is connected to how. So if we have a list like: [1, 1, 3], we know here that we have 3 channels the first one is connected with the rank 1, the second with the rank 2 and the third one with the rank 3.

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
- ```type```: Specify the type of the channel, in the Port Channel case it values will be ```port```.
- ```connectedTo```: Specify which channel is connected to how. So if we have a list like: [1, 1, 3], we know here that we have 3 channels the first one is connected with the rank 1, the second with the rank 2 and the third one with the rank 3.

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
- ```type```: Specify the type of the channel, in the Switch Channel case it values will be ```switch```.
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

### Thread Block
The thread block field describe the operation inside each thread block, we have the following fields:

- ```id```: The thread block id.
- ```ops```: The list with all the operation in order they will be executed by this tread block.
- ```channels```: The channels the thread block will use, referenced by the channel id. The channel id is based on the global channel description in the gpu, for example if the channel type is memory and the channel id is 0, it refers to the first channel of memory id type descriptioned in the gpu channels field.

The most important field here is the ops fields, where there is the description for all the operations, each operation has specific fields, let's given an overview in all the possible fields and after this go throught each of them:

- ```name```:
- ```ctype```:
- ```i_cids```:
- ```i_buff```
- ```srcs```:
- ```o_cids```:
- ```o_buff```
- ```dsts```:
- ```srcbuff```:
- ```srcoff```:
- ```dstbuff```:
- ```dstoff```:
- ```cnt```:
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
      "src": "o",
      "src_off": 0,
      "dst": "o",
      "dst_off": 0,
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
