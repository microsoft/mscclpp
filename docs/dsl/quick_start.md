# Quick Start

The MSCCL++ DSL (Domain Specific Language) provides a high-level Python API for defining custom collective communication algorithms. This guide will help you get started with writing and testing your own communication patterns.

## Installation

You can follow the same steps in the [Quick Start](quickstart).

After finishing the installation in the quick start section, you can add the following steps to install some default algorithms from the DSL:

```bash
python3 -m mscclpp --install
```

## Your First Algorithm: AllGather

Let's walk through a simple AllGather algorithm to understand the DSL basics. This example demonstrates the key concepts without diving into all the advanced features.

### Complete Example

```python
from mscclpp.language import *

def simple_allgather(name):
    """
    A simple AllGather implementation using the MSCCL++ DSL.
    
    This example demonstrates a 2-GPU AllGather where each GPU sends
    its data to all other GPUs, so all GPUs end up with everyone's data.
    
    Args:
        name: Algorithm name for identification
    """
    num_gpus = 2
    chunk_factor = 1  # Split data into num_gpus chunks
    
    # Define the collective operation
    collective = AllGather(num_gpus, chunk_factor, inplace=True)
    
    # Create the program context
    with CollectiveProgram(
        name,
        collective,
        num_gpus,
        protocol="Simple",  # Use Simple protocol (vs "LL" for low-latency)
        min_message_size=0,
        max_message_size=2**30  # 1GB
    ):
        # Loop over each source GPU rank
        for src_rank in range(num_gpus):
            # Create a Rank object for the source GPU
            rank = Rank(src_rank)
            # Get the output buffer where the data is stored
            src_buffer = rank.get_output_buffer()
            # Take a slice corresponding to this rank's data
            src_chunk = src_buffer[src_rank:src_rank + 1]
            
            # Loop over each destination GPU rank
            for dst_rank in range(num_gpus):
                # Skip sending from a rank to itself
                if src_rank != dst_rank:
                    # Create a Rank object for the destination GPU
                    dst_rank_obj = Rank(dst_rank)
                    # Get the destination buffer where data will be sent
                    dst_buffer = dst_rank_obj.get_output_buffer()
                    # Take a slice where the data will be placed
                    dst_chunk = dst_buffer[src_rank:src_rank + 1]
                    
                    # Define a channel from src_rank → dst_rank
                    channel = MemoryChannel(dst_rank, src_rank)
                    
                    # Step 1: Source signals it is ready to send data
                    channel.signal(tb=0, relaxed=True)
                    
                    # Step 2: Wait for destination to be ready
                    channel.wait(tb=0, data_sync=SyncType.after, relaxed=True)
                    
                    # Step 3: Source rank sends data to destination rank
                    channel.put(dst_chunk, src_chunk, tb=0)
                    
                    # Step 4: Signal that put operation is complete
                    channel.signal(tb=0, data_sync=SyncType.before)
                    
                    # Step 5: Wait for acknowledgment
                    channel.wait(tb=0, data_sync=SyncType.after)
            
        print(JSON())

simple_allgather("simple_allgather_2gpus")
```

### Key Concepts Explained

**1. Collective Definition**
```python
collective = AllGather(num_gpus, chunk_factor=1, inplace=True)
```
- Defines what collective operation to implement (AllGather in this case)
- `chunk_factor` determines data chunking strategy
- `inplace=True` means input and output use the same buffer. For AllGather, the input buffer is a slice of the output buffer. For example, on rank 0, the input buffer is the first half of the output buffer, and on rank 1, the input buffer is the second half of the output buffer.

**2. Program Context**
```python
with CollectiveProgram(name, collective, num_gpus, ...):
```
- Sets up the execution environment
- Configures protocol, threading, and message size ranges

**3. Ranks and Buffers**
```python
rank = Rank(src_rank)
src_buffer = rank.get_output_buffer()
src_chunk = src_buffer[src_rank:src_rank + 1]
```
- `Rank` represents a GPU in the collective
- Buffers hold the data being communicated
- Chunks are slices of buffers representing data portions

**4. Channels**
```python
channel = MemoryChannel(dst_rank, src_rank)
```
- Establishes communication paths between GPUs
- `MemoryChannel` for intra-node (fast, direct memory access)
- Created for each source-destination pair
- Can also use `PortChannel` for inter-node communication

**5. Synchronization and Data Transfer**
```python
channel.signal(tb=0, relaxed=True)
channel.wait(tb=0, data_sync=SyncType.after, relaxed=True)
channel.put(dst_chunk, src_chunk, tb=0)
```
- `signal()`: Notify remote GPU of state changes
- `wait()`: Wait for remote GPU to reach a certain state
- `put()`: Write data from local to remote GPU memory
- `tb=0` assigns operations to thread block 0
- `relaxed=True` uses relaxed memory ordering for performance

For more advanced concepts like synchronization, scratch buffers, and pipelining, refer to the [full DSL documentation](py_api).

## Testing Your Algorithm

Once you've written your algorithm, you need to run it:

```bash
python3 path/to/simple_allgather.py > /path/to/simple_allgather.json
```

After this, use `executor_test.py` to validate correctness and measure performance.

```bash
# Test with 2 GPUs on a single node
mpirun --allow-run-as-root -np 2 python3 python/test/executor_test.py \
 -path /path/to/simple_allgather.json \
 --size 1M \
 --in_place
```

## Next Steps

Now that you understand the basics:

1. **Explore Examples**: Check `python/mscclpp/language/tests/` for more algorithm examples
2. **Optimize**: Experiment with different chunk strategies, pipelining, and synchronization patterns
3. **Advanced Features**: Learn about scratch buffers, thread block groups, and packet-based communication

For detailed API documentation and advanced features, refer to:
- [Programming Guide](programming_guide)
- [Tutorials](tutorials)

## Troubleshooting

**Import Error**: If you see `ModuleNotFoundError: No module named 'mscclpp'`, ensure you've installed the package with `pip install .`

For more help, please file an issue on the [GitHub repository](https://github.com/microsoft/mscclpp/issues).
