# MSCCL++ DSL: Comprehensive Guide

> **Provenance & freshness**
> - **Authoritative source:** `python/mscclpp/language/` (DSL implementation) and `docs/dsl/concepts.md` (canonical conceptual docs) in this repository.
> - **Snapshot date:** 2026-05-18.
> - **Snapshot scope:** Program parameters, Rank/Buffer/Chunk API, Channel types and operations, synchronization primitives, data ops, ThreadBlocks/ThreadBlockGroups, pipeline loops, collectives, **operation fusion rules (derived from `__add__` methods of each operation class)**, instances/replication, and worked patterns.
> - **Drift risk:** Medium for §1 (parameters) and §6 (data ops); **High** for §10 (Operation Fusion) — that section transcribes fusion rules from source. If the DSL implementation changes, re-derive §10 from `python/mscclpp/language/` before relying on it for design decisions.
> - **Refresh process:** When updating, re-read the current DSL source, regenerate the parameter table in §1, the channel comparison in §4.3, and the fusion rule tables in §10, then bump the snapshot date above. Cross-check against `docs/dsl/concepts.md` and `docs/dsl/quick_start.md` for canonical phrasing.
> - **How agents should use this file:** Treat as the **primary DSL reference** during design and code generation. The repo's `docs/dsl/concepts.md` and `docs/dsl/quick_start.md` remain authoritative when this guide is unclear or appears stale; `python/mscclpp/language/` is the ultimate source of truth.

## Section index (for targeted reads)

| § | Topic | Read when |
|---|-------|-----------|
| 1 | Program structure & parameters (incl. `use_double_scratch_buffer`, `reuse_resources`, `instr_fusion`, `min/max_message_size`) | Setting program-level knobs in code generation |
| 2 | Ranks | Establishing per-rank operations |
| 3 | Buffers and Chunks | Working out chunk math (esp. AllGather `chunk_factor × num_ranks` output) |
| 4 | Channels (MemoryChannel, PortChannel, SwitchChannel) + comparison table | Choosing channels in design proposal |
| 5 | Synchronization (signal/wait, barrier, semaphore, `SyncType`, relaxed variants) | Designing the sync plan |
| 6 | Data operations (put, get, reduce, packets, copy) | Picking primitives for each step |
| 7 | Thread blocks and `ThreadBlockGroup` | Heterogeneous TB workloads, IB step layout |
| 8 | Pipeline loops (`LoopIterationContext`) | Pipelined multi-chunk designs |
| 9 | Collectives (AllReduce, AllGather, ReduceScatter, etc.) | Confirming buffer semantics |
| 10 | **Operation Fusion** — exhaustive `__add__` rule tables | Predicting which ops the compiler will fuse |
| 11 | Instances and replication | Setting `instances` and `replication_policy` |
| 12 | Complete 2-GPU AllGather example | End-to-end reference walkthrough |
| 13 | Sync primitives summary | Quick lookup |
| 14 | Common patterns (ring AllGather, NVSwitch AllReduce, etc.) | Pattern templates for the design proposal |

> File is ~66 KB — use `view_range` to load only the sections relevant to the current task rather than reading the whole file in one pass.

---

## Introduction

The MSCCL++ Domain-Specific Language (DSL) is a Python-native API for defining GPU-based collective communication algorithms. It allows you to express complex multi-GPU data movement and synchronization patterns at a high level, while the framework handles low-level CUDA code generation, dependency analysis, and operation fusion automatically.

This guide walks through every core concept in the DSL, explains when and how to use each primitive, and provides practical examples along the way.

---

## 1. Program Structure and Parameters

Every DSL program is built around a `CollectiveProgram` context manager. It is the top-level container that holds ranks, channels, buffers, and operations.

### Creating a Program

```python
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.collectives import AllReduce

collective = AllReduce(num_ranks=4, chunk_factor=1, inplace=False)

with CollectiveProgram(
    name="my_allreduce",
    collective=collective,
    num_ranks=4,
    instances=1,
    protocol="Simple",
    instr_fusion=True,
    num_threads_per_block=1024,
    use_double_scratch_buffer=False,
    buffer_alignment=16,
    min_message_size=0,
    max_message_size=2**64 - 1,
) as prog:
    # Define your communication algorithm here
    ...
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | A human-readable identifier for the program. Used in the generated JSON execution plan. |
| `collective` | `Collective` | *(required)* | The collective operation being implemented (e.g., `AllGather`, `AllReduce`, `ReduceScatter`). Determines the shape of input/output buffers. |
| `num_ranks` | `int` | *(required)* | Total number of GPUs (ranks) participating in the collective. |
| `instances` | `int` | `1` | Number of times to replicate the algorithm. Each instance operates on a separate data partition, increasing parallelism. Higher values can improve throughput for large messages. |
| `protocol` | `str` | `"Simple"` | Communication protocol. `"Simple"` is the standard mode. `"LL"` (Low-Latency) uses packet-based communication for smaller messages. |
| `instr_fusion` | `bool` | `True` | When enabled, the DSL analyzes operations within each thread block and fuses compatible ones (e.g., a `reduce` followed by a `put` on the same chunk becomes a single `reduce_send`). This reduces memory traffic and synchronization overhead. |
| `replication_policy` | `ReplicationPolicy` | `interleaved` | Controls how data chunks are distributed across instances. `interleaved` distributes chunks in round-robin fashion; `none` uses contiguous partitioning. |
| `reuse_resources` | `bool` | `False` | Whether to share channels and other resources across instances instead of creating separate ones. This is commonly used in non-zero-copy or packet-based algorithms where all communication happens through scratch buffers. In such cases, the channel setup (which involves exchanging memory registration information between ranks) is identical across iterations regardless of buffer offsets, so reusing the same channels avoids the overhead of creating and exchanging channel metadata for each instance. Typical use case: LL-protocol algorithms where channels always connect scratch buffers between ranks. |
| `num_threads_per_block` | `int` | `1024` | Number of CUDA threads per GPU thread block. Affects parallelism within each block. |
| `use_double_scratch_buffer` | `bool` | `False` | Doubles the allocated scratch buffer space. The most common use case is with the LL (Low-Latency) protocol, where there is no explicit synchronization mechanism like signal/wait. In LL mode, data is embedded with a flag for synchronization inside the packet format. To avoid data conflicts between consecutive iterations, the scratch buffer is split into two halves: the first half is used for even iterations and the second half for odd iterations. This alternation ensures that a new iteration never overwrites data that a previous iteration is still reading, since the synchronization flag embedded in each packet allows the consumer to distinguish between old and new data. |
| `buffer_alignment` | `int` | `16` | Memory alignment for buffers in bytes. |
| `min_message_size` | `int` | `0` | Minimum message size (in bytes) for which this program should be used. Useful when registering multiple algorithms for different size ranges. |
| `max_message_size` | `int` | `2^64 - 1` | Maximum message size (in bytes) for which this program should be used. |

### Using `AlgoSpec` for Configuration

For cleaner parameterization, you can use `AlgoSpec` to bundle all settings and create the program via `CollectiveProgram.from_spec(spec)`:

```python
from mscclpp.language.utils import AlgoSpec

spec = AlgoSpec(
    name="my_algo",
    collective=AllReduce(num_ranks=4, chunk_factor=1, inplace=False),
    nranks_per_node=4,
    world_size=4,
    in_place=False,
    instances=2,
    protocol="Simple",
)

with CollectiveProgram.from_spec(spec) as prog:
    ...
```

### Generating the Execution Plan

After defining all operations inside the `CollectiveProgram` block, call `JSON()` to serialize the program into a JSON execution plan:

```python
from mscclpp.language.general import JSON

with CollectiveProgram(...) as prog:
    # ... define operations ...
    print(JSON())
```

The output JSON is consumed by the MSCCL++ executor at runtime.

---

## 2. Ranks

A **Rank** represents a single GPU in the collective. Each rank has its own input buffer, output buffer, and can allocate scratch buffers. Ranks are identified by integer IDs from `0` to `num_ranks - 1`.

```python
from mscclpp.language.rank import Rank

rank0 = Rank(0)
rank1 = Rank(1)
```

### What You Can Do with a Rank

- **Access buffers**: `rank.get_input_buffer()`, `rank.get_output_buffer()`
- **Local data operations**: `rank.copy(...)`, `rank.reduce(...)`
- **Synchronization**: `rank.barrier(...)`
- **Packet operations**: `rank.copy_packets(...)`, `rank.unpack_packets(...)`

Ranks do not communicate directly with each other. Communication happens through **channels**.

---

## 3. Buffers and Chunks

### Buffer Types

The DSL has three buffer types:

| Buffer | Enum | Description |
|--------|------|-------------|
| **Input** | `BufferType.input` | Holds the data each rank contributes to the collective. Size is determined by the collective type. |
| **Output** | `BufferType.output` | Holds the final result of the collective on each rank. Size is determined by the collective type. |
| **Scratch** | `BufferType.scratch` | Temporary working space for intermediate computations. Must be explicitly allocated by the user. |

### Accessing Input/Output Buffers

Input and output buffers are created automatically based on the collective definition:

```python
rank = Rank(0)
input_buf = rank.get_input_buffer()   # Predefined by the collective
output_buf = rank.get_output_buffer() # Predefined by the collective
```

### Allocating Scratch Buffers

Scratch buffers are allocated manually using the `Buffer` class:

```python
from mscclpp.language.rank import Buffer

# Allocate a scratch buffer of size 4 on rank 0
scratch = Buffer(rank=0, size=4)
```

Multiple `Buffer` allocations on the same rank are placed contiguously in scratch memory; each new allocation starts where the previous one ended.

### Chunks: Slicing Buffers

A **Chunk** is a slice of a buffer, representing a contiguous region of data. Chunks are created using Python slice notation on buffers:

```python
input_buf = rank.get_input_buffer()

chunk_0 = input_buf[0:1]   # First chunk (index 0, size 1)
chunk_1 = input_buf[1:3]   # Chunks at index 1-2 (size 2)
all_data = input_buf[:]    # Entire buffer
```

Chunks carry metadata about their rank, buffer type, index, and size. This metadata is used by the DSL to validate operations and track dependencies.

### Buffer Sizes by Collective

| Collective | Input Size (per rank) | Output Size (per rank) |
|------------|----------------------|----------------------|
| `AllGather` | `chunk_factor` | `num_ranks * chunk_factor` |
| `AllReduce` | `num_ranks * chunk_factor` | `num_ranks * chunk_factor` |
| `ReduceScatter` | `num_ranks * chunk_factor` | `chunk_factor` |

---

## 4. Channels

Channels are the communication primitives that connect ranks. They define **how** data moves between GPUs. The DSL provides three channel types, each suited for different hardware topologies and use cases.

> **Important:** Each time you instantiate a channel object, a new channel is created. If you need to reuse a channel for multiple operations, store it in a variable.

### 4.1 MemoryChannel

Uses peer-to-peer GPU memory access (e.g., NVLink, PCIe). Best for **intra-node** communication where GPUs can directly access each other's memory.

```python
from mscclpp.language.channel import MemoryChannel

# Create a channel from rank 0 to rank 1
chan = MemoryChannel(dst_rank=1, src_rank=0)
```

**Supported Operations:**
- `put(dst_chunk, src_chunk, tb=...)` — Write local data to remote memory
- `get(dst_chunk, src_chunk, tb=...)` — Read remote data into local memory
- `signal(tb=..., data_sync=...)` — Send a synchronization signal
- `wait(tb=..., data_sync=...)` — Wait for a synchronization signal
- `reduce(local_src, remote_srcs, tb=...)` — Reduce local and remote data
- `put_packets(...)`, `read_put_packets(...)` — Packet-based transfers

**When to use:** Intra-node GPU-to-GPU communication where direct memory access is available and you need fine-grained control over data movement and synchronization.

### 4.2 PortChannel

Uses interconnection ports for communication. Suitable for **inter-node** or any topology that requires port-based data transfer.

```python
from mscclpp.language.channel import PortChannel

chan = PortChannel(dst_rank=1, src_rank=0)
```

**Supported Operations:**
- `put(dst_chunk, src_chunk, tb=...)` — Write local data to remote memory
- `put_with_signal(dst_chunk, src_chunk, tb=...)` — Put + signal in one operation
- `put_with_signal_and_flush(dst_chunk, src_chunk, tb=...)` — Put + signal + flush
- `signal(tb=..., data_sync=...)` — Send a synchronization signal
- `wait(tb=..., data_sync=...)` — Wait for a synchronization signal
- `flush(tb=..., data_sync=...)` — Force completion of pending operations
- `put_packets(...)`, `read_put_packets(...)` — Packet-based transfers

**When to use:** Inter-node communication (e.g., across InfiniBand), or when you need combined put+signal+flush semantics for guaranteed delivery.

### 4.3 SwitchChannel

Uses switch-based multi-memory access (e.g., NVSwitch with NVLS). Enables collective operations across a group of ranks simultaneously.

```python
from mscclpp.language.channel import SwitchChannel

# Create a switch channel across all 4 ranks, operating on input buffers
switch_chan = SwitchChannel(rank_list=[0, 1, 2, 3], buffer_type=BufferType.input)
```

**Supported Operations (via `at_rank()`):**
- `reduce(buffer_offset, size, dst_chunk, tb=...)` — Hardware-accelerated reduction across all ranks in the group
- `broadcast(src_chunk, buffer_offset, size, tb=...)` — Broadcast data from one rank to all ranks in the group

**When to use:** When hardware supports NVSwitch/NVLS and you want to leverage hardware-accelerated multicast reductions and broadcasts across all GPUs in a node.

```python
# AllReduce via SwitchChannel
for gpu in range(num_gpus):
    rank = Rank(gpu)
    input_buf = rank.get_input_buffer()
    
    # Reduce data at offset `gpu` from all ranks into local chunk
    switch_chan.at_rank(gpu).reduce(
        buffer_offset=gpu, size=1,
        dst_chunk=input_buf[gpu:gpu+1], tb=0
    )
    # Broadcast the reduced result to all ranks
    switch_chan.at_rank(gpu).broadcast(
        src_chunk=input_buf[gpu:gpu+1],
        buffer_offset=gpu, size=1, tb=0
    )
```

### Channel Comparison

| Feature | MemoryChannel | PortChannel | SwitchChannel |
|---------|--------------|-------------|---------------|
| Topology | Intra-node (NVLink, PCIe) | Inter-node (IB, etc.) | NVSwitch/NVLS |
| Point-to-point | Yes | Yes | No (group-based) |
| `get` support | Yes | No | No |
| `flush` support | No | Yes | No |
| `reduce` (remote) | Yes | No | Yes (hardware) |
| `broadcast` | No | No | Yes (hardware) |

---

## 5. Synchronization Operations

Synchronization is critical in multi-GPU programming. The DSL provides several levels of synchronization:

### 5.1 Signal and Wait (Cross-Rank Synchronization)

`signal` and `wait` are the primary cross-rank synchronization primitives. They operate on channels and coordinate execution between different GPUs.

#### Signal

Sends a notification through a channel to the remote rank, indicating that data is ready or an operation is complete.

```python
channel.signal(tb=0, data_sync=SyncType.before)
```

**Use cases:**
- Notify a remote rank that a `put` operation has completed and data is ready to be consumed
- Indicate readiness before a data transfer begins
- Coordinate multi-step algorithms where ranks must proceed in lockstep

#### Wait

Blocks execution on a thread block until a signal is received from the remote rank.

```python
channel.wait(tb=0, data_sync=SyncType.after)
```

**Use cases:**
- Ensure data from a remote `put` has arrived before reading it
- Gate the start of a computation until input data is available
- Prevent races between producers and consumers on different GPUs

#### The `data_sync` Parameter

The `data_sync` parameter (`SyncType`) controls where an intra-thread-block synchronization barrier (equivalent to `__syncthreads()`) is placed relative to the signal/wait operation:

| Value | Behavior | Typical Use |
|-------|----------|-------------|
| `SyncType.none` | No thread-block sync | When no data dependency exists around this operation |
| `SyncType.before` | Sync **before** the operation | Ensure all threads have finished writing data before signaling |
| `SyncType.after` | Sync **after** the operation | Ensure all threads see the wait completion before proceeding |
| `SyncType.both` | Sync both before and after | Maximum safety; use when unsure |

**Best practice pattern — Put with signal/wait:**

```python
# Rank 0 sends data to Rank 1
chan_0_to_1 = MemoryChannel(dst_rank=1, src_rank=0)
chan_1_to_0 = MemoryChannel(dst_rank=0, src_rank=1)

# Initial handshake: both ranks signal readiness
chan_0_to_1.signal(tb=0, data_sync=SyncType.none)
chan_1_to_0.signal(tb=0, data_sync=SyncType.none)
chan_0_to_1.wait(tb=0, data_sync=SyncType.after)
chan_1_to_0.wait(tb=0, data_sync=SyncType.after)

# Rank 0 puts data, then signals completion
chan_0_to_1.put(dst_chunk, src_chunk, tb=0)
chan_0_to_1.signal(tb=0, data_sync=SyncType.before)

# Rank 1 waits for the data
chan_1_to_0.wait(tb=0, data_sync=SyncType.after)
```

#### Relaxed Signal/Wait

MemoryChannel supports a `relaxed` parameter for signal and wait. Relaxed operations use relaxed memory ordering, which can improve performance when strict ordering guarantees are not needed (e.g., during initial handshakes where no data dependency exists yet).

```python
channel.signal(tb=0, relaxed=True)
channel.wait(tb=0, data_sync=SyncType.after, relaxed=True)
```

### 5.2 Flush (PortChannel Only)

`flush` forces all pending operations on a PortChannel to complete. This is necessary for port-based communication where writes may be buffered.

```python
port_channel.flush(tb=0, data_sync=SyncType.after)
```

**Use case:** After a sequence of `put` operations on a PortChannel, call `flush` to guarantee all data has been delivered before proceeding. PortChannel also offers combined operations (`put_with_signal`, `put_with_signal_and_flush`) that merge the data transfer, signaling, and flushing into a single call for convenience and performance.

### 5.3 Barrier (Intra-Rank, Cross-Thread-Block Synchronization)

A `barrier` synchronizes multiple thread blocks **within the same rank**. All specified thread blocks must reach the barrier before any can proceed.

```python
rank = Rank(0)
rank.barrier(tb_list=[0, 1, 2])  # Thread blocks 0, 1, 2 synchronize
```

**Use cases:**
- Ensure all thread blocks on a rank have finished a phase before starting the next
- Coordinate thread blocks that operate on overlapping buffer regions

If only one thread block is specified, the barrier degenerates into a thread-block-level sync (`__syncthreads()`).

### 5.4 Semaphore (Intra-Rank, Asynchronous Synchronization)

Semaphores provide **asynchronous** producer-consumer synchronization between thread blocks on the same rank. Unlike barriers (which require all participants to arrive simultaneously), semaphores allow one thread block to proceed as soon as another has released the semaphore.

```python
from mscclpp.language.rank import Semaphore

sem = Semaphore(rank=0, initial_value=0)

# Thread block 0 produces data, then releases the semaphore
rank.copy(scratch_chunk, input_chunk, tb=0)
sem.release(tb=0, data_sync=SyncType.before)

# Thread block 1 waits for the data, then consumes it
sem.acquire(tb=1, data_sync=SyncType.after)
channel.put(remote_chunk, scratch_chunk, tb=1)
```

#### Semaphore Acquire

Blocks the thread block until the semaphore value is greater than zero, then decrements it.

```python
sem.acquire(tb=1, data_sync=SyncType.after)
```

**Use cases:**
- Wait for a producer thread block to finish writing data before a consumer reads it
- Gate pipeline stages so that stage N+1 does not start until stage N completes

#### Semaphore Release

Increments the semaphore value, potentially unblocking a waiting thread block.

```python
sem.release(tb=0, data_sync=SyncType.before)
```

**Use cases:**
- Signal that a producer has finished writing data to a shared buffer
- Advance a pipeline by allowing the next stage to begin

#### `data_sync` in Semaphore Operations

The same `data_sync` semantics apply:
- On `release`, use `SyncType.before` to ensure all writes in the current thread block are visible before the semaphore is incremented.
- On `acquire`, use `SyncType.after` to ensure the semaphore decrement is complete before any reads in the current thread block proceed.

### 5.5 Automatic Intra-Thread-Block Synchronization

Within a single thread block, the DSL **automatically** tracks data dependencies at the chunk level. If two consecutive operations on the same thread block access the same chunk (e.g., one writes, the next reads), the DSL inserts the necessary synchronization (`nop`) between them. You do not need to manually add sync points for intra-thread-block dependencies.

```python
# The DSL detects that tb=0 writes dst_chunk, then reads it — sync is automatic
rank.copy(dst_chunk, src_chunk, tb=0)
channel.put(remote_dst, dst_chunk, tb=0)  # Automatic sync inserted before this
```

---

## 6. Data Operations

Data operations are the core of every DSL program. They define what computation and data movement happens on each rank. Understanding which operation to use in each scenario is critical for writing correct and performant algorithms.

### 6.1 Local Operations (on Rank)

These operate entirely within a single rank's memory. No channel is needed; they execute locally on the GPU that owns the rank.

#### Copy

Copies data between two local chunks on the same rank. Both the source and destination chunks must belong to the same rank.

```python
rank.copy(dst_chunk, src_chunk, tb=0)
```

**Use case:** Move data between input, output, and scratch buffers on the same GPU.

#### Reduce

Combines data from multiple local chunks using a reduction operation (sum, min, max). All chunks must belong to the same rank and have the same size.

```python
rank.reduce(
    src_chunk=chunk_a,
    other_chunks=[chunk_b, chunk_c],
    tb=0,
    dst_chunk=result_chunk,
    reduce_op=ReduceOperationType.sum
)
```

**Parameters:**
- `src_chunk`: The primary source chunk.
- `other_chunks`: A list of additional chunks to reduce with `src_chunk`. Must contain at least one chunk.
- `dst_chunk` (optional): Where to store the result. If omitted, the result overwrites `src_chunk`.
- `reduce_op`: The reduction operation — `ReduceOperationType.sum` (default), `.min`, or `.max`.
- `packet` (optional): Set to `True` when reducing data in packet format. When enabled, all chunks in `other_chunks` must be scratch buffers.

**Use cases:**
- **Local reduction after gather:** After using `get` or `put` to collect data from multiple peers into separate scratch buffer regions, reduce them locally. This is a common pattern in AllReduce algorithms where each rank gathers partial results from all peers, then reduces them.
- **In-place accumulation:** Reduce incoming data directly into the output buffer as it arrives from different ranks.
- **Packet-based reduction:** In LL protocol algorithms, reduce packet-formatted data directly in scratch buffers using `packet=True`, avoiding an unpack step before reduction.

```python
# Example: Reduce data from 3 peers that was gathered into scratch
rank.reduce(
    src_chunk=scratch[0:1],
    other_chunks=[scratch[1:2], scratch[2:3]],
    tb=0,
    dst_chunk=output_buf[0:1],
    reduce_op=ReduceOperationType.sum
)
```

#### Packet Operations

Packet operations convert data to/from a special packet format used by the LL (Low-Latency) protocol:

- `rank.copy_packets(dst_scratch, src_chunk, tb=0)` — Pack data into packet format (destination must be scratch)
- `rank.unpack_packets(dst_chunk, src_scratch, tb=0)` — Unpack packet data back to normal format (source must be scratch)

> **Important:** Data in packet format can **only** reside in scratch buffers. It can never be stored in input or output buffers. The packet format doubles the data size (each data element is paired with a synchronization flag), and the input/output buffers are sized to hold only the raw data. Always use scratch buffers as the source or destination when working with packet-formatted data.

**Use case:** When using `protocol="LL"`, data must be in packet format for channel transfers. The LL protocol embeds synchronization flags within the data packets themselves, eliminating the need for explicit signal/wait operations. The workflow is: pack data into packets in a local scratch buffer → transfer packets to a remote scratch buffer via `put_packets` or `read_put_packets` → unpack from the remote scratch buffer into the final output buffer.

### 6.2 Remote Operations (on Channels)

These move data between ranks through channels. Each operation type is available on specific channel types and has distinct performance characteristics and use cases.

#### Put

Writes data from the source rank's local memory to the destination rank's memory. Available on **MemoryChannel** and **PortChannel**.

```python
channel.put(dst_chunk, src_chunk, tb=0)
```

**Constraints:**
- `src_chunk` must belong to the channel's source rank
- `dst_chunk` must belong to the channel's destination rank
- Chunk sizes must match

**Use cases by channel type:**

- **MemoryChannel `put`:** Used for intra-node GPU-to-GPU transfers via direct memory access (NVLink, PCIe). The source rank writes directly into the destination rank's memory. This is the most common data transfer primitive. Requires explicit `signal`/`wait` synchronization to notify the receiver that data is available.

  ```python
  # Intra-node: put data then signal completion
  mem_chan = MemoryChannel(dst_rank=1, src_rank=0)
  mem_chan.put(dst_chunk, src_chunk, tb=0)
  mem_chan.signal(tb=0, data_sync=SyncType.before)  # Must signal manually
  ```

- **PortChannel `put`:** Used for inter-node transfers (e.g., InfiniBand). Writes go through the port subsystem, which may buffer them. Requires `flush` to guarantee delivery, or use the combined `put_with_signal` / `put_with_signal_and_flush` variants.

  ```python
  # Inter-node: put data, then signal+flush for guaranteed delivery
  port_chan = PortChannel(dst_rank=1, src_rank=0)
  port_chan.put(dst_chunk, src_chunk, tb=0)
  port_chan.signal(tb=0, data_sync=SyncType.before)
  port_chan.flush(tb=0)  # Ensure data is delivered
  ```

**Common algorithm patterns with `put`:**
- **AllGather:** Each rank puts its chunk into every other rank's output buffer at the appropriate offset.
- **Ring algorithms:** Each rank puts data to its next neighbor in a ring topology, iterating over steps.
- **Scatter:** A root rank puts different chunks to different destination ranks.

#### Get (MemoryChannel only)

Reads data from the remote (destination) rank's memory into the local (source) rank's memory. Only available on **MemoryChannel** because it requires direct memory access.

```python
channel.get(local_dst_chunk, remote_src_chunk, tb=0)
```

**Constraints:**
- `local_dst_chunk` must belong to the channel's source rank (the rank performing the get)
- `remote_src_chunk` must belong to the channel's destination rank (the rank being read from)
- Chunk sizes must match

**Use cases:**
- **Consumer-driven data fetch:** When the consumer knows exactly when it needs data and can pull it, rather than waiting for the producer to push. This can be more efficient when the consumer's timing is the bottleneck.
- **Gather patterns:** A rank collects data from multiple peers by performing `get` on each peer's buffer, useful in AllReduce where each rank gathers partial data from neighbors.
- **Asymmetric algorithms:** When different ranks have different roles (e.g., one rank aggregates data from many), `get` allows the aggregator to pull data on its own schedule.

```python
# Rank 0 pulls data from rank 1's input buffer into its own scratch
mem_chan = MemoryChannel(dst_rank=1, src_rank=0)
mem_chan.get(local_scratch[0:1], remote_input[0:1], tb=0)
```

> **Note:** Unlike `put`, the `get` operation reads from the remote side. The channel's `dst_rank` is the remote rank being read from, and `src_rank` is the local rank performing the read.

#### Remote Reduce (MemoryChannel only)

Reduces local data with data from a remote rank's memory, storing the result locally. Only available on **MemoryChannel** because it requires direct memory access to read remote data.

```python
channel.reduce(
    local_src_chunk=local_chunk,
    remote_src_chunks=[remote_chunk],
    tb=0,
    local_dst_chunk=result_chunk,
    reduce_op=ReduceOperationType.sum
)
```

**Parameters:**
- `local_src_chunk`: A local chunk on the source rank to use as one input to the reduction.
- `remote_src_chunks`: A list of chunks on the destination rank to read and reduce with the local data.
- `local_dst_chunk` (optional): Where to store the result locally. If omitted, overwrites `local_src_chunk`.
- `reduce_op`: The reduction operation (`sum`, `min`, `max`).

**Use cases:**
- **Zero-copy AllReduce:** Instead of first copying remote data into a local buffer and then reducing, `remote reduce` reads and reduces in a single step. This halves the memory traffic compared to a `get` followed by a local `reduce`.
- **ReduceScatter:** Each rank reads and reduces its portion of data from all peers in one operation, directly producing the scattered result.
- **Bandwidth optimization:** When the bottleneck is memory bandwidth, combining the read and reduction into one operation avoids writing intermediate data to memory.

```python
# Reduce rank 1's data with rank 0's local data in one step
mem_chan = MemoryChannel(dst_rank=1, src_rank=0)
mem_chan.reduce(
    local_src_chunk=input_buf[0:1],
    remote_src_chunks=[remote_input[0:1]],
    tb=0,
    local_dst_chunk=output_buf[0:1],
    reduce_op=ReduceOperationType.sum
)
```

#### Combined Operations (PortChannel only)

PortChannel offers combined operations that merge data transfer with synchronization in a single call. These are important for inter-node communication because port-based writes may be buffered and require explicit flushing.

- **`put_with_signal(dst, src, tb)`** — Performs a `put` and automatically sends a `signal` to the destination rank. Equivalent to calling `put()` followed by `signal()`, but executed as a single operation for better performance.

  **Use case:** When you want the receiver to know data has been sent but don't need guaranteed delivery (e.g., the receiver will do its own synchronization before reading).

- **`put_with_signal_and_flush(dst, src, tb)`** — Performs a `put`, sends a `signal`, and flushes the channel. This is the strongest guarantee: when the receiver's `wait` completes, the data is guaranteed to be fully written.

  **Use case:** The go-to operation for inter-node data transfers where correctness requires the data to be fully delivered before the receiver proceeds. Most inter-node algorithms use this as their primary transfer primitive.

```python
# Inter-node send with guaranteed delivery
port_chan = PortChannel(dst_rank=1, src_rank=0)
port_chan.put_with_signal_and_flush(dst_chunk, src_chunk, tb=0)

# On the receiving side (rank 1 → rank 0 channel)
recv_chan.wait(tb=0, data_sync=SyncType.after)  # Data is guaranteed to be there
```

#### Packet Transfer Operations

Packet transfer operations are used with the LL (Low-Latency) protocol. They encode data into a packet format that embeds synchronization flags alongside the payload, enabling self-validating transfers without explicit signal/wait. Available on both **MemoryChannel** and **PortChannel**.

- **`channel.put_packets(dst_scratch, src, tb)`** — Reads data from a local buffer (any type), converts it to packet format, and writes it to the remote rank's scratch buffer. The destination must be a scratch buffer.

  **Use case:** Sending data from an input or output buffer to a remote rank's scratch buffer in packet format. This is the initial step in LL-protocol communication: the sender packs and sends in one operation.

- **`channel.read_put_packets(dst_scratch, src_scratch, tb)`** — Reads packet-formatted data from a local scratch buffer, validates the embedded synchronization flag to ensure the data is ready (i.e., has been fully written by the sender), and writes it to a remote rank's scratch buffer preserving packet format. Both source and destination must be scratch buffers. The flag check acts as an implicit synchronization: the operation will only forward data once the packet's flag confirms it has been written by the previous sender, eliminating the need for explicit signal/wait between hops.

  **Use case:** Forwarding packet data that has already been received in scratch to another peer's scratch buffer. Common in multi-hop or ring algorithms using LL protocol, where data is relayed through intermediate ranks without unpacking. Because the flag validation is built into the read, each hop is self-synchronizing.

```python
# LL protocol workflow:
# 1. Pack and send input data to remote scratch
chan.put_packets(remote_scratch[0:1], local_input[0:1], tb=0)

# 2. Forward packet data from local scratch to another peer's scratch
chan2.read_put_packets(peer2_scratch[0:1], local_scratch[0:1], tb=0)

# 3. On the receiver: unpack from scratch to output
rank.unpack_packets(output_buf[0:1], local_scratch[0:1], tb=0)
```

> **Remember:** Packet-formatted data can only exist in scratch buffers. The packet format doubles the data size (each element is paired with a flag), so input/output buffers cannot hold it.

### 6.3 Switch Operations (SwitchChannel)

SwitchChannel provides hardware-accelerated group operations that leverage NVSwitch/NVLS capabilities. Unlike point-to-point channels, SwitchChannel operates on a group of ranks simultaneously, making it the most efficient option for collective operations when the hardware supports it.

#### Group Reduce

Reduces data from the same buffer offset across **all ranks** in the group, storing the result in a local destination chunk. The hardware reads data from each rank's buffer at the specified offset and combines them using the reduction operation.

```python
switch_chan.at_rank(gpu).reduce(
    buffer_offset=0, size=1,
    dst_chunk=local_chunk, tb=0,
    reduce_op=ReduceOperationType.sum
)
```

**Parameters:**
- `buffer_offset`: The starting offset in the buffer (specified by `buffer_type` at channel creation) from which each rank's data is read.
- `size`: Number of chunks to reduce.
- `dst_chunk`: Local chunk where the reduced result is stored.
- `reduce_op`: Reduction operation (default: `sum`).

**Use cases:**
- **AllReduce first phase:** Reduce data from all ranks into a local buffer. Each rank reduces a different chunk offset, so collectively all chunks are reduced.
- **Zero-copy reduction:** When using scratch buffers with SwitchChannel, the hardware reads directly from each rank's scratch at the given offset — no data copying needed before the reduction.

#### Group Broadcast

Broadcasts data from one rank's source chunk to all ranks in the group, writing to the specified buffer offset on every rank.

```python
switch_chan.at_rank(gpu).broadcast(
    src_chunk=local_chunk,
    buffer_offset=0, size=1, tb=0
)
```

**Parameters:**
- `src_chunk`: The local chunk containing the data to broadcast.
- `buffer_offset`: The destination offset in the buffer on every rank where the data will be written.
- `size`: Number of chunks to broadcast.

**Use cases:**
- **AllReduce second phase:** After reduction, each rank broadcasts its reduced chunk to all other ranks so every rank ends up with the full reduced result.
- **Initialization/configuration:** Broadcast configuration data or initial state from one rank to all others.

**Typical AllReduce pattern with SwitchChannel:**

```python
# Each rank reduces a different chunk, then broadcasts it to all
switch_chan = SwitchChannel(rank_list=list(range(N)), buffer_type=BufferType.input)
for gpu in range(N):
    buf = Rank(gpu).get_input_buffer()
    # Phase 1: Each rank reduces one chunk from all ranks
    switch_chan.at_rank(gpu).reduce(
        buffer_offset=gpu, size=1,
        dst_chunk=buf[gpu:gpu+1], tb=0
    )
    # Phase 2: Each rank broadcasts its reduced chunk to all ranks
    switch_chan.at_rank(gpu).broadcast(
        src_chunk=buf[gpu:gpu+1],
        buffer_offset=gpu, size=1, tb=0
    )
```

---

## 7. Thread Blocks and Thread Block Groups

### Thread Blocks

Every operation is assigned to a **thread block** via the `tb` parameter. Thread blocks are the unit of parallel execution on a GPU. Operations assigned to the same thread block execute sequentially (with automatic dependency tracking), while operations on different thread blocks can execute concurrently.

```python
# These two operations run on different thread blocks and can overlap
rank.copy(chunk_a, chunk_b, tb=0)
channel.put(remote_chunk, chunk_c, tb=1)
```

**Guidelines:**
- Use a single thread block (`tb=0`) for simple algorithms
- Use multiple thread blocks to overlap computation and communication
- Use semaphores or barriers to synchronize across thread blocks when needed

### Thread Block Groups

`ThreadBlockGroup` allows multiple thread blocks to cooperatively execute a single operation. This is useful for large data transfers that benefit from more parallelism.

```python
from mscclpp.language.thread_block_group import ThreadBlockGroup

tbg = ThreadBlockGroup(tb_list=[0, 1, 2, 3])
rank.copy(output_buf[0:1], input_buf[0:1], tb_group=tbg)
```

Each thread block in the group handles a portion of the work. The group is identified internally, and operations using `tb_group` automatically distribute work across the constituent thread blocks.

Thread Block Groups are also useful for creating groups of different sizes for different roles in the algorithm. For example, you might allocate a larger group (more thread blocks) for a bandwidth-intensive data transfer phase and a smaller group for a latency-sensitive signaling phase. This allows non-uniform distribution of GPU resources across different operations, tuning the parallelism level to match each operation's needs.

```python
# Large group for heavy data movement
tbg_data = ThreadBlockGroup(tb_list=[0, 1, 2, 3, 4, 5])
rank.copy(output_buf[0:1], input_buf[0:1], tb_group=tbg_data)

# Smaller group for lightweight operations
tbg_sync = ThreadBlockGroup(tb_list=[6, 7])
channel.put(remote_chunk, local_chunk, tb_group=tbg_sync)
```

> **Note:** Thread Block Groups are currently a prototype feature.

---

## 8. Pipeline Loops

Pipeline loops enable **overlapping** execution of operations across iterations, maximizing throughput for large data transfers. They are essential for algorithms that process data in chunks and can benefit from hiding latency.

An additional benefit of pipeline loops is that they significantly **reduce the size of the generated JSON execution plan**. Without a pipeline loop, an algorithm that processes N chunks would require N copies of every operation in the JSON file — one per chunk. With a pipeline loop, the operations are defined once and the executor repeats them across chunks at runtime. For algorithms with many chunks, this can reduce the JSON file size by orders of magnitude, making it faster to load and parse.

### How It Works

A `LoopIterationContext` defines a pipelined processing loop. Operations defined inside the context are executed repeatedly, with each iteration processing a `unit`-sized chunk of data. The `num_chunks` parameter specifies how many chunks are processed per iteration.

```python
from mscclpp.language.loop import LoopIterationContext

with LoopIterationContext(unit=2**20, num_chunks=1):
    # Operations here are pipelined across iterations
    rank.copy(scratch_chunk, input_chunk, tb=0)
    sem.release(tb=0, data_sync=SyncType.before)

    sem.acquire(tb=1, data_sync=SyncType.after)
    channel.put(remote_chunk, scratch_chunk, tb=1)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `unit` | The granularity of each pipeline iteration in bytes (e.g., `2**20` = 1 MB). |
| `num_chunks` | Number of chunks processed per iteration. Chunk sizes in operations inside the loop should match this value. |

### Use Case: Pipelined AllReduce

A typical pipelined pattern uses semaphores to synchronize stages:

1. **Stage 1 (tb=0):** Copy input data to scratch buffer
2. **Stage 2 (tb=1):** Perform reduction and cross-rank communication
3. **Stage 3 (tb=2):** Copy results to output buffer

Each stage runs on a different thread block. As soon as stage 1 finishes one chunk, it releases a semaphore so stage 2 can start on that chunk — while stage 1 begins the next chunk. This overlapping hides latency and keeps the GPU busy.

```python
sem0 = Semaphore(rank=0, initial_value=0)
sem1 = Semaphore(rank=0, initial_value=0)

with LoopIterationContext(unit=2**20, num_chunks=1):
    # Stage 1: Copy input → scratch (tb=0)
    rank.copy(scratch_chunk, input_chunk, tb=0)
    sem0.release(tb=0, data_sync=SyncType.before)

    # Stage 2: Reduce + communicate (tb=1)
    sem0.acquire(tb=1, data_sync=SyncType.after)
    channel.put(remote_chunk, scratch_chunk, tb=1)
    channel.signal(tb=1, data_sync=SyncType.before)
    sem1.release(tb=1)

    # Stage 3: Copy scratch → output (tb=2)
    sem1.acquire(tb=2, data_sync=SyncType.after)
    channel.wait(tb=2, data_sync=SyncType.after)
    rank.copy(output_chunk, scratch_chunk, tb=2)
```

---

## 9. Collectives

The DSL provides built-in collective definitions that determine buffer layouts:

### AllGather

Each rank starts with a unique block of data; by the end, every rank holds the concatenation of all data blocks.

```python
collective = AllGather(num_ranks=4, chunk_factor=1, inplace=True)
```

- **Input:** `chunk_factor` chunks per rank
- **Output:** `num_ranks * chunk_factor` chunks per rank
- **In-place:** When `True`, the input buffer is a slice of the output buffer

### AllReduce

Combines data from all ranks using a reduction operation, then distributes the result to all ranks.

```python
collective = AllReduce(num_ranks=4, chunk_factor=1, inplace=False)
```

- **Input:** `num_ranks * chunk_factor` chunks per rank
- **Output:** `num_ranks * chunk_factor` chunks per rank

### ReduceScatter

Reduces data across all ranks and scatters the result so each rank gets a unique portion.

```python
collective = ReduceScatter(num_ranks=4, chunk_factor=1, inplace=False)
```

- **Input:** `num_ranks * chunk_factor` chunks per rank
- **Output:** `chunk_factor` chunks per rank

### Custom Collectives

For testing or non-standard communication patterns, use `TestCollective`. Unlike the built-in collectives (`AllGather`, `AllReduce`, `ReduceScatter`), where buffer sizes are derived from `num_ranks` and `chunk_factor` according to fixed formulas, `TestCollective` lets you define the exact input and output buffer sizes per rank with complete freedom. This is useful when prototyping new algorithms, experimenting with non-standard buffer layouts, or testing DSL features in isolation.

```python
# Define arbitrary buffer sizes: 8 input chunks and 8 output chunks per rank
collective = TestCollective(num_ranks=4, input_size=8, output_size=8)
```

**Use cases:**
- **Algorithm prototyping:** When designing a new communication pattern that doesn't fit the AllGather/AllReduce/ReduceScatter mold, use `TestCollective` to set buffer shapes freely.
- **Unit testing:** Create minimal buffer configurations for testing specific DSL operations without the overhead of a full collective definition.
- **SendRecv / custom patterns:** For point-to-point or irregular communication patterns where each rank may have different data roles.

---

## 10. Operation Fusion

When `instr_fusion=True` (the default), the DSL automatically detects and fuses compatible operations within each thread block. This is a post-processing optimization that happens when `JSON()` is called. Operation fusion is one of the most important optimizations the DSL performs, as it directly impacts execution performance by reducing memory traffic, eliminating redundant synchronization, and decreasing the total number of GPU instructions.

### How Fusion Works

The fusion engine operates in two phases:

**Phase 1 — `data_sync` expansion:** Before fusion, the DSL expands every `data_sync` flag into explicit `nop` (sync) operations. For example, an operation with `data_sync=SyncType.before` gets a `nop` inserted before it; `data_sync=SyncType.after` gets a `nop` after it; `data_sync=SyncType.both` gets nops on both sides. This expansion also applies recursively inside pipeline operations. After expansion, the `data_sync` flags are consumed and synchronization becomes explicit `nop` operations in the instruction stream.

**Phase 2 — Pairwise fusion scan:** The DSL scans consecutive operation pairs within each thread block. For each pair `(A, B)`, it calls `A.__add__(B)` to check if the two can be fused. If fusion succeeds, the pair is replaced by the fused operation and the scan continues. If fusion fails but `A` can still optimize `B`'s flags (e.g., stripping a redundant `data_sync`), it does so as a side effect.

Two operations can be fused when:

1. **Compatible types:** The operation types have a defined fusion rule (not all combinations are fusable — see below for the complete list).
2. **Matching constraints:** Depending on the operation type, constraints like same channel type, same chunk size, non-overlapping channel IDs, or matching buffer references must be satisfied.
3. **No intervening dependencies:** No other operation between them accesses the same memory region (enforced by the DAG-based dependency analysis that orders operations before fusion runs).

### Complete Fusion Rules

The following sections document every fusion rule in the DSL, derived directly from the `__add__` methods of each operation class. Each rule lists the two operands, the conditions that must hold, and the resulting fused operation.

#### Synchronization Fusion

These rules eliminate redundant synchronization overhead by merging sync operations or removing unnecessary sync points.

**Sync (nop) fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `nop` (sync) | `nop` (sync) | — | Single `nop` |
| `nop` (sync) | `barrier` | — | `barrier` (the nop is absorbed) |
| `nop` (sync) | `pipeline` | Pipeline's first operation has `before` sync | `pipeline` (the nop is absorbed) |
| `nop` (sync) | Any `data_sync` op | — | **Side effect:** strips `before` from B's `data_sync` (the nop already provides the sync) |

**Barrier fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `barrier` | `nop` (sync) | — | `barrier` (the nop is absorbed) |
| `barrier` | Any `data_sync` op | — | **Side effect:** strips `before` from B's `data_sync` |

**Signal fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `signal` | `signal` | Same channel type, same relaxed mode, **non-overlapping** channel IDs | Single `signal` with merged channel IDs and merged `data_sync` |
| `relaxed_signal` | `relaxed_signal` | Same channel type, **non-overlapping** channel IDs | Single `relaxed_signal` with merged channel IDs |
| `signal` / `relaxed_signal` | Any `data_sync` op with `before`, or `nop`, or `barrier`, or `pipeline` with `before` | — | **Side effect:** strips `after` from A's `data_sync` (the following sync makes it redundant) |

**Wait fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `wait` | `wait` | Same relaxed mode, same channel type, **non-overlapping** channel IDs | Single `wait` with merged channel IDs and merged `data_sync` |
| `relaxed_wait` | `relaxed_wait` | Same channel type, **non-overlapping** channel IDs | Single `relaxed_wait` with merged channel IDs |
| `wait` / `relaxed_wait` | Any `data_sync` op with `before`, or `nop`, or `barrier`, or `pipeline` with `before` | — | **Side effect:** strips `after` from A's `data_sync` |

**Flush fusion (PortChannel only):**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `flush` | `flush` | Same channel type | Single `flush` with merged channel IDs and merged `data_sync` |
| `flush` | Any `data_sync` op with `before`, or `nop`, or `barrier`, or `pipeline` with `before` | — | **Side effect:** strips `after` from A's `data_sync` |

**Semaphore fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `sem_acquire` | `sem_acquire` | — | Single `sem_acquire` with merged semaphore IDs and merged `data_sync` |
| `sem_release` | `sem_release` | — | Single `sem_release` with merged semaphore IDs and merged `data_sync` |
| `sem_acquire` / `sem_release` | Any `data_sync` op with `before`, or `nop`, or `barrier`, or `pipeline` with `before` | — | **Side effect:** strips `after` from A's `data_sync` |

**Pipeline fusion:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `pipeline` | `nop` (sync) | Pipeline's last operation has `after` sync | `pipeline` (the nop is absorbed) |
| `pipeline` | Any `data_sync` op | Pipeline's last operation has `after` sync | **Side effect:** strips `before` from B's `data_sync` |

#### Data Transfer Fusion

These rules merge multiple data transfer operations into a single operation that handles multiple source/destination pairs, reducing the number of GPU instructions.

**Get fusion (MemoryChannel):**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `get` | `get` | Same chunk size, same channel type, same `ThreadBlockGroup` | Single `get` with merged src/dst buffer lists and channel IDs |

This is useful when a rank pulls data from multiple peers — instead of N separate get instructions, the GPU executes one fused get.

**Put fusion (MemoryChannel and PortChannel):**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `put` | `put` | Same chunk size, same channel type, same `ThreadBlockGroup` | Single `put` with merged src/dst buffer lists and channel IDs |
| `put_packet` | `put_packet` | Same chunk size, same channel type, same `ThreadBlockGroup` | Single `put_packet` with merged buffers |
| `put_with_signal` | `put_with_signal` | Same chunk size, same channel type, same `ThreadBlockGroup` | Single `put_with_signal` with merged buffers |
| `put_with_signal_and_flush` | `put_with_signal_and_flush` | Same chunk size, same channel type, same `ThreadBlockGroup` | Single `put_with_signal_and_flush` with merged buffers |
| `read_put_packet` | `read_put_packet` | Same **source buffer**, same channel type, same `ThreadBlockGroup` | Single `read_put_packet` with merged destination buffers (reads the same source once, writes to multiple destinations) |

> **Note:** Put fusion requires that both operations use the **same instruction type**. You cannot fuse a `put` with a `put_with_signal`, for example.

#### Reduce Fusion

Reduce fusion is where the DSL achieves its most significant performance gains. These rules cover both merging multiple reductions and chaining reductions with subsequent data transfers.

**Reduce merging (same-type accumulation):**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `reduce` | `reduce` | Same `local_src_buff[0]`, same `local_dst_buff`, same channel type, same reduce op, same `ThreadBlockGroup` | Single `reduce` with merged source buffers |
| `reduce_packet` | `reduce_packet` | Same constraints as above | Single `reduce_packet` with merged source buffers |
| `read_reduce` | `read_reduce` | Same constraints as above | Single `read_reduce` with merged remote source buffers and channel IDs |

This handles the common pattern of reducing data from multiple peers: instead of reducing peer 0's data, then reducing peer 1's data on top, the fused operation reads and reduces all peers' data in a single pass.

**Reduce + Put → Reduce-Send (cross-operation chaining):**

These are the most impactful fusion rules. They eliminate the intermediate memory write between a reduction and a subsequent data transfer.

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `reduce` + `put` | — | A's `local_dst_buff[0]` == B's `src_buff[0]`, B's channel type is `memory`, same `ThreadBlockGroup` | `reduce_send` |
| `reduce_send` + `put` | — | Same constraints | `reduce_send` with additional remote destination (sends to multiple peers) |
| `read_reduce` + `put` | — | Same constraints | `read_reduce_send` |
| `read_reduce_send` + `put` | — | Same constraints | `read_reduce_send` with additional remote destination |

**Reduce + Put for packet operations:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `reduce_packet` + `put_packet` | — | A's `local_dst_buff[0]` == B's `src_buff[0]`, B's channel type is `memory`, same `ThreadBlockGroup` | `reduce_send_packet` |
| `reduce_send_packet` + `put_packet` | — | Same constraints | `reduce_send_packet` with additional remote destination |

**Reduce + Copy Packet → Reduce-Copy-Packet:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `reduce_packet` + `copy_packet` | — | A's `local_dst_buff[0]` == B's `src_buff[0]`, same `ThreadBlockGroup` | `reduce_copy_packet` (reduces and packs into packet format in one step) |

**Reduce-Copy-Packet + Put → Reduce-Copy-Send-Packet:**

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `reduce_copy_packet` + `put_packet` | — | A's `local_dst_buff[0]` == B's `src_buff[0]`, B's channel type is `memory`, same `ThreadBlockGroup` | `reduce_copy_send_packet` |
| `reduce_copy_packet` + `read_put_packet` | — | A's `local_pkt_dst_buff[0]` == B's `src_buff[0]`, B's channel type is `memory`, same `ThreadBlockGroup` | `reduce_copy_send_packet` |
| `reduce_copy_send_packet` + `put_packet` | — | Same constraints as above | `reduce_copy_send_packet` with additional remote destination |
| `reduce_copy_send_packet` + `read_put_packet` | — | Same constraints as above | `reduce_copy_send_packet` with additional remote destination |

This chain represents the most aggressive fusion in the DSL: `reduce → copy_packet → put` becomes a single `reduce_copy_send_packet` instruction that reads multiple inputs, reduces them, packs the result into packet format, and sends it to one or more remote peers — all without any intermediate memory writes.

#### Switch Channel Fusion

| A | B | Conditions | Result |
|---|---|-----------|--------|
| `group_load_reduce` + `group_store` | — | Same buffer type, same size, A's `dst_chunk` == B's `src_chunk`, same channel IDs, same channel type | `group_load_reduce_store` (reduces and broadcasts in a single hardware operation) |

This fuses the two phases of a SwitchChannel AllReduce (reduce + broadcast) into a single operation when the reduced result is immediately broadcast to the same group.

### `data_sync` Optimization in Detail

The fusion engine doesn't just merge operations — it also optimizes synchronization flags as a side effect of the fusion scan. This happens when two operations can't be fully fused but one can infer that a sync point in the other is redundant.

The key rules are:

1. **Strip redundant `before` sync:** When operation A is a `nop`, `barrier`, or any operation that already provides a sync point (like a `pipeline` ending with sync), the following operation B doesn't need a `before` sync — A already guarantees thread-block synchronization. So B's `data_sync` has `before` stripped.

2. **Strip redundant `after` sync:** When operation B provides a `before` sync (or is a `nop`/`barrier`), the preceding operation A doesn't need an `after` sync. So A's `data_sync` has `after` stripped.

3. **Cascading elimination:** These optimizations cascade. Consider three operations: `signal(after)` → `nop` → `wait(before)`. First, the signal's `after` is stripped because the `nop` follows. Then the `nop` is absorbed by the `wait`'s `before`. The result is `signal(none)` → `wait(none)` with no sync between them — because the signal's own sync and the wait's own sync are at the same point.

This cascading behavior can eliminate a significant number of `__syncthreads()` calls in complex algorithms with many synchronization operations.

### Fusion Chain Example

To illustrate how fusion chains work in practice, consider this AllReduce pattern:

```python
# Step 1: Reduce data from a remote peer into local scratch
channel.reduce(local_input[0:1], [remote_input[0:1]], tb=0, local_dst_chunk=scratch[0:1])

# Step 2: Pack the reduced result into packet format
rank.copy_packets(pkt_scratch[0:1], scratch[0:1], tb=0)

# Step 3: Send packet data to peer's scratch buffer
channel_out.put_packets(remote_pkt_scratch[0:1], pkt_scratch[0:1], tb=0)

# Step 4: Send packet data to another peer
channel_out2.put_packets(remote_pkt_scratch2[0:1], pkt_scratch[0:1], tb=0)

# Step 5: Signal completion
channel_out.signal(tb=0, data_sync=SyncType.before)
channel_out2.signal(tb=0, data_sync=SyncType.before)
```

The fusion engine processes this chain step by step:

1. `read_reduce` + `copy_packet` → `reduce_copy_packet` (Step 1 + Step 2)
2. `reduce_copy_packet` + `put_packet` → `reduce_copy_send_packet` (fused + Step 3)
3. `reduce_copy_send_packet` + `put_packet` → `reduce_copy_send_packet` with 2 destinations (fused + Step 4)
4. `signal` + `signal` → single `signal` with merged channel IDs (Step 5a + Step 5b)

The final result is just **two instructions**: one `reduce_copy_send_packet` (that reads, reduces, packs, and sends to two peers) and one `signal` (that notifies both peers). The original six operations are reduced to two, eliminating four intermediate memory round-trips.

### Practical Impact

The effect of fusion depends on the algorithm:

- **Simple algorithms** (e.g., AllGather with direct puts): Fusion mainly merges multiple `signal` and `wait` operations, reducing sync overhead.
- **Reduction-heavy algorithms** (e.g., AllReduce, ReduceScatter): The `reduce` → `put` → `reduce_send` fusion eliminates intermediate writes, which can nearly double effective bandwidth.
- **LL-protocol algorithms** (e.g., packet-based AllReduce): The full `reduce_copy_send_packet` chain can fuse 4-5 logical operations into one, dramatically reducing instruction count and memory traffic.
- **SwitchChannel algorithms**: The `group_load_reduce` + `group_store` → `group_load_reduce_store` fusion lets the hardware perform both phases of AllReduce in a single pass.

### When to Disable Fusion

In rare cases, you may want to disable fusion by setting `instr_fusion=False`:
- **Debugging:** To see the exact operations as you wrote them in the generated JSON, making it easier to trace issues.
- **Correctness verification:** To compare fused vs. unfused execution and verify that fusion doesn't change behavior.
- **Performance analysis:** To understand the baseline cost of operations before fusion optimizations.

Fusion is always semantically transparent — the fused program produces identical results to the unfused version.

---

## 11. Instances and Replication

Setting `instances > 1` replicates the entire algorithm multiple times, with each instance operating on a separate data partition. This is a key mechanism for scaling algorithm throughput without rewriting the algorithm logic.

```python
with CollectiveProgram(
    ...,
    instances=4,
    replication_policy=ReplicationPolicy.interleaved,
):
    ...
```

### How Replication Works

When the DSL replicates an algorithm:

1. **Thread blocks are duplicated:** Each instance gets its own set of thread blocks. If instance 0 uses thread blocks `[0, 1]`, instance 1 gets `[2, 3]`, instance 2 gets `[4, 5]`, and so on.
2. **Channels are duplicated:** Each instance creates its own channels (unless `reuse_resources=True`), so instances don't interfere with each other's synchronization state.
3. **Buffer offsets are shifted:** Each instance operates on a different portion of the data. The `replication_policy` controls how chunks are assigned to instances.
4. **Semaphore and barrier IDs are shifted:** Each instance gets unique semaphore and barrier IDs to avoid cross-instance synchronization conflicts.

### Replication Policies

- **`ReplicationPolicy.interleaved` (default):** Data chunks are distributed across instances in round-robin fashion. With 4 chunks and 2 instances: instance 0 handles chunks 0, 2; instance 1 handles chunks 1, 3. This provides good load balancing and cache behavior.
- **`ReplicationPolicy.none`:** No automatic chunk redistribution. Instances operate on contiguous blocks. With 4 chunks and 2 instances: instance 0 handles chunks 0, 1; instance 1 handles chunks 2, 3.

### When to Increase Instances

- **Large message sizes:** A single instance may not generate enough thread blocks to fully utilize the GPU's streaming multiprocessors. More instances = more thread blocks = more parallelism.
- **Pipeline depth:** More instances can increase the effective pipeline depth, overlapping communication and computation across instances.
- **Bandwidth saturation:** For algorithms that are limited by link bandwidth, multiple instances can help keep the interconnect busy by having multiple transfers in flight.

### Interaction with `reuse_resources`

When `reuse_resources=True`, instances share channel objects instead of creating separate ones. This reduces setup overhead (fewer channels to negotiate) but means instances must be careful not to have conflicting synchronization state. This is safe when channels only access scratch buffers (since each instance operates on different scratch offsets) and is commonly used with LL-protocol algorithms.

### Example

```python
# Single instance: 1 thread block pair, processes all data
with CollectiveProgram("algo", collective, 4, instances=1):
    # tb=0 on each rank
    ...

# 4 instances: 4 thread block pairs, each processes 1/4 of the data
with CollectiveProgram("algo", collective, 4, instances=4):
    # Same algorithm code — the DSL replicates it automatically
    # Instance 0: tb=0, Instance 1: tb=1, Instance 2: tb=2, Instance 3: tb=3
    ...
```

---

## 12. Complete Example: 2-GPU AllGather

Here's a complete, annotated example that ties together all the core concepts:

```python
from mscclpp.language.channel import MemoryChannel
from mscclpp.language.rank import Rank
from mscclpp.language.general import JSON
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.collectives import AllGather
from mscclpp.language.internal.types import SyncType

def allgather_2gpu():
    num_gpus = 2
    collective = AllGather(num_gpus, chunk_factor=1, inplace=True)

    with CollectiveProgram("allgather_2gpu", collective, num_gpus, instances=4):
        # Create rank objects
        rank0 = Rank(0)
        rank1 = Rank(1)

        # Access buffers
        r0_input = rank0.get_input_buffer()
        r0_output = rank0.get_output_buffer()
        r1_input = rank1.get_input_buffer()
        r1_output = rank1.get_output_buffer()

        # Create bidirectional channels
        chan_0_to_1 = MemoryChannel(dst_rank=1, src_rank=0)
        chan_1_to_0 = MemoryChannel(dst_rank=0, src_rank=1)

        # --- Phase 1: Initial Handshake ---
        # Both ranks signal readiness (no data dependency yet, so no sync needed)
        chan_0_to_1.signal(tb=0, data_sync=SyncType.none)
        chan_1_to_0.signal(tb=0, data_sync=SyncType.none)

        # Both ranks wait until the other is ready
        chan_0_to_1.wait(tb=0, data_sync=SyncType.after)
        chan_1_to_0.wait(tb=0, data_sync=SyncType.after)

        # --- Phase 2: Data Exchange ---
        # Rank 0 sends its input to rank 1's output buffer (slot 0)
        chan_0_to_1.put(r1_output[0:1], r0_input[0:1], tb=0)

        # Rank 1 sends its input to rank 0's output buffer (slot 1)
        chan_1_to_0.put(r0_output[1:2], r1_input[0:1], tb=0)

        # --- Phase 3: Completion Synchronization ---
        # Signal that put operations are done (sync before to flush writes)
        chan_0_to_1.signal(tb=0, data_sync=SyncType.before)
        chan_1_to_0.signal(tb=0, data_sync=SyncType.before)

        # Wait for remote puts to complete (sync after to read safely)
        chan_0_to_1.wait(tb=0, data_sync=SyncType.after)
        chan_1_to_0.wait(tb=0, data_sync=SyncType.after)

        print(JSON())

allgather_2gpu()
```

### Running the Example

```bash
# Generate the execution plan
python allgather_2gpu.py > allgather_2gpu.json

# Execute with 2 GPUs
mpirun --allow-run-as-root -np 2 python3 python/test/executor_test.py \
    -path allgather_2gpu.json --size 1M --in_place
```

---

## 13. Summary of Synchronization Primitives

| Primitive | Scope | Blocking | Use Case |
|-----------|-------|----------|----------|
| `channel.signal()` | Cross-rank | No | Notify remote rank that data/operation is ready |
| `channel.wait()` | Cross-rank | Yes | Wait for remote rank's signal |
| `channel.flush()` | Cross-rank (PortChannel) | Yes | Force completion of buffered writes |
| `rank.barrier()` | Intra-rank, cross-TB | Yes | Synchronize multiple thread blocks |
| `sem.acquire()` | Intra-rank, cross-TB | Yes | Wait for producer to release (consumer side) |
| `sem.release()` | Intra-rank, cross-TB | No | Signal that production is done (producer side) |
| `data_sync` | Intra-TB | Yes | Insert `__syncthreads()` before/after an operation |
| *(automatic)* | Intra-TB | Yes | DSL auto-inserts sync for data dependencies |

---

## 14. Quick Reference: Common Patterns

### Pattern 1: Simple Point-to-Point Transfer

```python
chan = MemoryChannel(dst_rank=1, src_rank=0)
chan.signal(tb=0, data_sync=SyncType.none)       # Handshake
chan.wait(tb=0, data_sync=SyncType.after)
chan.put(dst_chunk, src_chunk, tb=0)               # Transfer
chan.signal(tb=0, data_sync=SyncType.before)       # Notify completion
```

### Pattern 2: Bidirectional Exchange

```python
chan_fwd = MemoryChannel(dst_rank=1, src_rank=0)
chan_bwd = MemoryChannel(dst_rank=0, src_rank=1)
# Handshake
chan_fwd.signal(tb=0, data_sync=SyncType.none)
chan_bwd.signal(tb=0, data_sync=SyncType.none)
chan_fwd.wait(tb=0, data_sync=SyncType.after)
chan_bwd.wait(tb=0, data_sync=SyncType.after)
# Exchange
chan_fwd.put(r1_chunk, r0_chunk, tb=0)
chan_bwd.put(r0_chunk, r1_chunk, tb=0)
# Completion
chan_fwd.signal(tb=0, data_sync=SyncType.before)
chan_bwd.signal(tb=0, data_sync=SyncType.before)
chan_fwd.wait(tb=0, data_sync=SyncType.after)
chan_bwd.wait(tb=0, data_sync=SyncType.after)
```

### Pattern 3: Producer-Consumer Pipeline (Semaphore)

```python
sem = Semaphore(rank=0, initial_value=0)
rank.copy(scratch, input_data, tb=0)           # Producer
sem.release(tb=0, data_sync=SyncType.before)   # Signal consumer

sem.acquire(tb=1, data_sync=SyncType.after)    # Consumer waits
channel.put(remote, scratch, tb=1)             # Consumer uses data
```

### Pattern 4: Ring AllGather

```python
for step in range(num_ranks - 1):
    for r in range(num_ranks):
        next_r = (r + 1) % num_ranks
        offset = (r - step) % num_ranks
        src_chunk = Rank(r).get_output_buffer()[offset:offset+1]
        dst_chunk = Rank(next_r).get_output_buffer()[offset:offset+1]
        channels[next_r, r].put(dst_chunk, src_chunk, tb=0)
        channels[next_r, r].signal(tb=0)
        channels[(r-1) % num_ranks, r].wait(tb=0)
```

### Pattern 5: NVSwitch AllReduce

```python
switch_chan = SwitchChannel(rank_list=list(range(N)), buffer_type=BufferType.input)
for gpu in range(N):
    buf = Rank(gpu).get_input_buffer()
    switch_chan.at_rank(gpu).reduce(buffer_offset=gpu, size=1, dst_chunk=buf[gpu:gpu+1], tb=0)
    switch_chan.at_rank(gpu).broadcast(src_chunk=buf[gpu:gpu+1], buffer_offset=gpu, size=1, tb=0)
```
