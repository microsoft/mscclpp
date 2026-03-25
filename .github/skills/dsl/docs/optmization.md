# MSCCL++ DSL Optimization Guide

## 1. Instruction Fusion

The MSCCL++ executor fuses consecutive operations of the same type within a thread block into a single GPU instruction. This dramatically reduces kernel overhead.

### Signal Fusion

When a thread block needs to signal multiple peers, **group all signals together** before any waits. The executor fuses consecutive `signal` calls into one instruction.

**Bad — alternating signal and wait (no fusion possible):**

```python
# Each signal/wait pair is a separate instruction = 6 instructions
ch_to_rank1.signal(tb=0)
ch_from_rank1.wait(tb=0)
ch_to_rank2.signal(tb=0)
ch_from_rank2.wait(tb=0)
ch_to_rank3.signal(tb=0)
ch_from_rank3.wait(tb=0)
```

**Good — signals grouped, then waits grouped (fusion enabled):**

```python
# 3 signals fuse into 1 instruction, 3 waits fuse into 1 instruction = 2 instructions
ch_to_rank1.signal(tb=0)
ch_to_rank2.signal(tb=0)
ch_to_rank3.signal(tb=0)

ch_from_rank1.wait(tb=0)
ch_from_rank2.wait(tb=0)
ch_from_rank3.wait(tb=0)
```

### Wait Fusion

The same principle applies to `wait` operations. Consecutive waits on the same thread block are fused into a single instruction that waits on all channels at once.

### Put / Get Fusion

Consecutive `put` or `get` calls on the same thread block and same channel type can also be fused. Group data transfer operations together when possible.

**Good — puts grouped for fusion:**

```python
for peer in range(gpus_per_node):
    if g != peer:
        ch.put(dst_chunk, src_chunk, tb=peer)
# All puts on the same tb are fused
```

### General Fusion Rule

**Same-type consecutive operations on the same thread block are fused.** Any different operation type between them breaks the fusion chain. Structure your code so that operations of the same type are emitted in sequence without interleaving.

## 2. Semaphores vs Barriers

### When to Use Barriers

`Rank.barrier(tb_list=[...])` synchronizes **all** thread blocks in the list — every thread block both signals and waits for every other thread block in the group. This is a symmetric, all-to-all synchronization.

Use barriers when multiple thread blocks on the same rank need to synchronize with each other and all of them have work on both sides of the synchronization point.

```python
# All thread blocks in the list sync with each other
r = Rank(src_rank)
r.barrier(tb_list=[0, 1, 2, 3])
```

### When to Use Semaphores

When synchronization is **asymmetric** — one side produces data and the other side consumes it — use `Semaphore` instead of `barrier`. With a semaphore, only the consumer waits and only the producer signals. This avoids unnecessary synchronization in the reverse direction.

**Example: one producer thread block, multiple consumer thread blocks.**

With a barrier, all thread blocks would wait for each other (unnecessary overhead):

```python
# Bad — barrier forces all TBs to wait for each other
r.barrier(tb_list=[producer_tb, consumer_tb_0, consumer_tb_1])
```

With semaphores, only consumers wait for the producer:

```python
# Good — asymmetric sync: producer releases, consumers acquire
sem = Semaphore(rank=src_rank, initial_value=0)

# Producer side (runs on producer_tb)
sem.release(tb=producer_tb, data_sync=SyncType.before)

# Consumer side (runs on each consumer_tb)
sem.acquire(tb=consumer_tb_0, data_sync=SyncType.after)
sem.acquire(tb=consumer_tb_1, data_sync=SyncType.after)
```

### Decision Guide

| Scenario | Use |
|---|---|
| All TBs need to sync with each other | `barrier` |
| One TB produces, others consume | `Semaphore` |
| Only one side needs to wait | `Semaphore` |
| Two TBs need mutual sync | `barrier` (simpler) or two semaphores |

## 3. data_sync Placement

The `data_sync` parameter controls where `__syncthreads()` is placed relative to a signal/wait operation. Choosing the right value avoids redundant thread synchronization.

| Value | Meaning |
|---|---|
| `SyncType.both` | sync before **and** after the operation (safest, default) |
| `SyncType.before` | sync only before — use when no data is read after |
| `SyncType.after` | sync only after — use when no data was written before |
| `SyncType.none` | no sync — use when data ordering is guaranteed by other means |

**Common patterns:**

- `signal(data_sync=SyncType.none)` — when signaling without preceding writes on this TB (e.g., initial handshake)
- `wait(data_sync=SyncType.after)` — when the wait must complete before subsequent reads
- `signal(data_sync=SyncType.before)` — when preceding writes must complete before the signal

Avoid `SyncType.both` unless truly needed — it adds two synchronization points.

## 4. Relaxed Memory Ordering

The `relaxed` parameter on `signal` and `wait` (MemoryChannel only) enables relaxed memory ordering, which can improve performance when strict ordering guarantees are not required.

Use `relaxed=True` for initial handshakes or synchronization points where the data being protected has not yet been modified:

```python
# Initial sync — no data was written, relaxed is safe
ch.signal(tb=0, data_sync=SyncType.none, relaxed=True)
ch.wait(tb=0, data_sync=SyncType.after, relaxed=True)
```

Do **not** use `relaxed=True` when the signal is protecting a preceding data write that the peer must observe.

## 5. Channel Type Selection

| Channel | Best for | Latency | Bandwidth |
|---|---|---|---|
| `MemoryChannel` | Intra-node (SM-level direct access) | Lowest | High |
| `PortChannel` | Inter-node (NVLink/IB proxy-based) | Higher | Highest for large messages |

Use `MemoryChannel` for communication between GPUs on the same node. Use `PortChannel` for cross-node communication. Mixing them appropriately in hierarchical algorithms (like hierarchical allgather) yields the best results.

## 6. Thread Block Assignment

Distribute work across thread blocks to maximize parallelism, but keep related operations on the same thread block to enable fusion.

**Guidelines:**
- Assign inter-node communication to a dedicated thread block (e.g., `tb=tb_offset`) so it does not block intra-node work.
- Assign intra-node peer communication to separate thread blocks per peer for parallelism.
- Keep signal/wait/put sequences for the same communication flow on the same thread block to enable fusion.

## Priority

Always apply optimizations in this order:
1. **Correctness** — never sacrifice correctness for performance.
2. **Instruction fusion** — biggest impact, easiest to apply.
3. **Semaphores over barriers** — reduces unnecessary synchronization.
4. **data_sync tuning** — eliminates redundant `__syncthreads()`.
5. **Relaxed ordering** — minor gains, use carefully.
6. **Channel type selection** — topology-dependent, high impact for multi-node.