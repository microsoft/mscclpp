# MSCCL++ DSL Testing Guide

This document describes how to test DSL algorithms using `executor_test.py`.

## Overview

The testing workflow has two steps:

1. **Use the DSL algorithm to generate the JSON execution plan** — run the script to produce a `.json` file.
2. **Run the executor test** — launch `executor_test.py` with MPI to validate correctness and measure performance.

## Prerequisites

- MSCCL++ Python package installed (`pip install -r python/requirements_cuda12.txt` and the mscclpp package built/installed).
- MPI available (`mpirun` from OpenMPI or equivalent).
- CUDA-capable GPUs (one per rank).

## Step 1: Generate the JSON Execution Plan

Run the algorithm script to produce a JSON file depeneding on the algorithm parameters:

```bash
python3 path/to/algorithm.py \
    --name my_algorithm \
    --num_gpus <N> \
    --instances 1 \
    > my_algorithm.json
```

For hierarchical algorithms that require `--gpus_per_node`:

```bash
python3 path/to/hierarchical_algorithm.py \
    --name my_algorithm \
    --num_gpus 8 \
    --gpus_per_node 4 \
    --instances 1 \
    > my_algorithm.json
```

Verify the JSON was generated correctly (non-empty, valid JSON):

```bash
python3 -m json.tool my_algorithm.json > /dev/null && echo "Valid JSON"
```

## Step 3: Run the Executor Test

### Single Node

For single-node testing where all GPUs are on the same machine:

```bash
mpirun --allow-run-as-root -np <NUM_GPUS> --bind-to numa \
    python3 python/test/executor_test.py \
    -path my_algorithm.json \
    --size 1M \
    --in_place
```

**Example — 4-GPU allgather on a single node:**

```bash
mpirun --allow-run-as-root -np 4 --bind-to numa \
    python3 python/test/executor_test.py \
    -path my_allgather.json \
    --size 1M \
    --in_place
```

**Example — 8-GPU hierarchical allgather simulating 2 nodes of 4 GPUs:**

```bash
mpirun --allow-run-as-root -np 8 --bind-to numa \
    python3 python/test/executor_test.py \
    -path my_hierarchical_allgather.json \
    --size 1M \
    --in_place
```

### Multi-Node

For multi-node testing across multiple machines:

```bash
mpirun --allow-run-as-root \
    -np <TOTAL_GPUS> \
    --host node1:<GPUS_PER_NODE>,node2:<GPUS_PER_NODE> \
    --bind-to numa \
    -x LD_LIBRARY_PATH \
    -x PATH \
    python3 python/test/executor_test.py \
    -path my_algorithm.json \
    --size 1M \
    --in_place
```

**Example — 2 nodes with 4 GPUs each:**

```bash
mpirun --allow-run-as-root \
    -np 8 \
    --host node1:4,node2:4 \
    --bind-to numa \
    -x LD_LIBRARY_PATH \
    -x PATH \
    python3 python/test/executor_test.py \
    -path my_hierarchical_allgather.json \
    --size 1M \
    --in_place
```

When using a hostfile instead:

```bash
mpirun --allow-run-as-root \
    -np 8 \
    --hostfile hostfile.txt \
    --bind-to numa \
    -x LD_LIBRARY_PATH \
    -x PATH \
    python3 python/test/executor_test.py \
    -path my_algorithm.json \
    --size 1M \
    --in_place
```

Where `hostfile.txt` contains:

```
node1 slots=4
node2 slots=4
```

## executor_test.py Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `-path` / `--execution_plan_path` | Yes | — | Path to the JSON execution plan |
| `--size` | Yes | — | Buffer size. Supports suffixes: `K`, `M`, `G` (e.g., `1M` = 1 MiB) |
| `--in_place` | No | `False` | Use in-place buffers (input and output share memory) |
| `--dtype` | No | `float16` | Data type: `float16`, `float32`, or `int32` |
| `--packet_type` | No | `LL16` | Packet type: `LL8` or `LL16` |
| `--n_iters` | No | `10` | Number of iterations for correctness and benchmarking |
| `--n_graph_iters` | No | `10` | Number of CUDA graph iterations for benchmarking |

## Interpreting Results

### Success

A successful run prints one line per rank with the execution time:

```
Rank: 0 Execution time: 43.27 us, data size: 1048576 bytes data type: float16 packet type: mscclpp._mscclpp.CppPacketType.LL16
Rank: 1 Execution time: 43.26 us, data size: 1048576 bytes data type: float16 packet type: mscclpp._mscclpp.CppPacketType.LL16
...
```

The test automatically validates correctness by:
1. Filling input buffers with known data patterns.
2. Running the collective.
3. Checking the output against expected results using a CUDA verification kernel.

If correctness fails, the process will crash or report an assertion error.

### Failure

Common failure modes:

| Symptom | Likely Cause |
|---|---|
| Hang (no output) | Deadlock in the algorithm — check signal/wait pairing and dependencies |
| CUDA error or segfault | Invalid buffer index, out-of-bounds chunk access, or wrong channel direction |
| Assertion error | Incorrect collective result — data not matching expected output |
| `RuntimeError` at JSON generation | DSL validation error — check rank bounds, channel setup, chunk sizes |

### Debugging Tips

- Start with the smallest size (`--size 1K`) to catch correctness issues quickly.
- Use `--n_iters 1` and `--n_graph_iters 1` for faster debugging cycles.
- Test with the minimum number of ranks first, then scale up.
- For hangs, check that every `signal` has a matching `wait` and every `put`/`get` has proper synchronization.

## Testing Checklist

Before considering an algorithm validated:

- [ ] JSON generation completes without errors.
- [ ] Single-node test passes with the target number of ranks.
- [ ] Test passes with multiple buffer sizes (`1K`, `1M`, `16M`).
- [ ] If hierarchical, multi-node test passes (or simulated multi-node with all GPUs on one machine).
- [ ] No warnings or errors in output (OpenFabrics/btl warnings are benign and can be ignored).