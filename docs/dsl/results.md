# Results

This page presents performance benchmarks for collective communication algorithms implemented using the MSCCL++ DSL (Domain Specific Language).

## Available Algorithms

The following reference implementations are provided:

### Single-Node AllReduce on H100 (NVLS)

We evaluate a single-node AllReduce algorithm designed for NVIDIA H100 GPUs leveraging NVLink Switch (NVLS) technology. This algorithm demonstrates optimal performance for intra-node collective operations.

**Source Code Location:**

The algorithm implementation can be found at:
```
mscclpp/python/mscclpp/language/tests
```

**Running the Benchmark:**

Users can generate the corresponding JSON execution plan by following the steps described in the Quick Start section. Once the JSON file is generated, it can be executed using the `executor_test.py` tool to measure performance.

**Performance Results:**

The following figures show the achieved bandwidth for message sizes ranging from 1KB to 1GB:

```{figure} ./figs/single_node_allreduce_results_1K_to_1M.png
:name: single-node-allreduce-small
:alt: Single-node AllReduce performance (1KB to 1MB)
:align: center

Single-node AllReduce performance on H100 with NVLS (1KB to 1MB message sizes)
```

```{figure} ./figs/single_node_allreduce_results_1M_to_1G.png
:name: single-node-allreduce-large
:alt: Single-node AllReduce performance (1MB to 1GB)
:align: center

Single-node AllReduce performance on H100 with NVLS (1MB to 1GB message sizes)
```

### Two-Node AllReduce on H100 (Small Message Sizes)

We also provide a two-node AllReduce algorithm for H100 GPUs, specifically optimized for small message sizes. This algorithm uses a non-zero-copy communication path to minimize latency for small data transfers.

**Installation:**

This algorithm is installed by default when running:
```bash
python3 -m mscclpp --install
```

**Execution Plan Location:**

After installation, the generated JSON execution plan can be found at:
```
~/.cache/mscclpp/default/
```

**Performance Results:**

The figure below shows the performance characteristics for small message sizes in a two-node configuration:

```{figure} ./figs/2node_all_reduce_results.png
:name: two-node-allreduce-small
:alt: Two-node AllReduce performance for small message sizes
:align: center

Two-node AllReduce performance on H100 for small message sizes
```


