# Hardware Profile: NVIDIA H100 (single node)

> Active default profile for the MSCCL++ DSL agents (AllReduce, AllGather, etc.). Update this file when hardware details change.

## Topology
- **num_gpus:** 8 (single node, 8 × H100 SXM per node, fully connected via 4th-gen NVLink + NVSwitch).
- NVLink bandwidth: ~900 GB/s per GPU (bidirectional aggregate).
- NVSwitch provides all-to-all GPU connectivity within the node.
- Multi-node: InfiniBand (NDR / 400 Gbps class) via `PortChannel`.

## Compute
- CUDA capability 9.0 (Hopper).
- 132 SMs per GPU; large register file; 128 KB shared memory per SM (configurable).
- Tensor Cores with fp16 / bf16 / fp8 (e4m3, e5m2 — OCP variants on NVIDIA).

## NVLink SHARP (NVLS)
- **Available.** In-switch multimem reductions and broadcasts via `SwitchChannel`.
- Strong fit for small/medium AllReduce (in-switch reduce) and AllGather (in-switch broadcast/multimem load).
- Supported reduce ops/dtypes for AllReduce: confirm against the runtime; common combos: fp16/bf16 sum.

## Channels (DSL mapping)
- `MemoryChannel` — peer-to-peer NVLink memory access (default intra-node).
- `SwitchChannel` — NVLS multimem reductions/broadcasts.
- `PortChannel` — inter-node IB transport (multi-node only).

## Common known pitfalls (apply to all collectives)
- Zero-copy designs require **identical input/output buffer offsets across ranks** (see `docs/dsl/concepts.md` § Executor limitations). Do not propose zero-copy without explicit `symmetric_memory: yes`.
- Avoid oversubscribing SMs with too many TBs × `instances` — register pressure can spill and hurt latency.
- NVLS ops have constraints on supported dtypes/reduce ops — verify before designing around them.

## Baselines
- See `docs/dsl/results.md` and `docs/dsl/figs/single_node_allreduce_results_*.png` for measured AllReduce baselines.
- AllGather baselines: TBD — measure and link when available.

---

## AllReduce

### Recommended starting points by message size
| Regime | Size (typical) | Suggested algorithm family | Reference example |
| --- | --- | --- | --- |
| Latency-bound | ≤ 32 KB | LL packet AllReduce, or NVLS packet | `python/mscclpp/language/tests/single_node/allreduce_pkt.py`, `src/ext/collectives/allreduce/allreduce_nvls_packet.cu` |
| Medium / one-shot | 32 KB – 1 MB | NVLS one-shot, zero-copy | `python/mscclpp/language/tests/single_node/allreduce_nvls_zero_copy.py` |
| Bandwidth-bound | ≥ 1 MB | Reduce-Scatter + AllGather, pipelined; or NVLS block/warp pipeline | `python/mscclpp/language/tests/single_node/allreduce_pipeline.py`, `src/ext/collectives/allreduce/allreduce_rsag_pipeline.cu`, `src/ext/collectives/allreduce/allreduce_nvls_warp_pipeline.cu` |
| Multi-node | any | Intra-node NVLS RSAG + inter-node IB AllGather | `python/mscclpp/default_algos/allreduce_multi_nodes.py` |

### Recommended starting parameters
- `num_threads_per_block`: 1024 (try 512 / 768 / 1024 during tuning).
- `instances`: 1–8. Start at 8 for NVLS one-shot (matches `allreduce_nvls_zero_copy.py`); 1–4 is typical for pipelined RSAG or all-pair `MemoryChannel` designs where additional replication competes for SMs and channels.
- Thread blocks: 4–16 depending on algorithm; favor fewer TBs for latency-bound paths.
- `protocol`: `"LL"` for small messages, `"Simple"` otherwise.
- `use_double_scratch_buffer`: off by default; turn on for pipelined large-message designs that need overlap.

---

## AllGather

### Recommended starting points by message size
| Regime | Size (typical) | Suggested algorithm family | Reference example |
| --- | --- | --- | --- |
| Latency-bound | ≤ 32 KB | LL packet AllGather (`Simple` protocol; small per-rank chunks) | `python/mscclpp/language/tests/single_node/allgather_pkt.py`, `allgather_pkt_rppkt.py` |
| Medium | 32 KB – 1 MB | Fullmesh all-pair `MemoryChannel` put | `python/mscclpp/language/tests/single_node/allgather.py`, `src/ext/collectives/allgather/allgather_fullmesh.cu` |
| Bandwidth-bound | ≥ 1 MB | Ring AllGather (low SM, high BW) | `python/mscclpp/language/tests/single_node/allgather_ring.py` |
| Heterogeneous TB allocation | any | Thread-Block-Group variant for uneven copy/send workload | `python/mscclpp/language/tests/single_node/allgather_tbg.py` |
| Multi-node | any | Intra-node fullmesh + inter-node IB ring/fullmesh | `python/mscclpp/language/tests/multi_node/allgather.py`, `allgather_pkt.py`, `allgather_tbg.py` |

### Recommended starting parameters
- `num_threads_per_block`: 1024 (try 512 / 768 / 1024 during tuning).
- `instances`: 1–32. Start at 1–8 for fullmesh / packet; ring AllGather scales well with high replication (`allgather_ring.py` uses `instances=32`) because each instance's per-step transfer is small and bandwidth-friendly.
- Thread blocks: 4–8 per instance for fullmesh; ring uses fewer TBs but more instances.
- `protocol`: `"LL"` for small messages, `"Simple"` otherwise.
- `use_double_scratch_buffer`: off by default; AllGather rarely needs it (no reduction = no intermediate accumulation).

### AllGather-specific notes
- `AllGather(num_ranks, chunk_factor, inplace)` produces `chunk_factor` input chunks per rank and `num_ranks × chunk_factor` output chunks per rank. With `inplace=True`, the per-rank input is a slice of the output buffer.
- NVLS is not typically used for AllGather alone (NVLS shines on reductions); on H100 the `SwitchChannel.broadcast` path can still help in mixed RS+AG patterns, but pure AllGather usually picks `MemoryChannel`-based fullmesh or ring.
- Ring is the classic bandwidth-optimal pattern. Fullmesh is latency-optimal at the cost of `(N-1)` channels per rank.
