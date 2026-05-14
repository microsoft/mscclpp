# Hardware Profile: NVIDIA H100 (single node)

> Active default profile for the AllReduce DSL agent. Update this file when hardware details change.

## Topology
- 8 × H100 SXM GPUs per node, fully connected via 4th-gen NVLink + NVSwitch.
- NVLink bandwidth: ~900 GB/s per GPU (bidirectional aggregate).
- NVSwitch provides all-to-all GPU connectivity within the node.
- Multi-node: InfiniBand (NDR / 400 Gbps class) via `PortChannel`.

## Compute
- CUDA capability 9.0 (Hopper).
- 132 SMs per GPU; large register file; 128 KB shared memory per SM (configurable).
- Tensor Cores with fp16 / bf16 / fp8 (e4m3, e5m2 — OCP variants on NVIDIA).

## NVLink SHARP (NVLS)
- **Available.** In-switch multimem reductions via `SwitchChannel`.
- Strong fit for small/medium messages and one-shot AllReduce.
- Supported reduce ops/dtypes: confirm against the runtime; common combos: fp16/bf16 sum.

## Channels (DSL mapping)
- `MemoryChannel` — peer-to-peer NVLink memory access (default intra-node).
- `SwitchChannel` — NVLS multimem reductions/broadcasts.
- `PortChannel` — inter-node IB transport (multi-node only).

## Recommended starting points by message size
| Regime | Size (typical) | Suggested algorithm family | Reference example |
| --- | --- | --- | --- |
| Latency-bound | ≤ 32 KB | LL packet AllReduce, or NVLS packet | `allreduce_pkt.py`, `src/.../allreduce_nvls_packet.cu` |
| Medium / one-shot | 32 KB – 1 MB | NVLS one-shot, zero-copy | `allreduce_nvls_zero_copy.py` |
| Bandwidth-bound | ≥ 1 MB | Reduce-Scatter + AllGather, pipelined; or NVLS block/warp pipeline | `allreduce_pipeline.py`, `src/.../allreduce_rsag_pipeline.cu`, `src/.../allreduce_nvls_warp_pipeline.cu` |
| Multi-node | any | Intra-node NVLS RSAG + inter-node IB AllGather | `default_algos/allreduce_multi_nodes.py` |

## Recommended starting parameters
- `num_threads_per_block`: 1024 (try 512 / 768 / 1024 during tuning).
- `instances`: 1–8. Start at 8 for NVLS one-shot (matches `allreduce_nvls_zero_copy.py`); 1–4 is typical for pipelined RSAG or all-pair `MemoryChannel` designs where additional replication competes for SMs and channels.
- Thread blocks: 4–16 depending on algorithm; favor fewer TBs for latency-bound paths.
- `protocol`: `"LL"` for small messages, `"Simple"` otherwise.
- `use_double_scratch_buffer`: off by default; turn on for pipelined large-message designs that need overlap.

## Known pitfalls
- Zero-copy designs require **identical input/output buffer offsets across ranks** (see `docs/dsl/concepts.md` § Executor limitations).
- Avoid oversubscribing SMs with too many TBs × `instances` — register pressure can spill and hurt latency.
- NVLS ops have constraints on supported dtypes/reduce ops — verify before designing around them.

## Baselines
- See `docs/dsl/results.md` and `docs/dsl/figs/single_node_allreduce_results_*.png` for measured baselines on this profile.
