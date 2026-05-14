# Hardware Profile: NVIDIA GB200 (NVL72 rack)

> Profile for MSCCL++ DSL agents (AllReduce, AllGather, etc.) targeting GB200 NVL72 systems. Defaults below assume a full NVL72 coherent NVLink domain (72 Blackwell GPUs). For smaller deployments (NVL36, single compute tray with 4 GPUs, or an 8-GPU HGX-style island), the agent should ask the user to confirm `num_gpus` and the NVL domain size before generating algorithms.

## Topology
- **num_gpus:** 72 (full NVL72 rack). Common alternatives to confirm with the user: 36 (NVL36), 8 (single HGX-style island), 4 (single compute tray = 2 GB200 superchips).
- **Superchip layout:** each GB200 superchip = 1 Grace CPU + 2 Blackwell (B200) GPUs, NVLink-C2C between Grace and each GPU (~900 GB/s aggregate Grace↔GPU coherent).
- **NVLink generation:** 5th-gen NVLink. ~1.8 TB/s per-GPU bidirectional aggregate (~900 GB/s each direction across 18 links × 100 GB/s).
- **NVSwitch:** 4th-gen NVSwitch fabric (NVLink Switch System) provides full all-to-all GPU connectivity within the NVL72 domain — a single flat 72-GPU NVLink domain, not a hierarchy of 8-GPU islands.
- **Inter-rack interconnect:** ConnectX-7/8 + Quantum-2/X800 InfiniBand (typically 400 / 800 Gbps class) via `PortChannel`. Confirm exact NIC count and speed with the user for multi-rack designs.
- **Grace ↔ Blackwell coherent memory:** unified-memory programming is supported; treat host pointers as accessible from GPU kernels, but performance-critical buffers should still live in HBM.

## Compute
- CUDA capability **10.0** (Blackwell).
- ~148 SMs per B200 GPU (confirm with `cudaGetDeviceProperties` on target node — varies by SKU bin).
- 228 KB shared memory per SM (configurable via opt-in), large register file.
- Tensor Cores with fp16 / bf16 / fp8 (e4m3, e5m2 — OCP variants) and new **fp6 / fp4** datatypes via the 2nd-gen Transformer Engine. **Note:** the MSCCL++ DSL may not yet expose fp6/fp4 reductions — verify dtype support in `mscclpp.language` before designing around them.
- 192 GB HBM3e per GPU (confirm — some SKUs ship with different capacities).

## NVLink SHARP (NVLS)
- **Available** and significantly more capable than H100: NVLS multimem reductions and broadcasts operate across the full NVL72 domain via `SwitchChannel`, not just an 8-GPU island.
- Strong fit for AllReduce (in-switch reduce) and AllGather (in-switch broadcast / multimem load) at all message sizes within the domain.
- Supported reduce ops/dtypes: confirm against the runtime; common combos are fp16/bf16 sum. fp8/fp6/fp4 in-switch reductions: **verify before relying on them.**

## Channels (DSL mapping)
- `MemoryChannel` — peer-to-peer NVLink memory access across the NVL domain (up to 72 peers — far more than H100's 8). Be deliberate about all-pair fan-out: with `num_gpus=72`, naive fullmesh creates 71 channels per rank, which can be punitive on registers and instance count.
- `SwitchChannel` — NVLS multimem reductions/broadcasts; the natural primitive for collectives at this scale.
- `PortChannel` — inter-rack IB transport for multi-NVL72 deployments.

## Common known pitfalls (apply to all collectives)
- Zero-copy designs require **identical input/output buffer offsets across ranks** (see `docs/dsl/concepts.md` § Executor limitations). Do not propose zero-copy without explicit `symmetric_memory: yes`.
- Avoid oversubscribing SMs with too many TBs × `instances`. Blackwell has more SMs than Hopper, but register pressure still bites — start conservative and grow.
- NVLS ops have dtype / reduce-op constraints — verify against the runtime before designing around them.
- **Fullmesh `MemoryChannel` does not scale linearly past ~8–16 ranks.** Prefer NVLS, ring, or hierarchical (RS+AG) designs at NVL72 scale.
- Hierarchical designs that worked on multi-node H100 (intra-node NVLS + inter-node IB) collapse here: within a single NVL72 the whole 72-GPU set is one NVLink domain. Don't artificially split it unless modeling multi-rack.

## Baselines
- TBD — link to measured GB200 results once available. Until then, treat the algorithm recommendations below as starting points for tuning, not validated optima.

---

## AllReduce

### Recommended starting points by message size
| Regime | Size (typical) | Suggested algorithm family | Reference example |
| --- | --- | --- | --- |
| Latency-bound | ≤ 64 KB | NVLS packet AllReduce, or LL packet AllReduce | `src/ext/collectives/allreduce/allreduce_nvls_packet.cu`, `python/mscclpp/language/tests/single_node/allreduce_pkt.py` |
| Medium / one-shot | 64 KB – 4 MB | NVLS one-shot (zero-copy if buffer offsets are symmetric) | `python/mscclpp/language/tests/single_node/allreduce_nvls_zero_copy.py` |
| Bandwidth-bound | ≥ 4 MB | NVLS warp/block pipeline, or Reduce-Scatter + AllGather pipelined over NVLS | `src/ext/collectives/allreduce/allreduce_nvls_warp_pipeline.cu`, `src/ext/collectives/allreduce/allreduce_rsag_pipeline.cu`, `python/mscclpp/language/tests/single_node/allreduce_pipeline.py` |
| Multi-rack | any | Intra-rack NVLS RSAG + inter-rack IB AllGather | `python/mscclpp/default_algos/allreduce_multi_nodes.py` (adapt domain size from 8 → 72) |

### Recommended starting parameters
- `num_threads_per_block`: 1024 (try 512 / 768 / 1024 during tuning).
- `instances`: 1–8 for NVLS one-shot. Pipelined large-message designs can push to 8–16 if SM occupancy allows.
- Thread blocks: 8–32 depending on algorithm. With 148 SMs/GPU there's more headroom than on H100, but favor fewer TBs on latency-bound paths.
- `protocol`: `"LL"` for small messages (≤ ~64 KB), `"Simple"` otherwise.
- `use_double_scratch_buffer`: off by default; turn on for pipelined large-message designs that need compute/transfer overlap.

---

## AllGather

### Recommended starting points by message size
| Regime | Size (typical) | Suggested algorithm family | Reference example |
| --- | --- | --- | --- |
| Latency-bound | ≤ 64 KB | LL packet AllGather (`Simple` protocol; small per-rank chunks); or NVLS broadcast/multimem-load | `python/mscclpp/language/tests/single_node/allgather_pkt.py`, `allgather_pkt_rppkt.py` |
| Medium | 64 KB – 4 MB | NVLS broadcast or ring AllGather over NVLS | `python/mscclpp/language/tests/single_node/allgather.py`, `allgather_ring.py` |
| Bandwidth-bound | ≥ 4 MB | Ring AllGather (low SM, high BW); NVLS-assisted ring for mixed RS+AG patterns | `python/mscclpp/language/tests/single_node/allgather_ring.py` |
| Heterogeneous TB allocation | any | Thread-Block-Group variant for uneven copy/send workload | `python/mscclpp/language/tests/single_node/allgather_tbg.py` |
| Multi-rack | any | Intra-rack ring / NVLS broadcast + inter-rack IB ring or fullmesh | `python/mscclpp/language/tests/multi_node/allgather.py`, `allgather_pkt.py`, `allgather_tbg.py` |

### Recommended starting parameters
- `num_threads_per_block`: 1024 (try 512 / 768 / 1024 during tuning).
- `instances`: 1–32. Start at 1–8 for NVLS / packet; ring AllGather over 72 ranks scales well with higher replication (e.g. 16–32) because each instance's per-step transfer is small and bandwidth-friendly.
- Thread blocks: 4–8 per instance for NVLS-assisted designs; ring uses fewer TBs but more instances.
- `protocol`: `"LL"` for small messages, `"Simple"` otherwise.
- `use_double_scratch_buffer`: off by default; AllGather rarely needs it.

### AllGather-specific notes
- `AllGather(num_ranks, chunk_factor, inplace)` produces `chunk_factor` input chunks per rank and `num_ranks × chunk_factor` output chunks per rank. With `inplace=True`, the per-rank input is a slice of the output buffer.
- With 72 ranks, naive fullmesh AllGather (71 channels per rank) is usually a bad tradeoff — prefer ring or NVLS broadcast.
- NVLS `SwitchChannel.broadcast` is far more attractive at NVL72 scale than at H100/8-GPU scale because the domain itself is large; consider it even for pure AllGather, not just RS+AG composites.

---

### Activation checklist (for the agent)

Before treating this profile as authoritative for a specific deployment:
1. Confirm with the user: `num_gpus` and NVL domain size (NVL72 / NVL36 / single tray / HGX island). The defaults above assume NVL72.
2. Confirm whether the workload is single-rack (one NVL72) or multi-rack (use the multi-rack rows).
3. Confirm IB fabric details (NIC count per node, link speed) if generating a multi-rack design.
4. Verify DSL-level dtype support for fp8 / fp6 / fp4 reductions before designing around them.
5. Note in the generated DSL file's header comment that the design was tuned for GB200 NVL72 per this profile revision.
