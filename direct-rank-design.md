# Direct Remote Rank-Major Output Design

## 1. Status

The fused GEMM peer-store proposal below is superseded. The accepted
implementation does not change rank-major dispatch or the GEMM kernel:

```text
rank-major dispatch
  -> existing GEMM2 weighted route output [rank_rows, top_k, hidden]
  -> no post-GEMM producer-local top-k sum
  -> MSCCL++ combine remotely pulls route rows
  -> source-local FP32 top-k reduction
  -> BF16 output
```

Only the post-GEMM output contract and rank-major combine change. The original
proposal is retained below as rejected design history; do not implement its
peer-address GEMM epilogue or completion protocol.

The corrected path is selected by `CombineMode::DIRECT_SEND` together with
`DispatchLayout::RANK_MAJOR`. `CombineMode::RANK_LOCAL_REDUCE` retains the
existing producer-local reduction for compatibility and baseline comparison.
The local-reduce mode aliases its combine input to the dispatch output and does
not allocate a second 2-D buffer. Direct mode alone allocates the larger 3-D
route buffer.
SGLang uses one `MSCCLPPDispatchOutput` and one `MSCCLPPCombineInput` across
MSCCL++ layouts and modes. The dispatch format tag identifies layout-specific
metadata. Rank-major dispatch exposes `route_output_buffer` only for direct
mode; MAI receives exactly one output tensor, either the 2-D alias or the 3-D
route buffer. Its `reduce_output` flag explicitly selects local reduction
instead of inferring behavior from tensor rank.

GB200 CUDA-Graph results at `T=64`, `H=4096`, `I=6656`, 16 local experts,
and top-8:

| GPUs | Rank-local reduce | Combine-side reduce | Delta |
|---:|---:|---:|---:|
| 4 | 477.9 us | 470.1 us | -1.62% |
| 8 | 548.8 us | 533.8 us | -2.73% |
| 16 | 680.2 us | 654.3 us | -3.81% |
| 32 | 948.3 us | 896.1 us | -5.50% |

The 32-GPU values are means of two matched runs: 947.0/949.5 us for rank-local
reduce and 893.2/898.9 us for combine-side reduce. All runs had exact
eager/CUDA-Graph parity. A 4-GPU identity-GEMM test also matched the analytical
route-level BF16 reference exactly in both eager and CUDA-Graph execution.

At 32 GPUs and 32 tokens/rank, while retaining 16 local experts/rank (512
global experts), rank-local reduce took 711.0 us and combine-side reduce took
683.8 us (-3.83%). A second combine-side run took 685.7 us. The reverse-order
baseline repeat did not launch because a production SGLang deployment started
on the test nodes; it produced no benchmark sample.

The original all-routes MAI composition applied SwiGLU to the full rank-major
capacity, including routes with local expert ID `-1`. The shared production
path now uses SGLang's filtered `act_and_mul_triton` with those existing IDs,
without changing dispatch, W8A16 GEMM, route-output addressing, or combine.
A quick 5-replay by 20-iteration CUDA-Graph A/B while the resident deployment
was compute-idle measured:

| GPUs | Dense SwiGLU | Active-route SwiGLU | Delta |
|---:|---:|---:|---:|
| 4 | 470.5 us | 440.8 us | -6.31% |
| 16 | 655.2 us | 507.9 us | -22.48% |
| 32 | 898.2 us | 584.9 us | -34.88% |

The 32-GPU active value is the mean of 585.7 and 584.1 us. All runs had exact
eager/CUDA-Graph parity. Focused activation tests matched the previous dense
kernel within the established BF16 tolerance (`max_abs=0.0625`), and the full
MAI path matched its numerical reference.
At the EP32 route sparsity, filtered Triton activation outperformed SGLang's
filtered native JIT kernel: 10.3 vs 13.2 us at T32, 14.6 vs 20.8 us at T64,
and 24.6 vs 36.9 us at T128. The native JIT kernel was bit-exact to the old
dense kernel; Triton had max absolute difference 0.0625 within BF16 tolerance.
The shared production path uses the filtered native JIT kernel to preserve
bit-exact activation numerics.

A 32-GPU T64 CUDA-Graph stage profile measured rank-mean dispatch at 23.9 us,
MAI compute at 555.1 us, and combine at 26.1 us. External timing nodes raised
the profiled graph sum to 605.2 us versus 584.3 us uninstrumented; preserving
the measured proportions gives an approximate uninstrumented breakdown of
23.1 us dispatch, 536.0 us compute, and 25.2 us combine.

At 32 GPUs and 32 tokens/rank, active-route SwiGLU measured 534.6 us E2E.
Stage markers measured 25.4 us dispatch, 498.5 us MAI compute, and 31.8 us
combine (555.7 us with marker overhead). Normalized to the uninstrumented E2E,
the approximate breakdown is 24.4 us dispatch, 479.6 us compute, and 30.6 us
combine.

Matching the production trace dimensions (`H=4352`, `I=6528`) at 32 GPUs and
32 tokens/rank measured 564.4 us E2E and 1.814M global tokens/s with exact
eager/CUDA-Graph parity.

At 128 tokens/rank, the same EP32 shape measured 1015.7 us E2E and 4.033M
global tokens/s with filtered native JIT activation and exact eager/CUDA-Graph
parity. It uses one FC1, one
active-route SwiGLU, and one FC2 call; the experimental T1024 chunk loop was
removed. A prior filtered-Triton rank-zero sequence measured 20.4 us dispatch,
9.2 us alignment, 22.7 us count/sort, 164.8 us FC1 gather/populate, 321.3 us
FC1 GEMM, 25.6 us activation, 110.0 us FC2 gather/populate, 194.6 us FC2 GEMM,
7.1 us metadata kernels, and 127.3 us combine.
The prior matched filtered-Triton rank-local-reduce baseline measured 1100.7 us
and 3.721M global tokens/s versus 1008.7 us for direct combine, an E2E reduction
of 92.0 us (8.36%). The baseline
eager/CUDA-Graph outputs remained within the configured BF16 tolerance, with
global max absolute difference 0.25. Representative rank-zero sequences showed
that the common dispatch/GEMM/activation kernels were within normal run
variation. Rank-local reduce added an 83.4 us producer top-k reduction and
14.0 us BF16 copy before its 119.0 us combine; direct mode replaced that tail
with a 127.3 us combine, reducing the dispatch-to-combine span by 92.8 us.

A rank-zero CUDA kernel profile of that shape attributed 430.2 us to the two
W8A16 GEMMs, 70.5 us to their gather/populate prologues, 13.0 us to active-route
SwiGLU, 10.6 us to route alignment/sorting, 7.4 us to other compute kernels,
14.0 us to dispatch, and 16.7 us to combine. These kernels total 562.4 us,
within 2.0 us of the uninstrumented 564.4 us E2E.

The proposed mode lets the expert rank's GEMM2 epilogue write each weighted
route row directly into the originating rank's symmetric route buffer. The
originating rank then performs only a local FP32 top-k reduction.

The design is intentionally different from a separate producer-side push
kernel. The remote write must be fused into GEMM2; otherwise it adds another
read of the local GEMM2 output and another kernel launch.

## 2. Goal

Replace:

```text
GEMM2 local weighted-route output
  -> persistent remote-pull combine kernel
  -> remote TMA loads
  -> FP32 top-k reduction
  -> BF16 output
```

with:

```text
GEMM2 weighted epilogue
  -> direct peer-mapped route stores
  -> per-phase system-release completion
  -> source-local FP32 top-k reduction
  -> BF16 output
```

The expected benefit is:

1. No persistent pull workers competing with GEMM2.
2. No GEMM2 SM reservation.
3. No physical GEMM release gate.
4. Network stores overlap naturally with GEMM2 epilogue execution.
5. The remaining post-GEMM work reads only local HBM.

## 3. Non-goals

The first implementation will not:

1. Atomically accumulate directly into the final BF16 token output.
2. Compact the source route buffer below `[tokens, top_k, hidden]`.
3. Pipeline GEMM1 or activation by expert.
4. Change rank-major dispatch token compaction.
5. Require MSCCL++ types inside MAI/CUTLASS.

Direct accumulation into one output row would require conflicting remote
writes or FP32 atomics. A route buffer keeps every writer conflict-free and
preserves the existing numerics.

## 4. Production shape

```text
tokens/rank             = 64
top_k                   = 8
hidden                  = 4096
intermediate            = 6656
local experts/rank      = 16
global experts          = 16 * world_size
route output dtype      = BF16
reduction accumulator   = FP32
```

Per source rank and epoch:

```text
route rows = 64 * 8 = 512
route bytes = 512 * 4096 * 2 = 4 MiB
final output bytes = 64 * 4096 * 2 = 0.5 MiB
```

The route-buffer size is independent of world size. At 32 GPUs, every expert
rank still produces about 512 route rows and every source rank still receives
512 route rows under balanced routing.

## 5. Current rank-major mapping

Rank-major dispatch already compacts one token row per source/destination-rank
pair.

For source token `token_idx` and top-k lane `lane`:

```text
global_expert   = topk_ids[token_idx, lane]
destination_rank = global_expert / local_experts
local_expert      = global_expert % local_experts
destination_slot  = compact slot allocated for destination_rank
```

The destination dispatch row is:

```text
dispatch_row =
    source_rank * max_tokens_per_rank + destination_slot
```

All top-k lanes targeting the same destination rank share that compact token
row. Current code stores `destination_slot` in
`WorkspaceView::rankMajorSendIndices_`.

Relevant current sources:

```text
src/ext/ep/low_latency/dispatch.cu
  RankMajorRoute
  prepareRankMajorRoute
  sendRankMajorMetadata
  dispatchSendRankMajorBf16

src/ext/ep/low_latency/config.cuh
  WorkspaceView::rankMajorSendIndices_

src/ext/ep/include/config.hpp
  rankMajorTopkIdsBuffer_
  rankMajorTopkWeightsBuffer_
  rankMajorTokenBuffer_
```

## 6. Additional dispatch metadata

The destination rank currently knows the source rank and compact destination
slot from the dispatch row, but direct write also needs the original source
token index.

Add:

```text
rankMajorSourceTokenIdx[
    source_rank,
    destination_slot
] -> source_token_idx
```

Shape:

```text
[num_ranks, max_tokens_per_rank] int32
```

`dispatchSendRankMajorBf16` writes this metadata with the token and top-k
metadata. It is stable for the lifetime of the dispatched row.

Do not replace compact `destination_slot` with `source_token_idx`. Fixed
source-token slots would create holes in the destination GEMM input and force
GEMM to process unused rows. Keep compact dispatch and carry one extra int32.

## 7. Direct route destination

Each source rank owns a symmetric receive buffer:

```text
directRankRouteRecv[
    buffer_slot,
    source_token_idx,
    topk_lane,
    hidden
] BF16
```

Recommended initial ring depth:

```text
buffer_slots = 2
buffer_slot = dispatch_epoch & 1
```

For one producer route:

```text
remote_route =
    source_token_idx * top_k + topk_lane

remote_dst =
    peer_route_base[source_rank]
    + buffer_slot * route_buffer_stride
    + remote_route * hidden * sizeof(BF16)
```

Every `(source_token_idx, topk_lane)` has exactly one expert owner, so every
route row has exactly one writer. No atomic data stores are required.

For the production shape:

```text
one route buffer slot = 4 MiB/rank
two slots             = 8 MiB/rank
```

## 8. Address preparation

MAI should not include MSCCL++ headers or access private MSCCL++ objects.

MSCCL++ exposes a plain device context containing stable addresses:

```text
peer_route_bases[num_ranks]      uint64
peer_ready_bases[num_ranks]      uint64
source_token_idx metadata        int32*
dispatch epoch                   uint32*
phase source masks               uint64*
```

The Python API passes these tensors or pointers to the W8A16 grouped GEMM.
The CUDA Graph captures stable device addresses; no per-replay host address
construction is allowed.

Recommended public object:

```text
DirectRankWriteContext
  peer_route_bases
  peer_ready_bases
  source_token_indices
  dispatch_epoch
  phase_source_masks
  num_ranks
  max_tokens_per_rank
  top_k
  hidden
```

The context is data-only. MAI receives plain tensors and scalar dimensions.

## 9. GEMM2 epilogue change

The existing W8A16 epilogue already:

1. Maps grouped-GEMM output rows through scatter row IDs.
2. Multiplies the FP32 accumulator by the route weight.
3. Stores BF16 vectors.
4. Counts completed output tiles per expert.

Current source:

```text
/home/azhpcuser/mai/yolo/mai_kernels/csrc/w8a16_grouped_gemm/
  cutlass_ext/cutlass/epilogue/collective/
  sm100_epilogue_array_nosmem_rank4.hpp
```

Add an optional direct-rank scatter mode:

```text
local scatter:
  output_base + token_id * ld + hidden_offset

direct-rank scatter:
  peer_route_bases[source_rank]
    + buffer_slot_offset
    + (source_token_idx * top_k + topk_lane) * hidden
    + hidden_offset
```

The epilogue continues applying the routing weight before the BF16 store.

Required per routed row:

```text
source_rank
source_token_idx
topk_lane
```

`source_rank` comes from `dispatch_row / max_tokens_per_rank`.
`source_token_idx` comes from the new dispatch metadata.
`topk_lane` comes from the flattened route row ID.

The preferred implementation builds a device array of 64-bit route-row base
addresses during the existing gather/populate prologue:

```text
direct_route_row_ptr[flattened_route]
```

The epilogue then performs:

```text
dst = direct_route_row_ptr[token_id] + hidden_offset
```

This keeps peer-address arithmetic out of the vector store loop.

## 10. Completion granularity

Use the existing ordered four-expert phase size for the first implementation:

```text
phase 0: experts 0..3
phase 1: experts 4..7
phase 2: experts 8..11
phase 3: experts 12..15
```

Completion state on each source rank:

```text
directRankReadyEpoch[
    buffer_slot,
    producer_rank,
    phase
] uint32
```

The destination rank also builds:

```text
phaseSourceMask[phase]
```

Bit `r` is set when that producer phase writes at least one route to source
rank `r`. Empty source/phase pairs receive no notification.

## 11. Completion publication without a persistent controller

Do not launch a persistent MSCCL++ data/control kernel.

Each expert's final tile:

1. Waits for its epilogue stores to be issued.
2. Participates in the per-expert tile-completion atomic chain.
3. Publishes `expertReadyEpoch[expert]`.

The final expert CTA that observes all experts in its phase ready:

1. Claims the phase with an epoch-tagged CAS.
2. Executes the required system-scope fence.
3. Iterates `phaseSourceMask[phase]`.
4. System-release stores the epoch into each source rank's mapped
   `directRankReadyEpoch[slot, producer_rank, phase]`.

This reuses real GEMM2 CTAs and reserves no SM for a side kernel.

## 12. Required memory ordering

The required visibility chain is:

```text
peer route stores from every expert tile
  -> epilogue CTA barrier
  -> acq_rel expert tile counter chain
  -> device-release expert-ready epoch
  -> phase publisher device-acquire
  -> system fence/release
  -> peer ready epoch system-release store
  -> source system-acquire
  -> source reads local route buffer
```

The implementation must use CUDA system-scope ordering for the final remote
completion publication. A device-scope epoch is insufficient for peer-mapped
route data.

Do not publish readiness from an arbitrary tile. Expected tile counts must be
derived from the actual expert problem shape and routing imbalance.

## 13. Source-local reduction

After local GEMM2, the source rank launches one local reduction kernel on its
main stream:

```text
directRankReduceKernel(
    local_route_buffer,
    topk_ids,
    remote_ready_epochs,
    dispatch_epoch,
    output
)
```

Recommended topology:

```text
one CTA per source token
256 threads/CTA
FP32 accumulation
one BF16 output store
```

For each token:

1. Lanes 0..top-k-1 derive `(producer_rank, producer_phase)` from the original
   top-k expert IDs.
2. The control warp waits until every required epoch equals the current
   dispatch epoch.
3. The CTA reads the eight local route rows.
4. Threads sum in FP32 in top-k lane order.
5. The CTA writes one BF16 output row.

This kernel starts after local GEMM2, so it does not reserve resources from
GEMM2. It may wait for slower remote producers, but that wait occurs after the
local compute has released all SMs.

There is no separate combine stream or CUDA event join in the initial design:

```text
dispatch -> GEMM1 -> activation -> GEMM2/direct writes
         -> local wait/reduce -> next dispatch
```

## 14. Buffer reuse and CUDA Graph safety

The source rank owns reuse of its receive slots.

Invariant:

```text
source rank does not launch dispatch epoch E
until local reduction for epoch E-1 has completed
```

A producer can write source epoch `E` only after receiving that source's
dispatch metadata for epoch `E`. Therefore the source has already completed
the previous local reduction before any producer can reuse the selected slot.

Use two slots initially to make alternating graph iterations explicit.

Every metadata, route, and epoch buffer must be allocated before CUDA Graph
capture. No allocation, registration, or peer-address discovery may occur
during capture or replay.

## 15. MSCCL++ API changes

Add a new mode:

```text
CombineMode::RANK_MAJOR_DIRECT_WRITE
```

MSCCL++ changes:

```text
src/ext/ep/include/config.hpp
  add source-token metadata
  add two-slot direct route receive buffer
  add direct ready epochs
  add stable peer pointer tables

src/ext/ep/low_latency/config.cuh
  add workspace phase masks and phase-published epochs

src/ext/ep/low_latency/dispatch.cu
  publish source_token_idx for each compact destination row
  build per-phase source masks

include/mscclpp/ext/ep/types.hpp
  define the direct-rank context/accessors

src/ext/ep/runtime/fixed_buffer.cc
  allocate and expose direct-rank buffers/context

src/ext/ep/bindings.cpp
  expose context tensors/pointers

python/mscclpp/ep/{types.py,low_latency.py,communicator.py}
  expose the new mode and context
```

In direct-write mode, `communicator.combine()` launches only the source-local
wait/reduce kernel. It does not send or pull route data.

## 16. MAI API changes

Extend `W8A16_GroupedGEMM` with one optional public data bundle containing:

```text
peer_route_bases
peer_ready_bases
source_token_indices
phase_source_masks
dispatch_epoch
num_ranks
max_tokens_per_rank
expert_phase_size
```

Supply the bundle only for GEMM2.

MAI changes:

```text
mai_kernels/src/mai_kernels/w8a16_grouped_gemm.py
  validate the optional direct-rank tensors

mai_kernels/csrc/w8a16_grouped_gemm/
  plumb the context into CUTLASS arguments
  build direct route-row pointers in gather/populate

cutlass_ext/.../sm100_epilogue_array_nosmem_rank4.hpp
  add direct-rank vector stores
  publish phase completion
```

The MAI interface must remain usable without MSCCL++ installed.

## 17. Correctness invariants

1. Every valid `(source token, top-k lane)` is written exactly once.
2. No two experts write the same route row.
3. The route weight is applied once in FP32 before BF16 storage.
4. Source reduction converts BF16 routes to FP32.
5. Source reduction sums lanes in a deterministic order.
6. Final output is cast to BF16 once.
7. A source never reads a route before its producer phase epoch.
8. A producer never overwrites a source buffer slot before the source reuses
   that epoch slot.
9. Empty experts and invalid/padding rows publish completion without writing
   route data.
10. Epoch comparison, not zero initialization, determines readiness.

Reference numerics:

```text
GEMM2 FP32 accumulator
  -> multiply route weight in FP32
  -> BF16 route store
  -> source BF16-to-FP32 conversion
  -> FP32 top-k sum
  -> BF16 final store
```

## 18. Performance model

For 64 tokens/rank:

```text
network payload/rank = 4 MiB
wire floor at 660 GB/s = about 6.4 us
local reduction read = 4 MiB
local output write = 0.5 MiB
```

The previous push experiment was slower because it materialized producer
output and used additional producer/consumer work. This design is worth
retesting only because it:

1. Fuses the network write into GEMM2.
2. Removes the remote-pull kernel.
3. Removes the 17-SM reservation.
4. Removes the GEMM release wait.

Expected costs:

```text
GEMM2 remote-store slowdown      target <= 5 us
post-GEMM local wait/reduction   target <= 5 us after last producer
```

The design should be abandoned if peer stores slow GEMM2 by more than the
serial direct-pull combine it replaces.

## 19. Implementation sequence

### Phase 1: metadata and local emulation

1. Add `source_token_idx` metadata.
2. Build route-row addresses targeting a local test buffer.
3. Make GEMM2 scatter into `[token, top_k, hidden]`.
4. Run the source-local reduction.
5. Verify serial parity without peer writes.

### Phase 2: synchronous peer writes

1. Replace local route bases with peer-mapped route bases.
2. Synchronize all ranks after GEMM2.
3. Run local reduction.
4. Verify 4- and 8-GPU correctness.
5. Measure isolated GEMM2 slowdown from peer stores.

### Phase 3: epoch ordering

1. Add expert and phase completion.
2. Remove the global post-GEMM synchronization.
3. Make local reduction wait only for its required producer phases.
4. Stress epoch reuse over long CUDA Graph replays.

### Phase 4: performance tuning

1. Compare cached, no-allocate, and write-through peer store policies.
2. Tune route-row pointer layout.
3. Tune local reduction CTA size.
4. Compare two- and four-expert completion phases.
5. Test 32, 64, and 128 tokens/rank.

## 20. Acceptance criteria

Correctness:

```text
graph/eager max abs = 0
serial/direct-write difference within established BF16 tolerance
no stale rows over at least 100,000 graph iterations
```

Performance at 8 GPUs, T=64, H=4096, I=6656, 16 local experts:

```text
direct-write E2E < serial direct-pull E2E
no persistent SM reservation
no standalone remote data-movement kernel
local reduction tail <= 5 us after last required producer
```

Scale:

```text
4 GPUs: correctness and debugging
8 GPUs: primary performance gate
16 GPUs: communication scaling
32 GPUs: production validation
```

## 21. Primary risks

1. Peer epilogue stores may reduce GEMM2 HBM efficiency.
2. Small vector stores may not combine efficiently across NVLink.
3. System-scope publication may be more expensive than expected.
4. A slow producer still determines the source's final reduction start.
5. Pointer-array setup may add gather/populate overhead.
6. Incorrect transitive ordering can expose partially written remote rows.
7. Buffer-slot reuse can corrupt long CUDA Graph runs if epoch ownership is
   not enforced.

These risks must be measured independently before optimizing the complete
pipeline.
