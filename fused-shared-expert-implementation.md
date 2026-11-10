# Fused v2 (shared expert inside the MegaMoE kernel)

`--fused-shared-expert` runs the shared expert inside the routed W4A8 kernel (`runFusedW4A8` in
`src/ext/megamoe/megamoe_w4a8.cu`). Goal: routed + shared ≈ routed-only, so the shared expert is fully hidden.

## Result

EP32 (8 nodes × 4 GB300), 16 balanced tokens/rank, routed H4096/I4352 MXFP4 × static FP8, k256/ls4, graph replay,
`latency_us_max_across_ranks` median. The shared expert is fp8_nv weights with static FP8 activations.

| Shared H / I | Fused v2 | Routed-only | v2 delta | Fused v1 delta (start) |
|---|---|---|---|---|
| 8704 / 2048 (model) | 103.9 | 101.0 | +2.9 | +14.4 |
| 4096 / 2048 | 103.1 | 101.1 | +2.0 | — |
| 4096 / 128 | 102.2 | 100.5 | +1.7 | — |

The routed-only time is unchanged by these edits. It runs the same `megaMoeW4A8` routed path, which shares only:
- the `TaskRange` cluster-subset defaults (inactive for routed-only);
- the route-subpass skip, which never triggers with 512 routing threads.

`--check` passes at 8704/2048 and 4096/128.

## Approaches tried (summary)

Every variant was built with compile-time switches, run with `--check`, then timed twice against the routed control
in the same benchmark run. The deltas below are fused − routed at 8704/2048.

| # | Approach | Delta (us) | Kept? |
|---|---|---|---|
| 0 | Fused v1 (other session): routing on 4 dispatch warps at 32 registers; shared FC1 on clusters 0–15; shared FC2 before the final grid barrier; memset + 2 quantize graph nodes. Measured cost: ~4 graph nodes, ~6 routing on 4 warps (stack spill), ~2.5 drains, ~1.8 shared GEMM | +14.4 | baseline |
| 1 | Diagnostic: all 16 warps route before the role split (`FULL_ROUTING`) | ≈ +19 (120.3 us; shared FC1 no longer overlaps routing) | no |
| 2 | v2 base: one prologue kernel; 12-warp subset routing at entry registers; early grid arrive; shared FC2 after routed work on non-combine clusters with one routed task fewer | +5.8 | yes |
| 3 | Skip builds (no shared FC1 / no shared FC2 / neither) to attribute cost. On the +4.0 version: structure 1.7, shared FC1 1.3, shared FC2 1.0. FC1-only skip hangs by design, because FC2 waits on `hiddenReady` | — | diagnostic |
| 4 | CTA-local `routingReady` mbarrier instead of polling a global epoch flag | +4.5 | yes |
| 5 | Shared FC2 only after the whole grid finished routed work (HBM idle during join/combine) | +5.1 | no |
| 6 | Shared FC1 on clusters 2..17 (off cluster 0, which publishes the rank's routes) | +4.0 | yes |
| 7 | Shared FC1 on clusters 30..45 | +4.2 | no |
| 8 | Epilogue warps route before the TMEM allocation; MMA warp publishes `tmemReady` itself | +4.0 (trace: routing ends 0.7 us earlier) | yes |
| 9 | Remove the global `routingReadyEpoch` release store; epilogue warps use a non-blocking init barrier (`NamedBarrier::arrive`) | +3.9 | yes |
| 10 | Barrier init moved to the spare warp 7, so the epilogue warps start routing at kernel entry | +3.55 | yes |
| 11 | Delay the shared FC1 start (time scan, all clusters): 1500 / 2000 / 2500–3000 / 4000 / 6000 ns | +2.93 / +2.78 / +2.76–2.83 / +3.0 / +5.35 | yes, 2500 |
| 12 | Same delay but only on shared FC1 clusters (runtime parameter) | +3.2–3.3 | no (all-cluster delay is better) |
| 13 | Nanosleep backoff for idle compute warps during routing (with and without delay) | no measurable gain | no |
| 14 | Final (2 + 4 + 6 + 8 + 9 + 10 + 11, delay on all load warps, 2500 ns) | +2.86 / +2.95 | final |

## Detailed final design and why each step helped

**How the cost was attributed.** Three tools were used:
- An in-kernel trace build (`MSCCLPP_MEGAMOE_W4_TRACE=1`, dumped with `MEGAMOE_KTRACE_DIR`) gave per-warp phase
  timestamps (routing begin/end, loads, dispatch, join, combine).
- torch-profiler kernel durations separated kernel time from graph-node time.
- "Skip" builds removed shared FC1 and/or FC2 to split the delta.

The trace build perturbs absolute timing by ~9 us, so it was used only for relative comparisons. Every decision was
confirmed with non-trace timing.

Facts that drove the design:
- **The routed kernel is bound by its HBM weight stream.** It streams ~428 MB of MXFP4 weights per rank: 16 local
  experts × (17.8 MB FC1 + 8.9 MB FC2) in ~85 us, about 5 TB/s. The shared expert adds 53 MB (FC1 35.6 MB + FC2
  17.8 MB fp8). It can only be hidden where HBM is underused: during routing (~9 us, latency-bound) and on clusters
  that get one routed task fewer.
- **The static round-robin schedule leaves slack on some clusters.** 800 routed tasks over 76 clusters gives 40
  clusters 11 tasks and 36 clusters 10 tasks. One routed task ≈ 8 us, so the 10-task clusters have ~8 us of free
  time.
- **Routing is latency-bound.** It is a cross-rank exchange plus two grid barriers. Anything that delays its start
  or adds memory-latency pressure lands directly on the critical path.

**Final schedule, per CTA (16 warps):**

1. **Launch prologue (`megamoe_launch.cu`).** A single `quantizeFusedInputsStaticKernel` does three jobs: it
   quantizes the routed input, quantizes the shared input (when the hidden sizes differ), and zeroes the shared FC1
   `hiddenReady` counters.
   - Before: a memset node + 2 quantize kernels, which delayed the MegaMoE kernel start by ~4–6 us.
2. **Kernel entry.**
   - Warp 7 (the spare warp of the MMA/load warpgroup) initializes the fused mbarriers (`tmemReady`,
     `computeReady`, `phaseReady`, `routingReady`). It also acts as the accumulator pipeline's
     `initializingWarp`.
   - Epilogue warps 0–3 do no initialization. They `NamedBarrier::arrive` (non-blocking) and
     `cluster_arrive_relaxed`, then route immediately.
   - Dispatch/idle warps 8–15 route immediately.
   - Result: all 12 routing warps start together, like the routed-only kernel. Routing at entry registers removed
     v1's stack spill (STACK 120 → 8).
3. **Subset routing (`megamoe_routing.cuh`, `DispatchOnly` mode, used only by the fused kernel).**
   - Participants are warps [0, EpilogueEnd) ∪ [DispatchBegin, 16), synchronized with named barrier 3; the routing
     grid barrier is driven by routing thread 0.
   - Later subpasses skip warps that lie past the live routes.
   - No global "routing done" flag is published.
4. **Shared FC1 overlaps routing.**
   - The MMA/load warpgroup of clusters 2..17 runs the 16 shared FC1 tasks (H8704 × 2·I2048) into TMEM.
   - Clusters 0–1 are excluded: CTA 0 publishes the rank's source routes (critical path), and those clusters do the
     combine.
   - Routed ordinals start right after shared FC1, so the shared FC1 clusters fall in the 10-task group and FC1 uses
     their slack.
   - The load warps first wait `sharedFc1DelayNs` (default 2500 ns, env `MSCCLPP_MEGAMOE_FUSED_FC1_DELAY_NS`).
     Starting the 36 MB weight stream at t=0 slowed the routing exchange by ~1.5–2 us (memory-latency contention).
     Starting ~2.5 us later still finishes FC1 within those clusters' slack. Delaying all load warps measured ~0.4 us
     better than delaying only the FC1 clusters.
5. **Routing → routed work.**
   - After routing, epilogue thread 0 does `cluster_wait`, which makes every barrier init visible, then arrives on
     the CTA-local `routingReady` mbarrier.
   - Load/MMA warps wait on it after `syncFusedCompute`. This replaced polling a global epoch written by block 0,
     which cost ~1 us of propagation plus a release fence.
   - The MMA warp publishes the TMEM allocation itself (`__syncwarp` + `tmemReady.arrive`), so the epilogue warps
     never join a named barrier with the MMA warp before routing.
   - The epilogue warps then drain shared FC1 from TMEM (SiLU·up, FP8 store, `hiddenReady` release). After a
     cluster-wide `syncFusedCompute`, every role switches to the routed W4A8 task range.
6. **Early grid arrive (`megamoe_epilogue.cuh`).**
   - When a CTA's routed epilogue stores land (`bulkStoreWait` + epilogue barrier), thread 0 calls
     `arriveResidentGrid` and keeps the phase.
   - The peer signal and combine therefore never wait for shared FC2.
7. **Shared FC2 in the tail.**
   - It runs only on clusters without combine work (`combineCtaCount` mirrors `hasCombineWork`).
   - Its ordinals continue after the routed range (`TaskRange` cluster subset), so they land first on the clusters
     that got one routed task fewer.
8. **Finish.** `finishAndCombineArrived`:
   - non-combine CTAs exit;
   - combine CTAs wait on the stored grid phase, then signal peers and combine (unchanged combine code).

**Where the last ~2.9 us goes** (skip builds plus trace):
- ~1 us: shared FC1 HBM traffic during routing.
- ~1 us: shared FC2 weight reads contending with the routed tail.
- ~1 us: structure (12 instead of 16 routing warps, a slightly later routing start, a ~0.06 us larger prologue
  kernel).

Ideas not yet tried:
- L2-prefetch the shared FC2 weights with an evict-last hint during the low-HBM routing/dispatch window. Routed
  weights already use EVICT_FIRST, so they would not evict them.
- Split shared FC1 across more clusters with split-K to lower its per-cluster duration.

## Files touched (fused path only)

| File | Change |
|---|---|
| `megamoe_w4a8.cu` | `runFusedW4A8` schedule, `routingReady`, init warp, FC1 placement and delay |
| `include/megamoe_routing.cuh` | subset routing helpers; subpass skip; no epoch publish |
| `include/megamoe_epilogue.cuh` | `signalAndCombine`, `combineCtaCount`, `arriveResidentGrid`, `waitResidentGrid`, `finishAndCombineArrived` |
| `include/megamoe_tasks.cuh` | `TaskRange` cluster subset + `taskOrdinalStride` (task loops in `megamoe_mma.cuh`, `megamoe_w4a8_roles.cuh`, `megamoe_epilogue.cuh`) |
| `include/megamoe_mma.cuh` | `makeAccumulatorPipeline(..., initializingWarp)`, default unchanged |
| `include/megamoe_device.cuh` | `routingReady` barrier in `W4A8FusedSharedStorage`; `W4A8FusedParameters::sharedFc1DelayNs`; removed `Control::routingReadyEpoch` |
| `megamoe_launch.cu` | `quantizeFusedInputsStaticKernel`; delay env read |
| `python/mscclpp/ext/megamoe/benchmark.py` | `MEGAMOE_KTRACE_DIR` in-kernel trace dump |

The full diff against the base (including the fused v1 work) is
`/home/caiorocha/test/run-dg-mscclpp/native-megamoe-shared-tune-20261105T095309Z/fused-analysis/fused-v2-final.patch`.

Source worktree with the change: `/home/caiorocha/test/run-dg-mscclpp/native-megamoe-shared-tune-20261105T095309Z/source-fusedx`.


## Sweep: batch sizes 1–64 × routing patterns (fused v2)

Setup:
- EP32, routed H4096/I4352 MXFP4 × static FP8, shared H8704/I2048 fp8_nv, graph replay, median latency (max
  across ranks).
- Both columns use the same builtin k256/ls4 kernel. "Routed only" is the routed-control graph in the same run.
- Tuned per case: `--sm-margin` {0, 4, 8} × `MSCCLPP_MEGAMOE_FUSED_FC1_DELAY_NS` {0, 2500, 5000}, i.e. 9 runs per
  case, 315 runs in total. Routed-only best is the best margin, averaged over its 3 runs.
- The fused path currently accepts only the builtin kernel, so the JIT routed policies (`k128-*`) can't be used with
  it yet.

| Routing | Tokens | Routed only, best (margin) | Routed + shared, best (margin, FC1 delay) | Delta vs best routed | Routed, same margin | Delta, same margin | Default (m0, 2500 ns) delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 1 | 70.29 (0) | 72.88 (4, 2500) | +2.59 (+3.7%) | 70.52 | +2.36 | +2.96 |
| balanced | 2 | 83.79 (0) | 86.70 (4, 2500) | +2.91 (+3.5%) | 83.96 | +2.75 | +3.48 |
| balanced | 4 | 97.33 (4) | 100.43 (4, 0) | +3.10 (+3.2%) | 97.33 | +3.10 | +2.86 |
| balanced | 8 | 98.92 (4) | 101.96 (0, 2500) | +3.04 (+3.1%) | 99.26 | +2.69 | +2.66 |
| balanced | 16 | 100.62 (4) | 103.90 (0, 2500) | +3.29 (+3.3%) | 101.04 | +2.86 | +2.88 |
| balanced | 32 | 101.20 (4) | 104.51 (4, 0) | +3.31 (+3.3%) | 101.20 | +3.31 | +3.30 |
| balanced | 64 | 103.63 (4) | 107.78 (4, 2500) | +4.15 (+4.0%) | 103.63 | +4.15 | +4.40 |
| hot_experts | 1 | 60.09 (8) | 62.12 (4, 2500) | +2.03 (+3.4%) | 60.18 | +1.95 | +2.24 |
| hot_experts | 2 | 77.24 (4) | 80.18 (8, 2500) | +2.95 (+3.8%) | 77.30 | +2.88 | +2.97 |
| hot_experts | 4 | 78.90 (0) | 82.09 (0, 0) | +3.19 (+4.0%) | 78.90 | +3.19 | +4.32 |
| hot_experts | 8 | 80.35 (0) | 84.13 (0, 0) | +3.78 (+4.7%) | 80.35 | +3.78 | +6.25 |
| hot_experts | 16 | 83.42 (0) | 86.99 (0, 0) | +3.57 (+4.3%) | 83.42 | +3.57 | +5.32 |
| hot_experts | 32 | 87.00 (4) | 92.23 (4, 2500) | +5.23 (+6.0%) | 87.00 | +5.23 | +5.38 |
| hot_experts | 64 | 91.94 (0) | 98.58 (0, 5000) | +6.64 (+7.2%) | 91.94 | +6.64 | +7.08 |
| lognormal | 1 | 79.74 (0) | 83.63 (0, 0) | +3.89 (+4.9%) | 79.74 | +3.89 | +4.71 |
| lognormal | 2 | 98.44 (0) | 101.57 (0, 2500) | +3.13 (+3.2%) | 98.44 | +3.13 | +3.10 |
| lognormal | 4 | 98.08 (4) | 100.93 (0, 2500) | +2.85 (+2.9%) | 98.10 | +2.83 | +2.77 |
| lognormal | 8 | 98.62 (4) | 101.65 (0, 2500) | +3.03 (+3.1%) | 98.99 | +2.66 | +2.66 |
| lognormal | 16 | 100.63 (4) | 103.88 (0, 2500) | +3.25 (+3.2%) | 100.93 | +2.94 | +2.94 |
| lognormal | 32 | 101.89 (0) | 105.41 (0, 2500) | +3.52 (+3.5%) | 101.89 | +3.52 | +3.49 |
| lognormal | 64 | 105.96 (0) | 112.88 (0, 2500) | +6.92 (+6.5%) | 105.96 | +6.92 | +6.83 |
| zipf | 1 | 71.76 (4) | 74.37 (4, 2500) | +2.61 (+3.6%) | 71.76 | +2.61 | +3.19 |
| zipf | 2 | 85.15 (4) | 87.52 (4, 2500) | +2.36 (+2.8%) | 85.15 | +2.36 | +2.81 |
| zipf | 4 | 84.95 (4) | 87.21 (4, 2500) | +2.26 (+2.7%) | 84.95 | +2.26 | +2.52 |
| zipf | 8 | 85.35 (4) | 87.90 (4, 2500) | +2.55 (+3.0%) | 85.35 | +2.55 | +2.87 |
| zipf | 16 | 87.33 (4) | 90.28 (4, 2500) | +2.95 (+3.4%) | 87.33 | +2.95 | +2.94 |
| zipf | 32 | 89.22 (4) | 92.23 (4, 2500) | +3.01 (+3.4%) | 89.22 | +3.01 | +3.16 |
| zipf | 64 | 94.48 (4) | 101.24 (8, 5000) | +6.76 (+7.2%) | 94.86 | +6.38 | +7.20 |
| zipf_skewed | 1 | 79.24 (0) | 83.20 (0, 0) | +3.96 (+5.0%) | 79.24 | +3.96 | +5.23 |
| zipf_skewed | 2 | 80.34 (0) | 83.53 (0, 0) | +3.18 (+4.0%) | 80.34 | +3.18 | +5.14 |
| zipf_skewed | 4 | 80.95 (0) | 84.04 (0, 0) | +3.09 (+3.8%) | 80.95 | +3.09 | +4.88 |
| zipf_skewed | 8 | 83.49 (0) | 86.50 (0, 0) | +3.01 (+3.6%) | 83.49 | +3.01 | +3.24 |
| zipf_skewed | 16 | 87.47 (0) | 91.90 (0, 0) | +4.44 (+5.1%) | 87.47 | +4.44 | +4.70 |
| zipf_skewed | 32 | 100.31 (0) | 107.90 (0, 2500) | +7.59 (+7.6%) | 100.31 | +7.59 | +7.52 |
| zipf_skewed | 64 | 126.34 (0) | 134.01 (0, 2500) | +7.67 (+6.1%) | 126.34 | +7.67 | +7.75 |


### Findings

- **Small/medium batches (1–32 tokens), most patterns:** about +2.3 to +3.5 us (≈3%) over the best routed-only time.
  This matches the 16-token result.
- **Large or skewed loads:** the overhead grows:
  - +4.2 us at balanced 64;
  - +5.2 to +6.6 us at hot_experts 32/64;
  - about +6.8 us at lognormal/zipf 64;
  - about +7.6 us at zipf_skewed 32/64.

  Here the routed task count per cluster is no longer near-uniform, so the static placement no longer matches the
  idle time:
  - shared FC1 always runs on clusters 2..17;
  - shared FC2 goes to the clusters with the next task ordinals;
  - but the clusters that are actually idle are wherever the hot experts did not land.

  This is the case where a dynamic scheduler (or load-aware placement of the shared tasks) would help.
- **Tuning matters for the skewed patterns.**
  - With FC1 delay 0, hot_experts 4–16 and zipf_skewed 1–16 run 1–2.5 us faster than with the default 2500 ns.
    Example: hot_experts 8 drops from +6.25 to +3.78.
  - Balanced, lognormal and zipf prefer 2500 ns.
  - `--sm-margin` 4 helps a few tenths of a microsecond in several cases.
  - 5000 ns is rarely best.
- **Against the earlier two-kernel overlap tuning (`bt-final-table.md`, tokens 8–64):** fused v2 is 1.9–6.4 us faster in
  absolute routed + shared time in 18 of 20 cases. It is about 0.6 us slower at zipf 64 and zipf_skewed 32.
  Caveat: the routed-only baselines differ slightly between the two builds.
- **`--check`:**
  - Passes at 1–32 tokens for all patterns.
  - At 64 tokens, 61 of 557056 shared-output elements (one token row on one rank) exceed the absolute tolerance:
    0.021 against 0.01 allowed.
  - The failure is identical across repeats and across fused v2, v7 and the final build. That points to a
    deterministic FP8 rounding difference in the intermediate hidden, not a race. It has not been checked against the
    two-kernel shared path at 64 tokens.

Raw results are `fused-analysis/results/sw-tune-*.json` in the run directory. The sweep driver is
`fused-analysis/sweep/sweep.py`, and `table.py` builds this table.


## Large batches: 128 and 256 tokens per rank

Same setup and tuning as the sweep above (margin {0,4,8} × FC1 delay {0,2500,5000}). The 64-token rows are repeated
for reference.

| Routing | Tokens | Routed only, best (margin) | Routed + shared, best (margin, FC1 delay) | Delta vs best routed | Routed, same margin | Delta, same margin | Default (m0, 2500 ns) delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 64 | 103.63 (4) | 107.78 (4, 2500) | +4.15 (+4.0%) | 103.63 | +4.15 | +4.40 |
| balanced | 128 | 106.53 (4) | 112.65 (4, 0) | +6.12 (+5.7%) | 106.53 | +6.12 | +6.17 |
| balanced | 256 | 125.32 (0) | 151.32 (0, 2500) | +25.99 (+20.7%) | 125.32 | +25.99 | +26.01 |
| hot_experts | 64 | 91.94 (0) | 98.58 (0, 5000) | +6.64 (+7.2%) | 91.94 | +6.64 | +7.08 |
| hot_experts | 128 | 101.80 (4) | 111.04 (0, 5000) | +9.24 (+9.1%) | 101.85 | +9.19 | +9.89 |
| hot_experts | 256 | 123.29 (0) | 151.51 (0, 0) | +28.22 (+22.9%) | 123.29 | +28.22 | +28.34 |
| lognormal | 64 | 105.96 (0) | 112.88 (0, 2500) | +6.92 (+6.5%) | 105.96 | +6.92 | +6.83 |
| lognormal | 128 | 121.96 (0) | 131.24 (0, 0) | +9.28 (+7.6%) | 121.96 | +9.28 | +9.63 |
| lognormal | 256 | 154.10 (0) | 180.45 (0, 2500) | +26.35 (+17.1%) | 154.10 | +26.35 | +26.38 |
| zipf | 64 | 94.48 (4) | 101.24 (8, 5000) | +6.76 (+7.2%) | 94.86 | +6.38 | +7.20 |
| zipf | 128 | 104.84 (4) | 115.68 (0, 5000) | +10.83 (+10.3%) | 105.15 | +10.53 | +10.63 |
| zipf | 256 | 127.98 (0) | 153.58 (0, 2500) | +25.60 (+20.0%) | 127.98 | +25.60 | +25.59 |
| zipf_skewed | 64 | 126.34 (0) | 134.01 (0, 2500) | +7.67 (+6.1%) | 126.34 | +7.67 | +7.75 |
| zipf_skewed | 128 | 191.52 (0) | 200.46 (0, 2500) | +8.94 (+4.7%) | 191.52 | +8.94 | +9.00 |
| zipf_skewed | 256 | 330.28 (0) | 355.40 (0, 2500) | +25.12 (+7.6%) | 330.28 | +25.12 | +25.09 |

Reference for balanced routing, using the old two-kernel overlap with the same build. The shared expert runs as its
own kernel on N CTAs; routed-control uses `--sm-margin N`.

| Tokens | Shared CTAs | Two-kernel routed + shared | Shared alone | Fused v2 (best) | Routed-only (best) |
|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 115.69 | 55.3 | **112.65** | 106.53 |
| 128 | 32 | 120.86 | 37.7 | | |
| 256 | 16 | **144.58** | 88.3 | 151.32 | 125.32 |
| 256 | 32 | 150.34 | 53.5 | | |

The `--check` at 64/128/256 tokens shows one row on one rank over the absolute tolerance (≈0.02 against 0.01). The
two-kernel shared path at 64 tokens produces the identical mismatch (same rank, row and value), so it comes from the
shared-expert FP8 reference tolerance, not from fusion.

### Why the gap jumps at 256 tokens

- **The shared work doubles at once.** A task covers at most `TileN = 128` tokens, so at 256 tokens the shared expert
  has 2 token blocks: 32 FC1 tasks and 68 FC2 tasks instead of 16 and 34. Each token block re-streams the whole
  shared weight matrix, so shared HBM traffic goes from 53.5 MB to 107 MB.
  - Routed balanced load is 256 × 7 × 32 / 512 = 112 rows per expert. That is still one block, so there are still 800
    routed tasks.
- **The slack disappears.**
  - At 16–128 tokens: 816 FC1 + routed tasks on 76 clusters leaves 20 clusters with one task fewer, which absorbs the
    first 20 of the 34 FC2 tasks.
  - At 256 tokens: 832 tasks leave only 4 such clusters, and 68 FC2 tasks have to go somewhere. That is roughly one
    extra full task round on most clusters.
- **Routed tasks become more compute-heavy.** At ~112 tokens per block, an N128 task costs more than at 16 tokens, so
  each extra round is ~12+ us, not ~8.
- **The two-kernel overlap scales better here (144.6 vs 151.3).** It keeps its own 16 CTAs streaming the shared
  weights for the whole routed duration. The fused kernel instead stacks whole extra tasks onto routed clusters.
- **Skewed patterns (hot_experts, zipf, zipf_skewed at 128 tokens) are already +9–11 us.** The hot experts have more
  than 128 rows and produce extra routed blocks, which land on clusters chosen statically.

### Proposed optimizations (not implemented yet), by expected value

1. **Multi-block (N256) tasks for weights reused across token blocks.** One task loads a weight tile once and runs it
   against 2 token blocks with 2 TMEM accumulators.
   - It halves shared weight traffic at 256 tokens.
   - The same change helps hot routed experts with more than 128 rows, the skewed 128/256 cases.
   - Main constraint: TMEM columns. 2 × 128 FP32 columns per stage within 512 should fit, but needs checking against
     the current double-buffered accumulator.
2. **Dynamic task claiming:** a per-rank atomic queue over [routed…, shared FC2…].
   - The cluster lead claims one task ahead and broadcasts it through SMEM/mbarrier to its peer CTA and the roles.
   - Shared FC2 and extra hot-expert blocks then go to whichever cluster drains first. This targets the skewed rows,
     +7–11 us at 64–128 tokens.
3. **Size-aware fallback (a quick win).** When shared work exceeds the routed slack, use the two-kernel overlap,
   which is better at 256 tokens.
   - The test: `sharedFc1Tasks + routedTasks` leaves fewer clusters with a short round than there are FC2 tasks.
   - Alternatively, give the fused kernel dedicated shared clusters that stream the shared weights for the whole
     kernel, as the two-kernel version does.
4. **Smaller routed tasks or split-K at large N:** reduces the 10-vs-11 round granularity once the tasks become
   compute-heavy.


## Fused v2 vs two-kernel overlap, 1–256 tokens

All numbers come from the same build and the same routed kernel (builtin k256/ls4). They are medians, max across ranks.

- Fused v2 is tuned over margin {0,4,8} × FC1 delay {0,2500,5000}.
- Two-kernel is tuned over shared CTAs {8,16,24}, with routed `--sm-margin` = shared CTAs and the shared kernel not
  waiting for the routed start.
- Routed only is the best routed-control result from the fused sweep.
- "Delta" is the cost over routed-only.

| Routing | Tokens | Routed only (best) | Fused v2 (best) | Fused delta | Two-kernel (best, shared CTAs) | Two-kernel delta | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| balanced | 1 | 70.24 | 72.88 | +2.63 | 78.64 (24) | +8.40 | fused 5.8 |
| balanced | 2 | 83.73 | 86.70 | +2.97 | 92.20 (16) | +8.47 | fused 5.5 |
| balanced | 4 | 97.30 | 100.43 | +3.13 | 105.98 (16) | +8.68 | fused 5.5 |
| balanced | 8 | 98.88 | 101.96 | +3.07 | 107.25 (16) | +8.37 | fused 5.3 |
| balanced | 16 | 100.60 | 103.90 | +3.30 | 109.46 (16) | +8.86 | fused 5.6 |
| balanced | 32 | 101.18 | 104.51 | +3.33 | 110.22 (16) | +9.05 | fused 5.7 |
| balanced | 64 | 103.60 | 107.78 | +4.18 | 111.81 (16) | +8.21 | fused 4.0 |
| balanced | 128 | 106.46 | 112.65 | +6.18 | 115.73 (16) | +9.27 | fused 3.1 |
| balanced | 256 | 125.25 | 151.32 | +26.06 | 144.69 (16) | +19.43 | two-kernel 6.6 |
| hot_experts | 1 | 60.06 | 62.12 | +2.06 | 66.14 (24) | +6.08 | fused 4.0 |
| hot_experts | 2 | 77.22 | 80.18 | +2.96 | 84.80 (24) | +7.58 | fused 4.6 |
| hot_experts | 4 | 78.89 | 82.09 | +3.20 | 87.44 (16) | +8.55 | fused 5.4 |
| hot_experts | 8 | 80.35 | 84.13 | +3.79 | 89.08 (16) | +8.73 | fused 4.9 |
| hot_experts | 16 | 83.42 | 86.99 | +3.57 | 92.36 (16) | +8.94 | fused 5.4 |
| hot_experts | 32 | 86.94 | 92.23 | +5.29 | 94.69 (8) | +7.75 | fused 2.5 |
| hot_experts | 64 | 91.90 | 98.58 | +6.67 | 101.99 (8) | +10.09 | fused 3.4 |
| hot_experts | 128 | 101.78 | 111.04 | +9.27 | 113.56 (16) | +11.78 | fused 2.5 |
| hot_experts | 256 | 123.25 | 151.51 | +28.26 | 144.17 (16) | +20.91 | two-kernel 7.3 |
| lognormal | 1 | 79.73 | 83.63 | +3.89 | 87.91 (16) | +8.17 | fused 4.3 |
| lognormal | 2 | 98.37 | 101.57 | +3.20 | 106.39 (16) | +8.01 | fused 4.8 |
| lognormal | 4 | 98.05 | 100.93 | +2.87 | 106.39 (16) | +8.34 | fused 5.5 |
| lognormal | 8 | 98.58 | 101.65 | +3.07 | 107.00 (16) | +8.42 | fused 5.4 |
| lognormal | 16 | 100.62 | 103.88 | +3.25 | 109.56 (16) | +8.94 | fused 5.7 |
| lognormal | 32 | 101.86 | 105.41 | +3.56 | 112.54 (16) | +10.68 | fused 7.1 |
| lognormal | 64 | 105.85 | 112.88 | +7.03 | 116.32 (8) | +10.47 | fused 3.4 |
| lognormal | 128 | 121.91 | 131.24 | +9.34 | 135.21 (8) | +13.31 | fused 4.0 |
| lognormal | 256 | 154.05 | 180.45 | +26.40 | 177.23 (16) | +23.18 | two-kernel 3.2 |
| zipf | 1 | 71.75 | 74.37 | +2.62 | 79.25 (24) | +7.51 | fused 4.9 |
| zipf | 2 | 85.09 | 87.52 | +2.42 | 93.16 (16) | +8.06 | fused 5.6 |
| zipf | 4 | 84.94 | 87.21 | +2.27 | 93.38 (16) | +8.44 | fused 6.2 |
| zipf | 8 | 85.32 | 87.90 | +2.58 | 93.80 (16) | +8.48 | fused 5.9 |
| zipf | 16 | 87.32 | 90.28 | +2.97 | 96.43 (16) | +9.11 | fused 6.1 |
| zipf | 32 | 89.20 | 92.23 | +3.03 | 98.71 (16) | +9.51 | fused 6.5 |
| zipf | 64 | 94.38 | 101.24 | +6.86 | 102.68 (8) | +8.30 | fused 1.4 |
| zipf | 128 | 104.81 | 115.68 | +10.87 | 117.65 (8) | +12.84 | fused 2.0 |
| zipf | 256 | 127.96 | 153.58 | +25.62 | 150.01 (16) | +22.05 | two-kernel 3.6 |
| zipf_skewed | 1 | 79.12 | 83.20 | +4.08 | 87.65 (16) | +8.52 | fused 4.4 |
| zipf_skewed | 2 | 80.33 | 83.53 | +3.20 | 88.26 (16) | +7.93 | fused 4.7 |
| zipf_skewed | 4 | 80.89 | 84.04 | +3.15 | 89.29 (16) | +8.40 | fused 5.3 |
| zipf_skewed | 8 | 83.42 | 86.50 | +3.07 | 91.85 (16) | +8.42 | fused 5.4 |
| zipf_skewed | 16 | 87.41 | 91.90 | +4.49 | 97.68 (16) | +10.27 | fused 5.8 |
| zipf_skewed | 32 | 100.25 | 107.90 | +7.65 | 107.41 (8) | +7.16 | two-kernel 0.5 |
| zipf_skewed | 64 | 126.26 | 134.01 | +7.75 | 137.71 (8) | +11.45 | fused 3.7 |
| zipf_skewed | 128 | 191.46 | 200.46 | +9.00 | 206.89 (8) | +15.43 | fused 6.4 |
| zipf_skewed | 256 | 329.90 | 355.40 | +25.50 | 350.31 (8) | +20.41 | two-kernel 5.1 |

Findings:
- **Up to 128 tokens, fused v2 wins in 39 of 40 cases**, usually by 4–6 us. The exception is zipf_skewed 32 tokens,
  where two-kernel is ahead by 0.5 us.
  - Two-kernel costs +8 to +9 us almost everywhere at small batches. That is the reserved-SM cost: the routed kernel
    loses 16 SMs.
  - Fused costs +2 to +4 us.
- **Two-kernel degrades with batch size too.** At balanced routing it costs +8.2 us at 64 tokens, +9.3 at 128 and
  +19.4 at 256. At 256 tokens the shared expert alone needs ~88 us on 16 CTAs, so it is no longer fully hidden either.
- **At 256 tokens two-kernel wins, by 3–7 us.** It streams the 2-token-block shared weights continuously on its own
  CTAs, while fused v2 appends a full extra task round.
  - This confirms that the 256-token jump is mostly the shared work itself (2× tasks and 2× weight traffic), with the
    fused static placement making it a few us worse.
  - A size-aware switch (fused up to 128 tokens, two-kernel at 256) would be best of both today. Multi-block shared
    tasks (proposal 1 above) target the root cause.
