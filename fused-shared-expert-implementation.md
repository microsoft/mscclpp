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

## Merge with the small-batch optimization (`binyli/megamoe-opt`)

Branch `caiorocha/megamoe_shared_com_opt` merges `caiorocha/small_bs` (Binyang's optimization) into fused v2.
The **routed-only kernel is unchanged** from `small_bs`: an A/B against a `small_bs` build with the same defines gives the
same latency (16 tokens, margin 4: balanced 91.8 vs 91.7 µs, zipf_skewed 97.5 vs 97.6 µs).

### Merge decisions

- **Shared memory.** `W4A8FusedSharedStorage` wraps Binyang's `W4A8SharedStorage` at offset 0, because routing,
  dispatch and the epilogue reinterpret SMEM as `P::SharedStorage`. The fused routing scratch overlays the dispatch
  buffers and epilogue staging, which this CTA does not use until its routing finishes. The shared expert reuses the
  routed tensor buffers.
- **Routing.**
  - The routed kernel keeps Binyang's `processSourceRouteChunks` and `processReceivedRouteChunks` verbatim.
  - The fused kernel uses `*Subset` copies that run on the routing warp subset. They are selected through
    `processSourceRoutes` and `processReceivedRoutes`.
  - A first merge generalized the shared functions instead. That cost the routed-only kernel about 0.5 µs, found by
    bisection, so the routed path now uses the untouched code.
- **Task order.** `TaskWalk` reproduces Binyang's wave order for whole-grid ranges, which covers the routed tasks and
  shared FC1. It uses round-robin for cluster-subset ranges (shared FC2). Load, MMA and epilogue use the same walk.
  Binyang's readiness prefetch uses `TaskWalk::nextOrdinal`.
- **Combine.**
  - The routed kernel spreads combine over every CTA.
  - The fused kernel keeps it on the first `combineCtaCount` CTAs, so shared FC2 never sits on the combine path. That
    count now follows Binyang's `effectiveLoads` item sizing.
  - Spreading the fused combine over 2× or 4× more CTAs was tested and did not help (balanced +0.2 to +0.4 µs worse,
    zipf at 32 tokens 1.4 µs better, hot_experts at 32 tokens 1.1 µs worse). It was reverted.
- **Dispatch overlap.** Before the shared FC1 epilogue, the epilogue warps wait on Binyang's `dispatchDone` when the
  epilogue staging aliases the dispatch buffers.

### Results (k128, e16w8, LS10, dispatch warps 2, dispatch stages 1, fixed token count; best of margin {0,4,8} × FC1 delay {0,2500,5000} ns)

| Pattern | 8 tokens | 16 tokens | 32 tokens | 64 tokens |
|---|---|---|---|---|
| balanced | 91.54 → 95.92 (+4.37) | 92.33 → 96.60 (+4.27) | 92.82 → 97.35 (+4.53) | 102.49 → 107.08 (+4.59) |
| hot_experts | 74.58 → 78.40 (+3.83) | 79.38 → 83.19 (+3.80) | 94.58 → 99.85 (+5.27) | 102.18 → 106.26 (+4.08) |
| lognormal | 91.31 → 95.60 (+4.30) | 92.85 → 97.14 (+4.29) | 101.50 → 104.82 (+3.32) | 118.60 → 123.72 (+5.12) |
| zipf | 79.45 → 82.84 (+3.38) | 82.36 → 86.28 (+3.92) | 91.25 → 94.38 (+3.13) | 106.58 → 112.91 (+6.33) |
| zipf_skewed | 86.46 → 88.74 (+2.28) | 98.41 → 100.53 (+2.13) | 129.97 → 134.42 (+4.45) | 183.09 → 189.81 (+6.72) |

Each cell is routed-only → routed + shared (delta) in µs.

- **Latency.** Routed + shared is about 7 µs faster than before the merge (balanced, 16 tokens: 103.9 → 96.6 µs).
- **Overhead.** The shared-expert overhead grows by about 1 µs for balanced/lognormal: it was about +3.3 µs and is now
  about +4.3 µs.
- **Likely cause.** Binyang's routed kernel uses HBM more efficiently, so it has less idle bandwidth left to hide the
  shared expert's 53.5 MB of weights:
  - The routed weights are about 428 MB per rank, so the shared expert adds about 12.5% more bytes.
  - If the shared expert were not hidden at all, it would cost about 7–9 µs.
- **Routed-only reference.** The user's routed-only reference table took the best JIT policy per case, for example
  adaptive-cache or adaptive-expert-major for the hot and skewed patterns. With this fixed e16w8-ls10 configuration,
  routed-only at hot_experts/zipf_skewed 32–64 tokens matches `small_bs` built the same way, not that best-of table.

### Per-case config tuning (merged branch)

Fused v2 now supports Binyang's policy knobs. I rebuilt it with five configs (all k128, dispatch warps 2, dispatch stages 1, fixed token count):

| Name | Epilogue (tokens / warps / registers) | Load stages | Adaptive cache weights |
|---|---|---:|---|
| e16w8-ls10 | 16 / 8 / 160 | 10 | no |
| e16w8-ls8 | 16 / 8 / 160 | 8 | no |
| e32w4-r192 | 32 / 4 / 192 | 8 | no |
| e32w4-adaptive | 32 / 4 / 224 | 8 | yes |
| e16w8-ls10-adaptive | 16 / 8 / 160 | 10 | yes (new combination) |

`expert_major_tasks` is a no-op in the current kernel code (only reported as a flag), so `adaptive-expert-major` is the same as `adaptive-cache`.

Sweeps:
- Routed-only: margin {0, 2, 4}. All 20 cases reproduce the reference table within ±0.3 µs.
- Routed + shared: margin {0, 4, 8} × shared FC1 delay {0, 2500} ns, plus margin 2 for the best configs of each case.

"Routed-only with the routed + shared config" is a standalone routed-only run with the same build and margin as the winning fused run. The routed-only control measured inside the fused benchmark reads on average 0.6 µs slower than a standalone run (measured over 452 pairs), so the table does not use it.

| Pattern | Tokens | Your routed ref | Best routed-only (config, margin) | Best routed + shared (config, margin, FC1 delay) | Routed-only with the routed + shared config | Overhead vs same config | Overhead vs best routed |
|---|---:|---:|---|---|---:|---:|---:|
| balanced | 8 | 91.01 | 90.90 (e16w8-ls10, m4) | 95.95 (e16w8-ls10-adaptive, m4, 0 ns) | 91.17 | +4.78 | +5.05 |
| balanced | 16 | 91.81 | 91.69 (e16w8-ls10, m4) | 96.53 (e16w8-ls10, m4, 2500 ns) | 91.69 | +4.84 | +4.84 |
| balanced | 32 | 92.90 | 92.78 (e16w8-ls10-adaptive, m4) | 97.22 (e16w8-ls10-adaptive, m4, 2500 ns) | 92.78 | +4.44 | +4.44 |
| balanced | 64 | 98.33 | 98.36 (e32w4-r192, m4) | 103.19 (e32w4-adaptive, m4, 0 ns) | 98.58 | +4.60 | +4.82 |
| hot_experts | 8 | 72.67 | 72.57 (e16w8-ls10, m2) | 78.20 (e16w8-ls10, m2, 0 ns) | 72.57 | +5.63 | +5.63 |
| hot_experts | 16 | 76.21 | 76.04 (e16w8-ls8, m2) | 81.25 (e16w8-ls8, m0, 0 ns) | 76.15 | +5.10 | +5.21 |
| hot_experts | 32 | 82.83 | 82.51 (e32w4-adaptive, m4) | 88.03 (e32w4-adaptive, m2, 2500 ns) | 82.78 | +5.25 | +5.52 |
| hot_experts | 64 | 88.73 | 88.44 (e32w4-adaptive, m2) | 93.96 (e32w4-adaptive, m2, 2500 ns) | 88.44 | +5.52 | +5.52 |
| lognormal | 8 | 91.03 | 90.70 (e16w8-ls10, m4) | 95.40 (e16w8-ls10, m4, 0 ns) | 90.70 | +4.71 | +4.71 |
| lognormal | 16 | 92.52 | 92.23 (e16w8-ls10, m4) | 97.14 (e16w8-ls10, m4, 2500 ns) | 92.23 | +4.92 | +4.92 |
| lognormal | 32 | 96.73 | 96.74 (e16w8-ls8, m2) | 100.83 (e16w8-ls8, m4, 0 ns) | 96.84 | +3.99 | +4.08 |
| lognormal | 64 | 101.40 | 101.14 (e32w4-adaptive, m0) | 106.57 (e32w4-adaptive, m0, 2500 ns) | 101.14 | +5.43 | +5.43 |
| zipf | 8 | 78.30 | 77.89 (e16w8-ls10-adaptive, m2) | 82.81 (e16w8-ls10-adaptive, m4, 0 ns) | 77.89 | +4.91 | +4.91 |
| zipf | 16 | 81.37 | 81.04 (e16w8-ls10-adaptive, m4) | 86.28 (e16w8-ls10, m4, 2500 ns) | 81.27 | +5.01 | +5.24 |
| zipf | 32 | 82.70 | 82.70 (e16w8-ls8, m4) | 87.92 (e16w8-ls8, m4, 2500 ns) | 82.70 | +5.22 | +5.22 |
| zipf | 64 | 89.56 | 89.26 (e32w4-r192, m4) | 95.00 (e32w4-adaptive, m0, 2500 ns) | 89.77 | +5.23 | +5.74 |
| zipf_skewed | 8 | 75.43 | 75.34 (e16w8-ls8, m2) | 80.82 (e16w8-ls8, m0, 0 ns) | 75.34 | +5.48 | +5.49 |
| zipf_skewed | 16 | 82.94 | 82.70 (e32w4-r192, m4) | 87.80 (e32w4-adaptive, m2, 0 ns) | 83.12 | +4.68 | +5.09 |
| zipf_skewed | 32 | 96.56 | 96.22 (e32w4-adaptive, m4) | 101.76 (e32w4-adaptive, m2, 2500 ns) | 96.63 | +5.13 | +5.54 |
| zipf_skewed | 64 | 123.66 | 123.05 (e32w4-r192, m4) | 129.64 (e32w4-adaptive, m0, 2500 ns) | 123.87 | +5.78 | +6.59 |

Findings:
- With the right config per case, routed + shared adds +4.0 to +6.6 µs over the best routed-only result (median about +5.2).
- The tuned routed + shared latency is under 110 µs for every case except zipf_skewed at 64 tokens (129.6 µs, where routed-only alone is already 123 µs).
- The best fused config is usually the best routed config, or a close neighbour of it:
  - hot and skewed patterns: adaptive-cache;
  - balanced and lognormal at 8–16 tokens: e16w8-ls10.
- Fused prefers the same or a smaller SM margin than routed-only (m0–m4).

## Gap diagnosis (post-merge, ~4–5 µs)

### Method
- **Attribution build** (`-DMSCCLPP_MEGAMOE_FUSED_DEBUG_SKIP=1`, debug only, unstaged). The env `MSCCLPP_MEGAMOE_FUSED_SKIP` takes these flags:
  - 2 skips shared FC2.
  - 3 skips both shared GEMMs. Ordinals are kept, so the routed placement is unchanged.
  - +4 starts the routed ordinals at 0.
  - +8 spreads the combine over the whole grid.
- **Whole-grid trace build** (`TRACE=1 TRACE_CTAS=148 TRACE_CAPACITY=4096`), with routing sub-phase marks (`TokenReady` payload 1..5 in `prepareRoutesImpl`).
- Analyzers: `fused-analysis/{ftrace,fgap,froute}.py`.

### Attribution (real timing, margin 4, FC1 delay 2500, in-run routed control)

| Case | Full | No shared FC2 | No shared work | Of which: spread combine | Of which: routed ordinals at 0 |
|---|---|---|---|---|---|
| balanced 16 | +4.3 | +3.7 | +1.85 | −0.1 | +0.1 |
| hot_experts 8 | +2.6 | +2.3 | +1.25 | −0.1 | +1.4 (worse) |

The gap therefore splits into three parts:
- **Structure:** about 1.3–1.9 µs, present even with no shared work.
- **Shared FC1:** about 1.1–1.8 µs.
- **Shared FC2:** about 0.4–0.75 µs.

### Where the time goes (trace, balanced 16; trace µs are inflated about 1.5×)

1. **Routing is slower.** The routing end moves 8.8 → 11.6 µs. The routing sub-phases show two causes:
   - The subset routing (12 of 16 warps) adds about 0.8 µs, mostly in receiving routes (m3) and in the first grid barrier.
   - The shared FC1 weight stream adds about 1.6 µs more, mostly in the two grid barriers. Device-scope atomic and poll latency grows under 16 clusters of TMA traffic.
   - Even in routed-only, each routing grid barrier costs about 1.3 µs after the last arrival.
2. **Routed work starts later.** The first routed load moves 10.1 → 13.6 µs.
   - The 16 FC1 clusters (2–17) start routed work only at about 21.5 µs. One shared FC1 tile is 2.2 MB of fp8 at about 220 GB/s per cluster, about 1.7 routed tasks long.
   - The static wave gives those clusters one routed task fewer only when the partial wave avoids them. At 16 tokens, 800 tasks = 10 waves + 60, the partial wave wraps onto FC1 clusters 2 and 3. They do FC1 plus 11 tasks and are the rank's slowest clusters (+3 µs over the median).
   - The FC1 delay sweep confirms the FC1 clusters are near-critical. The optimum is flat over 1000–2500 ns, and every µs beyond 3000 ns adds about 1 µs.
3. **Routed tasks are not HBM-bound.** A W4 task moves about 0.5 MB in about 7 µs; the MXFP4 transform/MMA path is the limit. The shared fp8 tiles are per-SM-bandwidth-bound instead. So the hiding capacity is idle SM-time (the routing window and the tail spread), not spare HBM.
4. **The tail spread is large in routed-only too.** Per-cluster routed completion ranges about 92–106 µs (10- vs 11-task clusters, plus expert/FC1-vs-FC2 heterogeneity).
   - Shared FC2 is placed statically on clusters 8–41, regardless of which clusters finish first.
   - On the slowest rank, those FC2 tasks extend past the routed done.
5. **Combine concentration** (16 CTAs instead of the whole grid) costs only about 0.1–0.3 µs.

### Improvement plan (ordered by expected gain / effort)

| # | Change | Expected gain | Effort / risk |
|---|---|---|---|
| P1 | Dynamic (atomic) task claiming per rank for routed + shared tasks. The cluster leader claims ordinals in order (no deadlock: FC2 only waits on earlier-claimed FC1); a DSMEM mailbox feeds the load/MMA/epilogue warps of both CTAs. This absorbs the FC1 cluster lag, the partial-wave quantization, task heterogeneity and the FC2 placement. It also helps routed-only. | 1.5–3 µs on the gap, plus some for routed-only | High (task walk replaced in 3 roles) |
| P2 | Weighted static placement (cheap P1 subset): pick the routed ordinal begin at run time so the partial wave avoids the FC1 clusters, and put FC2 first on the clusters with the fewest routed tasks. | 0.5–1.5 µs (case dependent) | Low |
| P3 | Full-thread routing on clusters without shared FC1 (58 of 74; they include CTA 0 and all received-segment CTAs). The barrier protocol is compatible, so mixing per-CTA thread counts is safe. | about 0.5 µs | Medium |
| P4 | Smaller FC1 footprint per cluster: split-K × 2 (32 tasks with an fp32 fix-up by the last arrival) or FC1 on 32 clusters. This shortens the FC1 cluster lag and its interference burst. | 0.5–1 µs | Medium–High |
| P5 | Cheaper grid barriers (cluster-first hierarchical arrival, fewer polling CTAs). This helps both kernels and removes most of the FC1 sensitivity of the barriers. | 0.5–1 µs in both kernels | Medium |
| P6 | In-kernel static-FP8 quantization of the routed and shared inputs (Binyang's `fuseQuantize` is MXFP8 only). | 1–2 µs in both kernels (not the gap) | Medium |
| P7 | Let CTAs that finish FC2 join the combine (spread combine). | about 0.1–0.3 µs | Low |

### P2 result: static placement (not adopted)

- **What already holds.** The current placement already does what P2 asked:
  - The routed ordinals start right after the shared FC1 clusters, so the partial wave avoids the FC1 clusters whenever it fits (rem ≤ clusters − sharedFc1Tasks).
  - Shared FC2 starts at the first cluster without a partial-wave task.
- **What was tried.** The remaining lever was to make the FC1 clusters skip one whole routed wave, via placeholder ("hole") ordinals in wave 0. The code is kept in `fused-analysis/p2-holes/`.
  - Unconditional holes: worse in 16 of 20 cases (about +3 µs balanced, up to +7 µs).
  - Holes only when they add no wave:

    | Effect | Cases |
    |---|---|
    | Better | zipf 64 −2.4, zipf_skewed 8 −1.3, lognormal 64 −1.0 |
    | Worse | zipf 32 +4.0, hot_experts 32 +3.5, zipf_skewed 16 +2.4, hot_experts 64 +2.3 |
    | Neutral | the rest |
- **Conclusion.**
  - Placement swings the result by ±2–4 µs depending on which tasks share a wave (FC1→FC2 readiness waits, hot experts).
  - A per-case static rule cannot predict this, so the hole code was reverted.
  - The remaining placement upside needs run-time balancing (P1), or a per-case policy selected by the tuner.

### P6 result: in-kernel static-FP8 quantization (adopted, JIT toggle)

**Toggle.** `W4A8KernelConfig.fuse_static_quantize` (default `True`) → `-DMSCCLPP_MEGAMOE_W4_FUSE_STATIC_QUANTIZE=0/1`
(builtin builds: pass the define; benchmark: `--fuse-static-quantize/--no-fuse-static-quantize`). It is exported in the
JIT optimization flags (`MSCCLPP_MEGAMOE_JIT_FUSE_STATIC_QUANTIZE = 8`) and reported as
`effective_kernel_config["fuse_static_quantize"]`. With 0 the separate quantize launches are kept unchanged.

**When it applies.** Shared host/device predicate `fusesInputQuantize` (`megamoe_device.cuh`): captured fixed-token
launches, H4096, 1–64 tokens, `tokens * topK <= RouteChunk`, `ctas >= worldSize + 2 + tokens` — the same conditions as
Binyang's MXFP8 `fuseQuantize`, now also for `Fp8Static`. The fused kernel additionally requires a shared hidden size
different from the routed one (otherwise the shared expert reads the routed quantized input, so the separate launch is
kept).

**What changed.**

| Where | Change |
|---|---|
| `megamoe_routing.cuh` | `quantizeTokenStaticInRouting<DispatchOnly>`: CTAs 1..tokens quantize one token each (x · 1/scale → e4m3), looping for the 384-thread fused routing, then `fence.proxy.async.global` (peers read it with bulk copies after the routing barriers). `stageSourceRouteWeights` loops over routes for the fused routing thread count. The `fuseQuantize` condition now uses `fusesInputQuantize` and also runs in dispatch-only (fused) routing for static FP8. |
| `megamoe_w4a8.cu` (fused) | The idle init warp (warp 11) of every CTA quantizes a grid-strided slice of the shared input during the shared FC1 delay; CTA 0 also clears the shared FC1 readiness counters (previously done by the quantize kernel). Each init warp arrives once on `Control::sharedQuantizeSyncState` (own cache line, sense-reversing). Only the shared FC1 activation producer waits for the flip (its phase is read before the CTA init barrier so the flip cannot be missed); the weight producer streams FC1 weights meanwhile. The kernel takes the BF16 `input` / `sharedInput` (null = not fused). |
| `megamoe_launch.cu` | Routed and fused launches skip the quantize kernel and pass the BF16 inputs when the predicate holds. |
| JIT / bindings | `megamoe_specialization.hpp`, `megamoe_jit.{hpp,cu}`, `runtime.cc`, `megamoe.hpp`, `python/csrc/ext/megamoe.cpp`, `jit.py`, `w4a8.py`, `api.py`, `benchmark.py`, README, `test_megamoe_jit.py`. |

**Variants measured** (EP32, margin 4, FC1 delay 2500 ns, 20 cases, 2–3 reps, medians; mean over cases):

| Variant | Routed | Fused | Gap |
|---|---:|---:|---:|
| Off (separate quantize kernels) | 100.61 | 105.44 | +4.83 |
| Shared quantize by warps 8–11, 592 arrivals, all producers wait | 99.24 | 104.64 | +5.39 |
| Init warp only, 1 arrival per CTA, own cache line | 99.24 | 104.31 | +5.08 |
| + 4-deep batched shared loads | 99.24 | 104.72 | +5.48 |
| + shared quantize start delayed 1 µs | 99.25 | 104.80 | +5.56 |
| **Init warp, serial loads, only the activation producer waits (final)** | **99.24** | **104.21** | **+4.97** |

- Routed-only: −1.37 µs mean, every case improved (−1.19 to −2.02 µs; the quantize launch is gone).
- Fused: −1.23 µs mean, every case improved (−0.47 to −2.33 µs). The gap is unchanged (+0.14 µs), as predicted.
- Faster (batched) shared quantization is worse: its burst competes with the latency-bound routing exchange. Blocking
  the FC1 weight producer on the quantization is also worse.
- `--check` passes for routed and fused (8/16/32 tokens). The 64-token shared check fails on the same single element as
  with the toggle off (pre-existing tolerance issue).

Per-case table (µs):

| Case | Off routed | Off fused | On routed | On fused | Δ routed | Δ fused |
|---|---:|---:|---:|---:|---:|---:|
| balanced 8 | 91.72 | 96.63 | 90.53 | 95.92 | -1.19 | -0.71 |
| balanced 16 | 92.53 | 96.84 | 91.20 | 95.50 | -1.33 | -1.34 |
| balanced 32 | 92.96 | 97.69 | 91.71 | 97.22 | -1.25 | -0.47 |
| balanced 64 | 102.67 | 107.59 | 101.39 | 106.66 | -1.28 | -0.92 |
| hot_experts 8 | 79.02 | 81.68 | 77.64 | 80.56 | -1.38 | -1.12 |
| hot_experts 16 | 82.71 | 85.34 | 81.27 | 84.20 | -1.44 | -1.13 |
| hot_experts 32 | 94.92 | 100.21 | 93.56 | 98.66 | -1.36 | -1.55 |
| hot_experts 64 | 104.41 | 110.04 | 102.91 | 109.17 | -1.50 | -0.87 |
| lognormal 8 | 91.51 | 96.02 | 90.22 | 94.94 | -1.29 | -1.08 |
| lognormal 16 | 93.06 | 97.55 | 91.67 | 96.16 | -1.38 | -1.39 |
| lognormal 32 | 101.57 | 106.26 | 99.98 | 104.50 | -1.59 | -1.76 |
| lognormal 64 | 119.36 | 128.88 | 118.13 | 128.05 | -1.23 | -0.84 |
| zipf 8 | 79.63 | 83.32 | 78.33 | 82.09 | -1.30 | -1.23 |
| zipf 16 | 82.50 | 86.70 | 81.17 | 85.91 | -1.33 | -0.79 |
| zipf 32 | 92.32 | 96.53 | 91.00 | 95.58 | -1.32 | -0.95 |
| zipf 64 | 111.53 | 117.98 | 110.28 | 116.54 | -1.24 | -1.45 |
| zipf_skewed 8 | 87.54 | 92.22 | 86.26 | 90.68 | -1.28 | -1.54 |
| zipf_skewed 16 | 98.58 | 101.55 | 97.24 | 100.03 | -1.34 | -1.52 |
| zipf_skewed 32 | 130.12 | 135.24 | 128.73 | 133.47 | -1.39 | -1.77 |
| zipf_skewed 64 | 183.62 | 190.63 | 181.60 | 188.30 | -2.02 | -2.33 |


The table above uses one config for every case (e16w8-ls10, margin 4, FC1 delay 2500 ns), so it is not comparable
with the per-case tuned tables. The tuned comparison follows.

### P6 tuned results (per-case config re-selection)

**Method.** P6 was rebuilt on all five build configs (e16w8-ls10, e16w8-ls8, e32w4-r192, e32w4-adaptive,
e16w8-ls10-adaptive; packages `x-p6*`). Every case was swept again:
- routed-only: 5 configs × margin {0, 2, 4};
- routed + shared: 4 configs (r192 never won) × margin {0, 2, 4} × FC1 delay {0, 2500} ns.

"Before" is the pre-P6 tuned table ("Per-case config tuning"). Medians of max-across-ranks, µs.
Scripts: `fused-analysis/sweep/{tuned_cmp,p6_retune,sweep_cfg,table_p6}.py`.

| Pattern | Tokens | Best routed before | Best routed now (config, margin) | Best routed + shared before | Best routed + shared now (config, margin, FC1 delay) | Gap now vs best routed | Gap vs in-run routed control |
|---|---:|---:|---|---:|---|---:|---:|
| balanced | 8 | 90.90 | 89.77 (e16w8-ls10, m4) | 95.95 | 95.09 (e16w8-ls10, m2, 2500 ns) | +5.32 | +4.09 |
| balanced | 16 | 91.69 | 90.59 (e16w8-ls10, m4) | 96.53 | 95.31 (e16w8-ls10-adaptive, m4, 2500 ns) | +4.72 | +3.89 |
| balanced | 32 | 92.78 | 91.66 (e16w8-ls10-adaptive, m4) | 97.22 | 96.81 (e16w8-ls10-adaptive, m2, 2500 ns) | +5.16 | +4.69 |
| balanced | 64 | 98.36 | 97.14 (e32w4-r192, m4) | 103.19 | 102.16 (e32w4-adaptive, m0, 2500 ns) | +5.02 | +5.23 |
| hot_experts | 8 | 72.57 | 71.30 (e16w8-ls10, m2) | 78.20 | 77.52 (e16w8-ls10, m0, 0 ns) | +6.23 | +4.14 |
| hot_experts | 16 | 76.04 | 75.00 (e16w8-ls8, m2) | 81.25 | 81.58 (e16w8-ls8, m2, 2500 ns) | +6.58 | +5.22 |
| hot_experts | 32 | 82.51 | 81.06 (e32w4-adaptive, m4) | 88.03 | 86.70 (e32w4-adaptive, m4, 2500 ns) | +5.64 | +4.61 |
| hot_experts | 64 | 88.44 | 86.64 (e32w4-adaptive, m2) | 93.96 | 93.05 (e16w8-ls8, m0, 2500 ns) | +6.41 | +5.25 |
| lognormal | 8 | 90.70 | 89.64 (e16w8-ls10, m4) | 95.40 | 94.68 (e16w8-ls10-adaptive, m4, 0 ns) | +5.04 | +4.43 |
| lognormal | 16 | 92.23 | 91.00 (e16w8-ls10, m4) | 97.14 | 96.12 (e16w8-ls10, m2, 2500 ns) | +5.12 | +3.95 |
| lognormal | 32 | 96.74 | 95.79 (e16w8-ls8, m2) | 100.83 | 100.10 (e16w8-ls8, m2, 2500 ns) | +4.31 | +4.44 |
| lognormal | 64 | 101.14 | 99.67 (e32w4-adaptive, m0) | 106.57 | 106.97 (e32w4-adaptive, m4, 2500 ns) | +7.30 | +5.32 |
| zipf | 8 | 77.89 | 76.77 (e16w8-ls10-adaptive, m4) | 82.81 | 81.91 (e16w8-ls10-adaptive, m4, 0 ns) | +5.14 | +3.82 |
| zipf | 16 | 81.04 | 79.94 (e16w8-ls10-adaptive, m4) | 86.28 | 85.44 (e16w8-ls10-adaptive, m4, 2500 ns) | +5.50 | +4.40 |
| zipf | 32 | 82.70 | 81.78 (e16w8-ls8, m4) | 87.92 | 86.98 (e16w8-ls8, m4, 2500 ns) | +5.20 | +4.18 |
| zipf | 64 | 89.26 | 87.55 (e32w4-adaptive, m2) | 95.00 | 94.28 (e32w4-adaptive, m4, 2500 ns) | +6.72 | +6.04 |
| zipf_skewed | 8 | 75.34 | 74.41 (e16w8-ls8, m2) | 80.82 | 81.92 (e16w8-ls8, m0, 2500 ns) | +7.52 | +5.97 |
| zipf_skewed | 16 | 82.70 | 81.27 (e32w4-adaptive, m4) | 87.80 | 86.70 (e32w4-adaptive, m2, 2500 ns) | +5.43 | +4.04 |
| zipf_skewed | 32 | 96.22 | 94.70 (e32w4-adaptive, m4) | 101.76 | 101.07 (e32w4-adaptive, m0, 2500 ns) | +6.36 | +5.55 |
| zipf_skewed | 64 | 123.05 | 120.80 (e32w4-r192, m4) | 129.64 | 129.40 (e32w4-adaptive, m2, 2500 ns) | +8.60 | +8.30 |
| **mean** | | 89.12 | **87.82** | 94.31 | **93.69** | +5.87 (before +5.20) | |

**Summary.**
- Routed-only: −1.30 µs mean; every case is better (−0.9 to −2.3 µs). The best config per case is unchanged except
  zipf 64 and zipf_skewed 16 (now e32w4-adaptive).
- Routed + shared: −0.62 µs mean; 17 of 20 cases are better. Regressions: zipf_skewed 8 (+1.10), lognormal 64 (+0.40),
  hot_experts 16 (+0.33).
- Gap vs best routed-only: +5.87 µs (before +5.20). The routed kernel gains the whole quantize launch, while in the
  fused kernel part of that gain is lost to the shared quantization competing with the routing exchange (see variants
  above).
- Same-config A/B before re-tuning (pre-P6 tuned config/margin/delay, interleaved, 2 reps): routed −1.27, fused −0.16.
  So the margin/delay tuning moves with P6.

**Rejected follow-up: fuse only the routed quantization.** The shared input stayed on a separate (shared-only) quantize
launch, with the routed input quantized in routing. At the P6-tuned margin/delay this is worse in all 20 cases
(+1.26 µs mean): the extra launch costs more than the in-kernel shared quantization. Reverted.
