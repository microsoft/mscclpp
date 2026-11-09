# MegaMoE vs DeepGEMM: Kernel Structure and the Shared Expert

This note covers three questions:

- Why MSCCL++ MegaMoE and DeepGEMM (DG) take about the same time on routed
  experts alone.
- Why adding the shared expert costs MegaMoE much more than it costs DG.
- What fusing the shared expert into MegaMoE would take.

The setup:

- 32 GB300 GPUs (DP32/EP32), 16 tokens/rank, 512 routed experts, top-7.
- Routed experts: hidden 4096, intermediate 4352, MXFP4 weights and static FP8
  activations. That is about 454 MB of weights per rank per call.
- Shared expert: hidden 8704, intermediate 2048, FP8 weights and activations,
  about 55 MB.

Reproduction steps and the MegaMoE shared-expert implementation are in
[native-megamoe-shared-benchmark-instructions.md](native-megamoe-shared-benchmark-instructions.md).

## Summary

- Both kernels have the same three parts:
  - a serial start, where routes and token counts are exchanged;
  - an overlapped middle, where dispatch, FC1, FC2 and the combine sends run
    concurrently;
  - a serial end: a barrier, a wait for peers, then the combine sum.

  Neither one runs the routed experts sequentially.
- On routed experts alone they are within a few µs of each other. DG leads by
  1.9–5.3 µs on balanced, lognormal and zipf. MegaMoE leads by 0.3 µs on
  hot_experts and 8.2 µs on zipf_skewed.
- Adding the shared expert costs DG about 1–4 µs and MegaMoE about 10 µs:
  - DG runs the shared expert inside its kernel, while its GEMM warps would
    otherwise be idle.
  - MegaMoE runs it as a second kernel on 18 reserved SMs.
- To close the gap, MegaMoE needs the same structure as DG:
  - shared-expert tasks inside the kernel;
  - dynamic task claiming;
  - routing moved off the GEMM warps.

## Kernel structure

Code locations:

- MegaMoE: paths are relative to this repository. Line numbers are from the
  working tree of `caiorocha/megamoe_shared_com`, on top of `e17318f2`.
- DG default kernel (`sm100_fp8_fp4_mega_moe_impl`). `<DG>` is
  `/home/caiorocha/test/run-dg-mscclpp/dg-routed-compare-20261103T0733Z/packages/deep-gemm/deep_gemm/include/deep_gemm`.
- DG tuned kernel (`decode32_n64_sfa_l2_prefetch`, used for the fused numbers
  below). `<DG-tuned>` is
  `/home/caiorocha/test/run-dg-mscclpp/tuned-dg-no-shared-n8-64-20261103T213731Z/packages/decode32-n48/deep_gemm/include/deep_gemm`.

| Stage | MegaMoE | DeepGEMM |
| --- | --- | --- |
| Serial start | `prepareRoutes` runs on every warp of every CTA. It exchanges routes with all peers and has two grid barriers (`src/ext/megamoe/include/megamoe_routing.cuh:349-437`). | Runs on the dispatch warps only: count tokens, offset atomics, write source indices to peers, grid sync, push counts, NVLink barrier (`<DG>/impls/sm100_fp8_fp4_mega_moe.cuh:360-454`). |
| GEMM warps during the start | Busy inside `prepareRoutes`. Its scratch space uses the GEMM shared memory (`megamoe_routing.cuh:374-376`, `427-430`). | Free. With routed experts only, they wait for tasks while the scheduler waits in `fetch_expert_recv_count` (`<DG>/scheduler/mega_moe.cuh:499`). With a shared expert, they run shared-expert tasks. |
| Dispatch with FC1 | Dispatch warps pull tokens (`megamoe_dispatch.cuh:257`). FC1 activation loads wait per token chunk (`megamoe_w4a8_roles.cuh:112-122`); weight loads run ahead (`:124`). | Dispatch warps pull tokens, interleaving source ranks (`num_active_ranks`, `<DG>/impls/sm100_fp8_fp4_mega_moe.cuh:459-647`). FC1 waits per token block (`:738-742`). |
| FC1 to FC2 | Each FC2 task waits for its token block (`hiddenReady`, `megamoe_w4a8_roles.cuh:77-84`). | Each FC2 task waits per K-block (`L2KBlockDependency`, `<DG>/impls/sm100_fp8_fp4_mega_moe.cuh:748-751`). |
| Combine sends | The FC2 epilogue writes FP8 partial results to the source rank (`megamoe_epilogue.cuh:334-339`). | The FC2 epilogue writes BF16 rows to the remote combine buffer (`<DG>/impls/sm100_fp8_fp4_mega_moe.cuh:1346-1460`). |
| Serial end | Grid barrier, wait for all peers, then sum (`megamoe_epilogue.cuh:619-643`). | Grid sync, then each token waits only for the ranks that hold its experts, then sum (`<DG>/impls/sm100_fp8_fp4_mega_moe.cuh:1522-1600`). |
| Task order | Static round-robin (`megamoe_w4a8_roles.cuh:73`, `megamoe_mma.cuh:144`, `megamoe_epilogue.cuh:443`). | Dynamic, through atomic counters (`<DG>/scheduler/mega_moe.cuh:407-410`). |
| Shared expert | A separate kernel on reserved SMs. | Inside the kernel (see [How DG hides the shared expert](#how-dg-hides-the-shared-expert)). |

## Routed experts only

From `report.md`, which used the same harness for both, in µs per kernel at
16 tokens/rank:

| Routing | DG | MegaMoE | MegaMoE − DG |
| --- | --- | --- | --- |
| balanced | 92.46 | 97.73 | +5.27 |
| hot_experts | 79.67 | 79.41 | −0.26 |
| lognormal | 94.68 | 97.39 | +2.71 |
| zipf | 82.30 | 84.19 | +1.89 |
| zipf_skewed | 90.92 | 82.74 | −8.17 |

Both kernels are limited by the same work:

- streaming about 454 MB of routed weights per rank, which takes at least 57 µs
  at the 7.9 TB/s HBM peak;
- the serial start and end.

Their overlap is structurally the same, so neither gains much from it. The
cause of DG's lead on balanced routing was not measured. Possible causes:

- dynamic task claiming;
- scale-factor L2 prefetch (`sfa_l2_prefetch`);
- 152 SMs versus MegaMoE's 150 CTAs.

## With the shared expert

Times are in µs. The MegaMoE two-kernel run uses 134 routed CTAs and 18 shared
CTAs.

| Routing | DG routed only | DG fused | DG cost | MegaMoE routed only (150 CTAs) | MegaMoE two kernels | MegaMoE cost |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | 92.46 | 95.04 | +2.6 | 99.10 | 108.64 | +9.5 |
| hot_experts | 79.67 | 81.79 | +2.1 | 81.07 | 91.95 | +10.9 |
| lognormal | 94.68 | 95.81 | +1.1 | 99.39 | 109.05 | +9.7 |
| zipf | 82.30 | 84.07 | +1.8 | 85.26 | 95.84 | +10.6 |
| zipf_skewed | 90.92 | 95.04 | +4.1 | 86.16 | 96.86 | +10.7 |

Where the numbers come from:

- **DG routed only:** `report.md`, measured with CUDA events over graph replay.
- **DG fused:** the tuned `decode32-n48` policy, measured from profiler kernel
  durations. Each value is the largest per-rank median over 101 kernels, from
  `/home/caiorocha/test/run-dg-mscclpp/deepgemm-fused-n16-five-patterns-20261105T2301Z/kernel-tuning/report.txt`.
  - The harness and timing method differ from `report.md`, so the DG cost is
    approximate (about ±2 µs).
  - With DG's default decode policy, the fused kernel took
    101.8 / 99.0 / 109.6 / 98.2 / 149.4 µs (`kernel-profiles/report.txt` in the
    same directory). The fused kernel needs tuning too.
- **MegaMoE:** `benchmark.py` time per call, with the fp8_nv W8A8 shared
  expert. See
  [All routing patterns](native-megamoe-shared-benchmark-instructions.md#all-routing-patterns-mxfp8-and-fp8_nv-w8a8).
  Its routed-only times are 1.1–3.4 µs above `report.md`'s harness.

### How DG hides the shared expert

From the tuned scheduler
(`<DG-tuned>/scheduler/decode32_n64_shared64_tile.cuh:486-522`):

1. **Shared FC1 is claimed first.** It is claimed before the routed task list
   exists, because "Shared expert L1 tasks do not depend on dispatch"
   (`:492-494`). It runs while the dispatch warps handle the serial start,
   when the GEMM warps would otherwise be idle.
2. **Routed tasks come next.** They are claimed dynamically after
   `fetch_expert_recv_count` (`:501-511`).
3. **Shared FC2 is claimed last,** once the routed tasks run out (`:513-517`).
   It fills clusters that would otherwise be idle at the end of the GEMM phase,
   before the end barrier. The default DG kernel instead runs FC2 early when
   one wave fits (`<DG>/scheduler/mega_moe.cuh:486-495`).
4. **No SMs are reserved.** A cluster that finishes shared work takes routed
   tasks, and the other way around.
5. **The shared output is written separately** (`shared_y`,
   `kSeparateSharedOutput`). The shared expert is 8704 wide, while the routed
   combine is 4096 wide.

### Why MegaMoE pays about 10 µs

- `prepareRoutes` occupies every warp and the GEMM shared memory, so no shared
  work can run during the serial start.
- The shared expert therefore needs its own kernel and its own SMs. With 134
  CTAs the routed kernel needs one more round, which costs 3.8–5.2 µs.
- While both kernels run, they compete for HBM bandwidth, which costs another
  5.5–5.9 µs.

## Fusing the shared expert into MegaMoE

### Where to place the shared tasks

- **FC1:** at kernel entry, or right after routing. Its inputs (the local
  tokens) and its weights are known when the kernel starts.
- **FC2:** either as filler at the end of the GEMM phase (tuned DG) or right
  after FC1 (default DG).
  - Do not place it after the end barrier. At that point only the ranks that
    finished early are idle. The slowest rank, which sets the time, has no idle
    time there, so FC2 would add directly to it.
- **Shared output:** keep it separate, as DG does. It cannot be summed in
  `combineSlots` because the shared output is 8704 wide and the routed combine
  is 4096 wide.

### Changes needed

1. **Dynamic task claiming.** Add an atomic task counter and a way to pass each
   task to the load, MMA and epilogue roles. This replaces the static loops in
   `megamoe_w4a8_roles.cuh:73`, `megamoe_mma.cuh:144` and
   `megamoe_epilogue.cuh:443`.

   A static order does not work. A shared FC1 task streams 2.2 MB of weights
   (256 rows × 8704 FP8), about 4× a routed task (about 0.5 MB). The clusters
   that get one would finish their routed tasks late.
2. **Mixed MXFP4 routed and FP8 shared tasks in one kernel.** This needs:
   - a second set of TMA descriptors;
   - FP8 weight tiles, which take twice the bytes per stage;
   - a second MMA instruction descriptor;
   - per-matrix scales and a local BF16 output in the epilogue.

   The pieces exist in `megamoe_w8a8.cu`. However, the routed kernel is already
   tight on shared memory and registers; see the register reconfiguration at
   `megamoe_w4a8.cu:96`, `131` and `138-141`.
3. **Routing on the dispatch warps only.** Routing needs its own shared-memory
   scratch, and named barriers instead of whole-CTA syncs. The GEMM warps can
   then start shared FC1 at kernel entry. Without this change, shared work
   starts only after routing and misses the serial-start window.
4. **A compile-time flag** so that the routed-only kernel stays unchanged.

### Expected result

The shared-expert cost should drop from about 10 µs to about 2–5 µs (DG pays
1–4 µs). Balanced routing would then take about 101–104 µs instead of
108.6 µs.

The main unknown is how long MegaMoE's idle windows are. Streaming the shared
weights takes about 7 µs at the HBM peak, and longer in practice because FC1
has only 16 cluster tasks.

### First step: measure the idle windows

Measure the windows before writing the fused kernel:

1. Build with `MSCCLPP_MEGAMOE_TRACE=1`. Only non-JIT builds export the trace;
   the host export is at `megamoe_w4a8.cu:16-24`. The phases are listed in
   `src/ext/megamoe/include/megamoe_trace.cuh`, and only CTAs 0 and 1 are
   recorded.
2. On the slowest rank, add the `Routing` phase to the first `TokenReady` wait.
3. Interpret the total:
   - 10 µs or more: fusion can hide most of the shared expert.
   - 2–3 µs: fusion mainly recovers the 3.8–5.2 µs lost to the SM split.
