# Native MegaMoE Shared-Expert Benchmark

The native benchmark is located at:

```text
/home/caiorocha/test/mscclpp-dg/python/mscclpp/ext/megamoe/benchmark.py
```

Run it with `--overlap-shared-expert`. The sections below record the exact
configuration, commands, environment and results of the routed-only versus
routed + shared comparison, so that it can be reproduced. There are two shared
expert kernels: W8A16 (the original) and W8A8 (added later, see
[W8A8 shared expert](#w8a8-shared-expert)). W8A8 is faster and is the
recommended one. W8A8 takes MXFP8 weights or `fp8_nv` weights (FP8 E4M3 with one
FP32 scale per weight matrix, see [fp8_nv weights](#fp8_nv-weights)). Both run
at the same speed; see
[All routing patterns](#all-routing-patterns-mxfp8-and-fp8_nv-w8a8).

## Shared expert

- One local world-size-one MSCCL++ MegaMoE context per rank.
- MXFP8 weights. The activations depend on `--shared-activation-dtype`:
  - `bf16` (default): BF16 activations, W8A16 kernel.
  - `fp8_static`: static FP8 activations with per-tensor scale
    `--shared-activation-scale` (default 1.0), W8A8 kernel. This needs the
    rebuilt `package-w8a8` package.
- `--shared-weight-dtype fp8_nv` (W8A8 only) replaces the MXFP8 weights with
  FP8 E4M3 weights and one FP32 scale per matrix: one for FC1 (gate and up
  together) and one for FC2. The benchmark quantizes each matrix with
  scale = amax / 448. This needs the `package-fp8nv` package. The default,
  `mxfp8`, keeps the MXFP8 weights.
- Gated SiLU/SwiGLU.
- Hidden size: 8704 (`--shared-hidden`, default).
- Post-SwiGLU intermediate size: 2048 (`--shared-intermediate`, default).
- One local expert with top-k one.
- No inter-rank communication.
- Uses a separate CUDA stream and communicator from routed MegaMoE. Each call
  forks the shared stream from the routed stream and joins it before the next
  routed call (`--shared-join iteration`, default).
- `--shared-sm-count N` sets the exact persistent shared CTA count. On GB300 the
  routed kernel gets 152 − `--sm-margin` CTAs, rounded down to a multiple of its
  two-CTA cluster. The benchmark rejects splits where routed plus shared CTAs
  exceed the SM count, because the two cooperative kernels could not then run
  at the same time.
- `--shared-wait-for-routed-start` (default on) launches shared work after the
  routed kernel signals entry. The measured runs disable it with
  `--no-shared-wait-for-routed-start`. With a co-resident split the wait is
  unnecessary and costs 1–8.5 µs per call.
- `--shared-weight-banks N` (default 1) rotates N copies of the shared weights
  so they are not L2-resident across calls, as in a multi-layer model. The
  measured runs use 4.
- `--input-mode staged` (default) copies each input into the registered
  workspace on every call. `direct` writes the inputs there once and skips the
  copy. Staged was faster with overlap and is used for all results below.

For each metric, the JSON output takes the slowest rank in every sample and
reports the median, mean, minimum and maximum over `--iterations` samples. The
tables below use the median.

1. `latency_us_max_across_ranks`: per-call latency. With overlap, this covers
   routed + shared in a graph without timing events. Compare this number.
2. `routed_control_latency_us_max_across_ranks`: routed only, at the same
   routed CTA count.
3. `shared_control_latency_us_max_across_ranks`: shared only.
4. `routed_overlap_latency_us_max_across_ranks`,
   `shared_overlap_latency_us_max_across_ranks`, and the shared start delay and
   tail: per-branch times from a second overlap graph with event nodes. Event
   nodes add latency, so these read higher than the per-call latency.
5. `kernel_trace_us_max_across_ranks` (only with `--trace-dir`): median kernel
   durations from profiled replays.

Per-call latency uses CUDA events around one replay of a graph that holds
`--graph-batch` back-to-back calls, divided by the batch size.
`--graph-timing steady-state` queues an untimed replay first, which excludes the
initial cross-rank submission skew.

## Measured configuration

The routed MegaMoE configuration matches `report.md`:

- World size 32: 8 nodes × 4 NVIDIA GB300 GPUs (152 SMs each).
- 16 tokens per rank, hidden size 4096, intermediate size 4352.
- 512 experts (16 per rank) with top-k seven.
- W4A8: MXFP4 weights and static FP8 activations with scale 1.0
  (`--activation-dtype fp8_static --activation-scale 1.0`).
- Tuned W4A8 kernel key
  `1955dc4fad24cb5a9b7e372b6aea5918fa6bdd41ee25ef60b8a8f38de6ebd14d`. See
  [Kernel key](#kernel-key) for other environments.
- Two routing patterns, shown here for rank 0:
  - `balanced`: every local expert receives 3–10 routes.
  - `hot_experts`: one expert receives 93 of 122 routes, 4 of 16 receive none
    and the rest receive 1–4.
- The `fp8_nv` comparison also covers the other three `report.md` patterns:
  `lognormal`, `zipf` and `zipf_skewed`, with the benchmark defaults
  (`--zipf-alpha 1.2`, `--lognormal-sigma 0.85`, `--seed 1234`).

A previous version of this file listed intermediate size 5120 and MXFP8
activations. None of the results below used that configuration.

## Commands

Arguments shared by every run:

```bash
COMMON=(
  --tokens 16 --hidden 4096 --intermediate 4352 --experts 512 --top-k 7
  --weight-dtype mxfp4 --activation-dtype fp8_static --activation-scale 1.0
  --kernel-key 1955dc4fad24cb5a9b7e372b6aea5918fa6bdd41ee25ef60b8a8f38de6ebd14d
  --cache-dir <JIT_CACHE_DIRECTORY>
  --graph --graph-batch 10 --graph-timing steady-state
  --warmup 5 --iterations 30
)
# Shared-expert overlap. The H8704/I2048 shape is the default.
SHARED=(--overlap-shared-expert --shared-weight-banks 4 --no-shared-wait-for-routed-start)
```

Per-run arguments, appended after `"${COMMON[@]}"`:

| Run | Arguments |
| --- | --- |
| Balanced, routed only (150 CTAs) | `--routing-pattern balanced --sm-margin 2` |
| Balanced, routed + shared (136 + 16) | `"${SHARED[@]}" --routing-pattern balanced --sm-margin 16 --shared-sm-count 16` |
| Hot experts, routed only (150 CTAs) | `--routing-pattern hot_experts --sm-margin 2` |
| Hot experts, routed + shared (128 + 24) | `"${SHARED[@]}" --routing-pattern hot_experts --sm-margin 24 --shared-sm-count 24` |

W8A8 shared-expert runs use the `package-w8a8` package and the
`jit-cache-w8a8` JIT cache (see [Software](#software)). That package rebuilds the
native library, so the content-addressed key of the routed kernel changes, even
though its machine code is identical. For these runs, use this key in `COMMON`
and add the W8A8 flags to the shared arguments:

```bash
# In COMMON, instead of the 1955dc4f… key:
--kernel-key 10a95c416084c944777e49e2852d80b82adf060320260269f39230d96a609248
# --shared-activation-dtype defaults to bf16 (W8A16).
SHARED_W8A8=("${SHARED[@]}" --shared-activation-dtype fp8_static --shared-activation-scale 1.0)
```

| Run (W8A8 package) | Arguments |
| --- | --- |
| Balanced, routed only (150 CTAs) | `--routing-pattern balanced --sm-margin 2` |
| Balanced, routed + W8A8 shared (134 + 18), best | `"${SHARED_W8A8[@]}" --routing-pattern balanced --sm-margin 18 --shared-sm-count 18` |
| Hot experts, routed only (150 CTAs) | `--routing-pattern hot_experts --sm-margin 2` |
| Hot experts, routed + W8A8 shared (134 + 18), best | `"${SHARED_W8A8[@]}" --routing-pattern hot_experts --sm-margin 18 --shared-sm-count 18` |

The other W8A8 splits in [Results](#results) change only `--sm-margin` and
`--shared-sm-count`, which are equal in every run.

The MXFP8 versus `fp8_nv` comparison over all routing patterns used the
`package-fp8nv` package, the `jit-cache-fp8nv` cache and another routed key,
again with identical routed machine code:

```bash
# In COMMON, instead of the 1955dc4f… key:
--kernel-key 064856c13c5ed5fa85ee0cf80cfe31d618766254a1979448ed93d0ab40225fbf
SHARED_W8A8=("${SHARED[@]}" --shared-activation-dtype fp8_static --shared-activation-scale 1.0)
SPLIT=(--sm-margin 18 --shared-sm-count 18)   # 134 routed + 18 shared CTAs
```

For each `<PATTERN>` in `balanced`, `hot_experts`, `lognormal`, `zipf` and
`zipf_skewed`:

| Run (fp8_nv package) | Arguments |
| --- | --- |
| Routed only (150 CTAs) | `--routing-pattern <PATTERN> --sm-margin 2` |
| Routed + MXFP8 W8A8 shared (134 + 18) | `"${SHARED_W8A8[@]}" "${SPLIT[@]}" --routing-pattern <PATTERN>` |
| Routed + fp8_nv W8A8 shared (134 + 18) | `"${SHARED_W8A8[@]}" "${SPLIT[@]}" --routing-pattern <PATTERN> --shared-weight-dtype fp8_nv` |

`$RUN/sweep-fp8nv.sh` runs all of these, plus a `--check` run (balanced,
fp8_nv) and a routed-only `zipf` run with `--sm-margin 4`, the margin of the
`report.md` zipf winner. Each run takes about 10 s.

Append `--json-output <RESULT_FILE>` to every run. Optional flags:

- `--check` compares the routed and shared outputs with a Torch reference. It
  passed for both W8A16 overlap runs, for the W8A8 136 + 16 runs with both
  routing patterns, and for the balanced fp8_nv 134 + 18 run.
- `--trace-dir <DIR>` writes per-rank Chrome traces and reports kernel
  durations. Profiling runs after the timed samples, so it does not change the
  per-call numbers.

Pass the following flags explicitly, because their defaults (in parentheses)
differ from the measured configuration:

- `--tokens` (32)
- `--experts` (64)
- `--activation-dtype` (mxfp8)
- `--sm-margin` (32)
- `--graph-timing` (isolated)
- `--shared-weight-banks` (1)
- `--shared-wait-for-routed-start` (on)

## Launch

The benchmark reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR` and
`MASTER_PORT`. MSCCL++ bootstraps on `MASTER_PORT + 1`. Any launcher that starts
one process per GPU works. The measured runs used MPICH Hydra inside the
`caiorocha-benchzen-original-r6p5` container, with this per-rank wrapper:

```bash
#!/usr/bin/env bash
# rank.sh
set -euo pipefail
ulimit -Sn 16384
export PYTHONPATH=<PACKAGE_DIR>
export MSCCLPP_HOME=<PACKAGE_DIR>/mscclpp
export MSCCLPP_MEGAMOE_CACHE_DIR=<JIT_CACHE_DIRECTORY>
export RANK=$PMI_RANK WORLD_SIZE=$PMI_SIZE LOCAL_RANK=$MPI_LOCALRANKID
export MASTER_ADDR=<RANK0_HOST> MASTER_PORT=$BENCH_PORT
export GLOO_SOCKET_IFNAME=<NETWORK_INTERFACE>
exec python -m mscclpp.ext.megamoe.benchmark "$@"
```

```bash
# <HOSTS_FILE> lists the 8 nodes, one "<host>:4" line each.
mpiexec.hydra -n 32 -ppn 4 -f <HOSTS_FILE> -genv BENCH_PORT <NEW_PORT> -l ./rank.sh \
  "${COMMON[@]}" <RUN_ARGUMENTS> --json-output <RESULT_FILE>
```

- Use a new port for every run, and check that all GPUs are idle first.
- The package and JIT cache must exist at the same path on every node.
- The exact scripts used are in
  `$RUN` = `/home/caiorocha/test/run-dg-mscclpp/native-megamoe-shared-tune-20261105T095309Z/`.
  The results and traces of the measured runs are in its `results/` and
  `traces/` directories.
  - `launch.sh <NAME> <PORT> <RUN_ARGUMENTS>` adds the common arguments and
    writes `results/<NAME>.json` and `logs/<NAME>.log`.
  - `rank.sh` is the per-rank wrapper.
  - `deploy.sh` copies the package and JIT cache to the worker nodes.
  - `gpu-idle.sh` checks that all GPUs are idle.
- The W8A8 runs were launched differently, because host SSH (port 22) stopped
  accepting sessions. They reach each node's benchmark container through its
  own sshd (root, port 3030):
  - `LAUNCHER=$RUN/mpi-container-ssh-3030.sh bash launch.sh ...` makes Hydra use
    the container sshd. It imports the container environment from
    `/proc/1/environ`, as `docker exec` does.
  - `deploy-3030.sh` and `gpu-idle-3030.sh` replace `deploy.sh` and
    `gpu-idle.sh`.
  - `sweep-w8a8.sh`, `sweep-w8a8-b.sh` and `sweep-w8a8-c.sh` are the W8A8
    sweeps, and `sweep-fp8nv.sh` is the MXFP8 versus fp8_nv sweep. They step
    the port by 10 between runs. A step of 1 makes a run's `MASTER_PORT`
    collide with the previous run's `MASTER_PORT + 1`, which hangs the run.
  - `routing_rounds.py` prints each pattern's busiest-rank task count and its
    rounds per routed CTA count (run it in the container with the package on
    `PYTHONPATH`).

## Software

- CUDA 13.0 (driver, runtime and `nvcc`), PyTorch 2.11.0 and Python 3.12.
- The MSCCL++ package at `<PACKAGE_DIR>` consists of the prebuilt libraries from
  `/home/caiorocha/test/run-dg-mscclpp/kernel-only-tuning-n8-64-20261103T224544Z/packages/mscclpp-prepared`
  with this repository's `python/mscclpp` copied over them, including the
  updated `benchmark.py`.
  - The libraries were built as Release with `MSCCLPP_BUILD_EXT_MEGAMOE=ON`,
    `MSCCLPP_BUILD_PYTHON_BINDINGS=ON` and `MSCCLPP_BUILD_EXT_COLLECTIVES=ON`.
  - The MegaMoE kernels are unmodified. This applies to all W8A16 results.
- The W8A8 runs used `$RUN/package-w8a8` and `$RUN/jit-cache-w8a8` instead.
  - The package is the package above, plus a rebuilt `libmscclpp_megamoe.so`
    and `share/mscclpp/megamoe` JIT sources, built from this repository's
    working tree by `$RUN/build-w8a8.sh`.
  - This build changes the MegaMoE sources (see
    [W8A8 kernel changes](#w8a8-kernel-changes-and-rollback)).
  - The routed kernel's machine code is identical to the original. Its key is
    `10a95c416084c944777e49e2852d80b82adf060320260269f39230d96a609248`.
  - To recreate the cache, run `GPU=<N> $RUN/jit-seed.sh <PACKAGE_DIR> <CACHE_DIR>`
    inside the container. It compiles the tuned routed policy of the
    `1955dc4f…` manifest against the given package and prints the new key.
- The MXFP8 versus fp8_nv runs used `$RUN/package-fp8nv` and
  `$RUN/jit-cache-fp8nv`.
  - The package is `package-w8a8` with a rebuilt `libmscclpp_megamoe.so`,
    `share/mscclpp/megamoe` JIT sources, `_mscclpp` binding module (for the
    `Fp8Nv` enum value) and this repository's `api.py`, `benchmark.py` and
    `README.md`.
  - The routed kernel's machine code is again identical to the original. Its
    key is `064856c13c5ed5fa85ee0cf80cfe31d618766254a1979448ed93d0ab40225fbf`.
  - The build source `$RUN/source-w8a8` is not a copy of this repository. It
    has extra `forwardPrepared` code from the original package, so update it
    with patches rather than by copying files.

### Kernel key

The JIT key is content-addressed. It hashes the GPU, the driver, the PyTorch and
CUDA versions, the MSCCL++ libraries and the JIT sources. As a result,
`1955dc4f…` loads only with that exact package and environment.

In any other environment, replace `--kernel-key ...` with the same policy given
as flags:

```bash
--epilogue-tokens 16 --epilogue-warps 8 --epilogue-registers 160 --dispatch-warps 2 --fixed-token-count
```

All other fields keep their `W4A8KernelConfig` defaults. The benchmark then
compiles an equivalent module into `--cache-dir`, which needs `nvcc` and
`MSCCLPP_MEGAMOE_CUTLASS_ROOT`. The tuned module was compiled with CUTLASS 4.8.0
headers from
`/home/caiorocha/test/run-dg-mscclpp/comparison-n16-20261102/native-megamoe-n16/dependencies/cutlass`.
Each result JSON records the policy under `ranks[*].kernel_config` and the key
under `ranks[*].kernel_id`.

## Results

"Per call" is `latency_us_max_across_ranks`. "Routed-only control" is
`routed_control_latency_us_max_across_ranks` at the same routed CTA count.
"Shared only" is `shared_control_latency_us_max_across_ranks`: the shared graph
alone, including its flag memset and, for W8A8, its input quantization. Kernel
times come from `--trace-dir` traces of the same configurations.

### Summary

Per call (µs), best split in parentheses:

| Routing | Routed only (150 CTAs) | W8A16 shared | W8A8 shared, 136 + 16 | W8A8 shared, best |
| --- | --- | --- | --- | --- |
| Balanced | 99.1 | 112.6–112.9 (136 + 16) | 109.1–109.7 | **108.7–108.8** (134 + 18) |
| Hot experts | 81.0 | 97.1 (128 + 24) | 92.2–92.8 | **91.8** (134 + 18) |

Routed only gave the same result with both packages: 99.1 µs balanced and
81.0 µs hot experts.

### All routing patterns (MXFP8 and fp8_nv W8A8)

Per call (µs) from `sweep-fp8nv.sh`, W8A8 shared expert at 134 routed + 18
shared CTAs. The `report.md` columns are its routed-kernel winners at 16
tokens/rank, measured by a different harness:

| Routing | Routed only (150 CTAs) | Routed only at 134 CTAs | + MXFP8 shared | + fp8_nv shared | fp8_nv vs routed only | `report.md` MSCCL++ / DG |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | 99.10 | 102.96 | 108.74 | **108.64** | +9.5 (+9.6%) | 97.73 / 92.46 |
| hot_experts | 81.07 | 86.08 | 91.82 | **91.95** | +10.9 (+13.4%) | 79.41 / 79.67 |
| lognormal | 99.39 | 103.19 | 109.05 | **109.05** | +9.7 (+9.7%) | 97.39 / 94.68 |
| zipf | 85.26 | 90.08 | 95.84 | **95.84** | +10.6 (+12.4%) | 84.19 / 82.30 |
| zipf_skewed | 86.16 | 91.40 | 96.80 | **96.86** | +10.7 (+12.4%) | 82.74 / 90.92 |

- "Routed only at 134 CTAs" is the routed-only control of the fp8_nv run.
- fp8_nv and MXFP8 differ by at most 0.13 µs per call, which is noise.
- Shared alone takes 41.2 µs with either format, in every pattern. Overlapped,
  it takes about 55.5 µs and ends 36–54 µs before the routed kernel.
- The routed-only `zipf` run at margin 4 (148 CTAs) measured 85.16 µs.
- `report.md` lists `k128-cache-expert-major@margin2` as the lognormal winner.
  In its screening, the `k128-epilogue16-warps8@margin2` policy used here
  measured 97.42 µs and `k128-cache-expert-major@margin2` 97.55 µs.
- The ≈10 µs overhead splits as in the balanced analysis under
  [W8A8 results](#w8a8-results): 3.8–5.2 µs for one more routed round, then
  5.5–5.9 µs of HBM contention with the shared kernel.
- 134 + 18 is the right split for every pattern. Busiest-rank tasks and
  rounds (`routing_rounds.py`):

  | Routing | Busiest-rank tasks | Rounds at 150 / 148 / 144–134 / 132 CTAs |
  | --- | --- | --- |
  | balanced, lognormal | 800 (16 token blocks) | 11 / 11 / 12 / 13 |
  | zipf, zipf_skewed | 650 (13 token blocks) | 9 / 9 / 10 / 10 |
  | hot_experts | 600 (12 token blocks) | 8 / 9 / 9 / 10 |

  Avoiding the extra round leaves at most 4 SMs for the shared expert (2 for
  hot_experts). At 6 SMs the shared expert already outlasts the routed kernel
  (balanced 146 + 6: 125.3 µs per call).
- The balanced fp8_nv `--check` run passed on all 32 ranks. The shared
  output's largest error is 0.0156 (MXFP8: 0.0078). The routed output's error
  is unchanged.

### Tuned batch-size sweep (8–64 tokens/rank)

Shared expert: fp8_nv W8A8, H8704/I2048, 4 weight banks,
`--no-shared-wait-for-routed-start`. Routed + shared CTAs always total 152.
All numbers are the mean of three interleaved runs; each set of three
repeats varied by at most 0.25 µs.

How the configs were tuned:

1. **Split screen** (120 runs). Each case's `report.md` routed-only winner
   policy at 8, 12, 16, 18, 20 and 24 shared CTAs.
2. **Policy screen** (400 runs). All 11 policies in `policies.json` at the two
   best splits from step 1.
3. **Repeats.** The top three configs per case, run twice more. The best mean
   becomes the case's config.
4. **Head-to-head.** Three alternating runs of:
   - that config with the shared expert;
   - the same policy and routed CTA count, routed only (`--sm-margin` = shared
     CTAs);
   - the `report.md` routed-only winner at its own margin.

| Routing | Tokens | Shared + routed config (routed + shared CTAs) | Shared + routed (µs) | Routed only, same config (µs) | Overhead vs same config | Routed only, `report.md` winner (policy, CTAs) | Overhead vs `report.md` winner | Shared alone (µs) |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| balanced | 8 | `k128-epilogue16-warps8`, 134 + 18 | 106.32 | 100.15 | +6.17 (+6.2%) | 97.05 (`k128-epilogue16-warps8`, 150) | +9.27 (+9.6%) | 41.24 |
| balanced | 16 | `k256-warps16`, 134 + 18 | 108.36 | 102.27 | +6.09 (+6.0%) | 99.09 (`k128-epilogue16-warps8`, 150) | +9.27 (+9.4%) | 41.20 |
| balanced | 32 | `k256-warps16`, 134 + 18 | 109.49 | 103.23 | +6.26 (+6.1%) | 100.58 (`k128-epilogue16-warps8`, 148) | +8.91 (+8.9%) | 41.36 |
| balanced | 64 | `k128-epilogue16-warps8`, 134 + 18 | 110.18 | 105.67 | +4.51 (+4.3%) | 102.40 (`k128-profile`, 148) | +7.77 (+7.6%) | 43.28 |
| hot_experts | 8 | `k128-epilogue16-warps8`, 134 + 18 | 88.57 | 81.39 | +7.18 (+8.8%) | 77.79 (`k128-epilogue16-warps8`, 150) | +10.78 (+13.9%) | 41.19 |
| hot_experts | 16 | `k128-expert-major`, 134 + 18 | 91.44 | 83.83 | +7.60 (+9.1%) | 81.03 (`k128-epilogue16-warps8`, 150) | +10.41 (+12.8%) | 41.19 |
| hot_experts | 32 | `k256-warps16`, 132 + 20 | 95.02 | 88.22 | +6.80 (+7.7%) | 85.57 (`k256-warps16`, 150) | +9.45 (+11.0%) | 41.27 |
| hot_experts | 64 | `k256-warps16`, 140 + 12 | 100.49 | 92.92 | +7.58 (+8.2%) | 90.81 (`k128-epilogue16-warps8`, 152) | +9.69 (+10.7%) | 57.75 |
| lognormal | 8 | `k128-epilogue16-warps8`, 134 + 18 | 106.29 | 100.11 | +6.19 (+6.2%) | 96.81 (`k128-epilogue16-warps8`, 150) | +9.48 (+9.8%) | 41.17 |
| lognormal | 16 | `k256-warps16`, 134 + 18 | 108.72 | 102.30 | +6.42 (+6.3%) | 99.40 (`k128-cache-expert-major`, 150) | +9.31 (+9.4%) | 41.26 |
| lognormal | 32 | `k128-adaptive-expert-major`, 134 + 18 | 111.78 | 105.32 | +6.46 (+6.1%) | 102.04 (`k128-cache-weights`, 148) | +9.74 (+9.5%) | 41.32 |
| lognormal | 64 | `k128-adaptive-cache`, 144 + 8 | 115.24 | 107.53 | +7.70 (+7.2%) | 104.79 (`k128-adaptive-cache`, 152) | +10.45 (+10.0%) | 75.07 |
| zipf | 8 | `k128-epilogue16-warps8`, 132 + 20 | 92.97 | 86.18 | +6.80 (+7.9%) | 82.76 (`k128-epilogue16-warps8`, 148) | +10.21 (+12.3%) | 41.20 |
| zipf | 16 | `k256-warps16`, 132 + 20 | 95.46 | 88.63 | +6.82 (+7.7%) | 85.15 (`k128-epilogue16-warps8`, 148) | +10.30 (+12.1%) | 41.17 |
| zipf | 32 | `k128-adaptive-expert-major`, 132 + 20 | 97.75 | 90.31 | +7.44 (+8.2%) | 87.25 (`k128-expert-major`, 150) | +10.50 (+12.0%) | 41.22 |
| zipf | 64 | `k128-cache-expert-major`, 140 + 12 | 100.67 | 92.33 | +8.34 (+9.0%) | 92.84 (`k128-expert-major`, 148) | +7.83 (+8.4%) | 57.75 |
| zipf_skewed | 8 | `k128-epilogue16-warps8`, 134 + 18 | 91.43 | 84.58 | +6.85 (+8.1%) | 81.46 (`k128-epilogue16-warps8`, 152) | +9.97 (+12.2%) | 41.21 |
| zipf_skewed | 16 | `k256-warps16`, 132 + 20 | 95.39 | 90.02 | +5.37 (+6.0%) | 86.19 (`k128-epilogue16-warps8`, 150) | +9.20 (+10.7%) | 41.17 |
| zipf_skewed | 32 | `k256-warps16`, 144 + 8 | 107.27 | 100.97 | +6.30 (+6.2%) | 100.00 (`k256-warps16`, 150) | +7.27 (+7.3%) | 72.23 |
| zipf_skewed | 64 | `k256-warps16`, 144 + 8 | 137.92 | 129.94 | +7.98 (+6.1%) | 126.60 (`k128-dispatch-stages2`, 152) | +11.32 (+8.9%) | 75.22 |

- **Overhead vs same config** (+4.5 to +8.3 µs) is interference only: HBM and
  L2 contention with the shared kernel, at identical routed CTAs and policy.
- **Overhead vs `report.md` winner** (+7.3 to +11.3 µs) adds the cost of giving
  SMs away, usually one more routed round.
- **Splits.** 134 + 18 or 132 + 20 wins up to 32 tokens/rank. At 64
  tokens/rank the busiest rank has more tasks, so 140 + 12 or 144 + 8 avoids
  an extra round. Shared alone then takes 58–75 µs instead of 41 µs, which
  still hides under the routed kernel. Tasks and rounds come from
  `routing_rounds.py <tokens>`.
- **Policies.** Policy choice moves results by at most about 1 µs.
  `k256-warps16` wins 9 of 20 cases with the shared expert, but only 2 of 20
  in `report.md` routed-only.
- **Two absolute numbers exceed 110 µs:** lognormal 64 (115.2) and zipf_skewed
  64 (137.9). This harness's `zipf_skewed` is heavier than `report.md`'s at
  32/64 tokens: routed only measures 100.0/126.6 µs here vs 92.4/116.1 µs in
  `report.md`.
- **Use the separate routed-only runs for the "same config" column.** The
  overlap run's own `routed_control` is 0.6–1.9 µs higher than a separate
  routed-only run.
- **Port warning.** Rank 0 must listen below 32768. Two early runs used ports
  in the 32768–60999 ephemeral range and hung on `EADDRINUSE`;
  `batch_sweep.py` now asserts against this.

Reproduce (in `$RUN`; JIT keys are seeded with `seed-policies.sh` and
deployed with `deploy-3030.sh`):

```bash
python3 make_plan.py A  && python3 batch_sweep.py plans/A.json  31100
python3 make_plan.py B  && python3 batch_sweep.py plans/B.json  20000
python3 make_plan.py C1 && python3 batch_sweep.py plans/C1.json 26000
python3 make_plan.py C2 && python3 batch_sweep.py plans/C2.json 27500
python3 final_table.py   # the table above (also saved as bt-final-table.md)
```

### Routed-start GPU signal (16 tokens/rank)

This uses the best 16-token configs from the table above with
`--shared-wait-for-routed-start`:

- The routed stream resets a device flag (memset + event).
- The routed kernel sets the flag when it enters.
- The shared stream waits on the event, then on `cuStreamWaitValue32` for the
  flag.

So the shared expert can only launch after the routed kernel is running
(`runtime.cc` `forward`/`waitUntilStarted`). Signal, no signal and routed only
ran alternately, three times each (`make_plan.py SIG`, `plans/SIG.json`); all
repeats were within 0.41 µs. Per call, µs:

| Routing | Config (routed + shared CTAs) | Routed only, same config | No signal | Signal | Signal cost | No signal vs routed only | Signal vs routed only | Routed only, `report.md` winner | Signal vs `report.md` winner | Shared start delay (µs), no signal / signal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | `k256-warps16`, 134 + 18 | 102.30 | 108.40 | 112.78 | +4.38 | +6.11 (+6.0%) | +10.48 (+10.2%) | 99.09 | +13.69 (+13.8%) | 3.0 / 11.1 |
| hot_experts | `k128-expert-major`, 134 + 18 | 83.79 | 91.45 | 95.53 | +4.08 | +7.65 (+9.1%) | +11.73 (+14.0%) | 81.03 | +14.50 (+17.9%) | 3.0 / 11.0 |
| lognormal | `k256-warps16`, 134 + 18 | 102.26 | 108.68 | 112.89 | +4.21 | +6.41 (+6.3%) | +10.63 (+10.4%) | 99.40 | +13.48 (+13.6%) | 2.9 / 11.1 |
| zipf | `k256-warps16`, 132 + 20 | 88.64 | 95.43 | 99.74 | +4.31 | +6.79 (+7.7%) | +11.09 (+12.5%) | 85.15 | +14.58 (+17.1%) | 3.0 / 11.0 |
| zipf_skewed | `k256-warps16`, 132 + 20 | 90.04 | 95.44 | 99.47 | +4.02 | +5.41 (+6.0%) | +9.43 (+10.5%) | 86.19 | +13.28 (+15.4%) | 3.0 / 11.0 |

The `report.md` winner column reuses the routed-only runs from the batch-size
sweep.

**Where the +4 µs goes.** Rank-0 `--trace-dir` traces of balanced
(`traces-sig/{SG,NS}`, results `sigtrace-{SG,NS}-balanced.json`):

| Step on the routed stream | No signal | Signal |
| --- | ---: | ---: |
| Flag memset before the routed quantize | none | 0.7–1.3 µs, then about 2 µs until the quantize starts |
| Quantize → routed kernel | 1.5 µs | 2.2 µs |
| Routed kernel, median of the overlap graph | 105.9 µs | 107.3 µs |
| Routed kernel end → next call's first node | 1.5 µs | 3.7 µs |
| Shared kernel start after routed kernel start | 5.6 µs | 11 µs |

- About 1.4 µs of the cost is the routed kernel itself running longer. The
  shared kernel now starts after the routed kernel's routing/dispatch phase,
  so more of its HBM traffic lands on the routed kernel's GEMM phase.
- The other ~2.7 µs comes from the extra memset/event/wait-value graph nodes
  on the routed critical path.
- In both modes the shared expert still ends 28–53 µs before the routed
  kernel, so the signal adds latency and hides no extra work.

**Correctness without the signal.** The routed kernel's 132–134 CTAs and the
shared kernel's 18–20 CTAs together fit in 152 SMs. In the no-signal trace
the routed kernel already started first (shared memset 0.06 µs earlier, shared
kernel 5.6 µs later). So the no-signal ordering is safe here, and the signal
only buys a guaranteed order.

### All 152 SMs for routed + 32 shared CTAs (16 tokens/rank)

Each case uses the routed policy from the tuned config above at
`--sm-margin 0` (152 CTAs), plus a 32-CTA shared expert with
`--shared-wait-for-routed-start`. Oversubscribing like this needs the new
diagnostic flag `--allow-sm-oversubscription`; the benchmark otherwise rejects
it.

Three alternating runs each of this and routed only at 152 CTAs
(`make_plan.py OVS`, `plans/OVS.json`). Per call, µs:

| Routing | Routed policy | Routed only, 152 | 152 + 32, signal | vs routed only at 152 | Best tuned split (routed + shared), no signal | Best tuned split, signal | Shared alone, 32 CTAs | Shared ends after routed |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| balanced | `k256-warps16` | 99.82 | 125.27 | +25.45 (+25.5%) | 108.40 (134 + 18) | 112.78 | 30.52 | +19.4 |
| hot_experts | `k128-expert-major` | 81.00 | 107.38 | +26.39 (+32.6%) | 91.45 (134 + 18) | 95.53 | 30.52 | +19.5 |
| lognormal | `k256-warps16` | 99.70 | 125.17 | +25.47 (+25.6%) | 108.68 (134 + 18) | 112.89 | 30.52 | +19.4 |
| zipf | `k256-warps16` | 85.76 | 111.91 | +26.15 (+30.5%) | 95.43 (132 + 20) | 99.74 | 30.52 | +19.6 |
| zipf_skewed | `k256-warps16` | 86.45 | 109.84 | +23.39 (+27.1%) | 95.44 (132 + 20) | 99.47 | 30.52 | +21.1 |

The routed and shared kernels effectively run one after the other. Each
routed CTA fills an SM:

- 512 threads × 128 registers = 65,536 registers, the whole register file;
- plus most of the shared memory.

So no shared CTA (not even the shared input-quantize kernel) can share an SM
with a running routed CTA.

The signal only orders the launches: it does not create SM capacity. In the
balanced rank-0 trace (`traces-sig/OV`, `results/ovtrace-balanced-152-32.json`):

- The routed kernel runs 0 → 100.8 µs.
- The shared quantize kernel waits until 95.1 µs, when the first routed CTAs
  exit.
- The shared kernel runs 97.2 → 121.4 µs.

Overlap is about 4 µs, and the shared expert adds about 20 µs after the routed
kernel. A shared expert overlapped with the routed kernel needs its own SMs
(routed CTAs + shared CTAs ≤ 152).

### W8A8 results

Balanced routing:

| Routed + shared CTAs | Per call (µs) | Routed-only control (µs) | Shared only (µs) |
| --- | --- | --- | --- |
| 150 + 0 | 99.1 | – | – |
| 146 + 6 | 125.3 | 99.6 | 97.9 |
| 144 + 8 | 111.2 | 104.2 | 71.9 |
| 140 + 12 | 110.4 | 103.8 | 55.5 |
| 138 + 14 | 109.8 | 103.3 | 53.5 |
| 136 + 16 | 109.1–109.7 | 103.0–103.2 | 43.4–43.6 |
| **134 + 18** | **108.7–108.8** | 102.8 | 41.2 |
| 128 + 24 | 112.0 | 106.7 | 39.1 |
| 136 + 16, W8A16 in the same session | 112.7 | 102.6 | 99.5 |

Hot experts:

| Routed + shared CTAs | Per call (µs) | Routed-only control (µs) | Shared only (µs) |
| --- | --- | --- | --- |
| 150 + 0 | 81.0 | – | – |
| 144 + 8 | 93.5 | 86.7 | 71.9 |
| 140 + 12 | 92.9 | 86.6 | 55.5 |
| 136 + 16 | 92.2–92.8 | 86.0–86.1 | 43.4 |
| **134 + 18** | **91.8** | 85.8–85.9 | 41.2 |
| 132 + 20 | 96.4 | 90.8 | 41.2 |
| 130 + 22 | 95.8 | 90.4 | 41.2 |
| 128 + 24 | 95.3 | 90.2 | 39.1 |

Kernel times at 136 + 16 (µs; the slowest rank's median):

| Routing | Routed kernel, alone | Routed kernel, overlapped | Shared kernel, alone | Shared kernel, overlapped |
| --- | --- | --- | --- | --- |
| Balanced, W8A8 | 100.9–101.0 | 106.9–107.2 | 38.2–38.3 | 51.4–51.7 |
| Hot experts, W8A8 | 83.8 | 89.9 | 38.3 | 51.8 |
| Balanced, W8A16 (from the next section) | – | 108.7 | – | 100.7 |

With W8A8, the shared kernel ends about 50 µs (balanced) or 35 µs (hot experts)
before the routed kernel. It is no longer on the critical path. The added
≈10 µs per call (balanced 136 + 16) has two parts:

1. **Fewer routed CTAs: +3.9 µs** (99.1 → 103.0 µs, from 150 to 136 CTAs).
   - The routed kernel assigns tasks to clusters round-robin:
     `for (i = cluster; i < tasks; i += gridDim.x / ClusterSize)` in
     `megamoe_mma.cuh` and `megamoe_w4a8_roles.cuh`. Its time therefore steps
     with ⌈tasks / clusters⌉.
   - Balanced routing gives each of the 16 local experts one token block of
     34 FC1 + 16 FC2 tasks, which is 800 tasks:

     | Routed CTAs (clusters) | Rounds | Routed-only time |
     | --- | --- | --- |
     | 150 (75) | 11 | 99.1 µs |
     | 134–144 (67–72) | 12 | 102.8–104.2 µs |
     | 128 (64) | 13 | 106.7 µs |

   - On rank 0, hot experts give 12 token blocks (four local experts receive no
     routes), which is 600 tasks: 8 rounds at 150 CTAs, 9 rounds at 134–144
     and 10 rounds at 128–132.
   - Both patterns therefore need at least 67 clusters (134 CTAs) to stay one
     round above routed only, and 134 + 18 gives the shared expert the most SMs
     at that cost. Other routings and token counts move these steps.
   - Fewer shared SMs do not help. At 144 + 8 the routed kernel still needs 12
     rounds, and the slower shared kernel overlaps almost all of it. At
     146 + 6 the routed kernel drops to 11 rounds, but the shared kernel then
     outlasts it.
2. **HBM contention: +6 µs** (103.0 → 109.2 µs).
   - Per call, the routed kernel streams about 454 MB of MXFP4 expert weights
     and the shared kernel about 55 MB of MXFP8 weights.
   - While the two kernels overlap, the routed kernel slows from 101.0 to
     106.9 µs and the shared kernel from 38.2 to 51.4 µs.
   - Storing the shared weights as MXFP4 would halve the shared traffic. That
     needs a weight format change, which was out of scope here.

Shared expert alone on one GPU (`$RUN/shared_probe2.py`: 16 tokens, four weight
banks, kernel time in µs):

| Shared CTAs | W8A16 | W8A8 |
| --- | --- | --- |
| 8 | 178.6 | 69–72 |
| 16 | 98.7–99.2 | 42.2–43.2 |
| 24 | 86.4–86.9 | 37.5–39.1 |
| 32 | 60.4 | 28.8 |

W8A8 removes the software MXFP8 → BF16 weight dequantization, which limited
W8A16. Both kernels' outputs differ from the Torch reference by at most 0.0078,
which is the BF16 rounding step at that magnitude.

### W8A16 results

Balanced routing:

| Routed + shared CTAs | Per call (µs) | Routed-only control (µs) | Routed kernel (µs) | Shared kernel (µs) |
| --- | --- | --- | --- | --- |
| 150 + 0 | 99.1 | – | 96.8 | – |
| **136 + 16** | **112.6–112.9** | 102.6 | 108.7 | 100.7 |
| 128 + 24 | 115.8 | 106.1 | – | – |

Hot experts:

| Routed + shared CTAs | Per call (µs) | Routed-only control (µs) | Routed kernel (µs) | Shared kernel (µs) |
| --- | --- | --- | --- | --- |
| 150 + 0 | 81.0 | – | 79.0 | – |
| 136 + 16 | 110.0 | 85.6 | 106.4 | 98.7 |
| **128 + 24** | **97.1** | 89.6 | 93.6 | 84.6 |
| 128 + 8 | 186.4 | 89.6 | 183.8 | 175.6 |
| 120 + 32 | 99.3 | 89.0 | 95.2 | 57.9 |
| 120 + 16 | 107.4 | 89.0 | 104.0 | 96.2 |

Variants of the balanced 136 + 16 run:

| Input mode, shared start | Per call (µs) |
| --- | --- |
| Staged, no wait (used above) | 112.6–112.9 |
| Direct, no wait | 114.8–115.1 |
| Direct, wait for routed start | 116.1 |
| Staged, wait for routed start | 121.1–121.2 |

Repeated runs varied by less than 0.3 µs per call. With `--graph-batch 40`,
per-call latency reads about 0.4 µs lower.

## Measurement notes

- Compare per-call numbers. Profiling inflates the memset and memcpy graph
  nodes, so do not derive per-call time from traces.
- When the shared kernel outlasts the routed kernel, ranks drift out of phase.
  The routed kernel then absorbs the skew while it waits in dispatch and
  combine, so its kernel time overstates the routed cost. Hot experts with
  136 + 16 shows this.
- With `--shared-weight-banks 1`, the 55 MB of shared weights stay in L2 across
  calls. A multi-layer model would not see this.
- With W8A16, no single split is best for both routing patterns: use 136 + 16
  for balanced routing and 128 + 24 for hot experts. With W8A8, 134 + 18 is the
  best for both.

## W8A8 shared expert

The W8A16 shared expert is the routed W8A16 kernel compiled in local mode.
`megaMoeW8A16<LocalMode>` with `W8A16CollectiveTypesT<true>` runs in a context
with world size one, one expert and top-k one. Local mode removes routing,
dispatch and combine, writes BF16 rows directly and uses route weight 1.

The W8A8 shared expert applies the same adaptation to the routed W4A8 kernel:

1. Compile-time local types. `W8A8LocalCollectiveTypes` is
   `W4A8CollectiveTypesT<…, Mxfp8Weights_ = true, Local_ = true>`:
   - MXFP8 weights instead of MXFP4. The block-scaled `kind::mxf8f6f4` MMA takes
     both.
   - `LocalExpert = true` removes routing, dispatch and combine.
2. The same world-size-one context, used through `forward_shared`.
3. The same local epilogue (`LocalMode == 2`): BF16 output and route weight 1.

There is one difference. The kernel is a separate function,
`megaMoeW8A8Local` in the new file `src/ext/megamoe/megamoe_w8a8.cu`, rather
than another instantiation of `megaMoeW4A8`. It reuses the W4A8 load, MMA and
epilogue roles, so the routed kernel body is unchanged.

Each shared call runs three operations on the shared stream:

1. A memset clears the FC1 → FC2 ready flags.
2. `quantizeInputStaticKernel` quantizes the BF16 input to FP8 E4M3 with the
   static scale.
3. `megaMoeW8A8Local` runs as a cooperative launch with two-CTA clusters on
   `--shared-sm-count` CTAs:
   - FC1 multiplies MXFP8 weights by FP8 activations.
   - SwiGLU produces the hidden activations, which are quantized to FP8 with
     the same scale.
   - FC2 writes BF16 output, multiplied by the scale.

### fp8_nv weights

`fp8_nv` stores each weight matrix as FP8 E4M3 values `q` with one FP32 scale
`s_w`, so that `w ≈ q × s_w`. FC1 (gate and up together) has `s_w1` and FC2
has `s_w2`. In the API, `fc1_scale` and `fc2_scale` are float32 tensors of
shape `(1,)`, one scale per local expert. With the static activation scale
`s_a`, the kernel computes:

```text
x_q    = fp8(x / s_a)
hidden = fp8(SwiGLU(s_a × s_w1 × (q1 · x_q)) / s_a)
y      = bf16(s_a × s_w2 × (q2 · hidden))
```

The kernel reuses the block-scaled MXFP8 pipeline unchanged and moves the
weight scale into the epilogue:

1. Packing (`packW4A8WeightsKernel<…, UnitScales = true>` in
   `megamoe_launch.cu`) copies the FP8 bytes as for MXFP8, but writes unit
   E8M0 block scales (127, which is 2⁰) instead of reading source block scales.
2. At construction, `runtime.cc` copies the two scales to the host once and
   checks that they are finite and positive.
3. `createKernelPlan` stores them in `W8A8LocalParameters::fc1WeightScale` and
   `fc2WeightScale`.
4. `staticDequantScale<Fc>` in `megamoe_epilogue.cuh` multiplies the FP32
   accumulators by `s_a × s_w1` before the FC1 clamp and SwiGLU, and by
   `s_a × s_w2` for the FC2 BF16 output.
   - The epilogue already multiplied by `s_a`, so this adds no per-element
     work.
   - MXFP8 weights get weight scales of 1.0, so their results are unchanged.
   - Routed kernels use `s_a` alone.

Because the scales are read only at construction, changing the scale tensors
afterwards has no effect, as for the weights themselves. `fp8_nv` is accepted
only for a single local W8A8 expert (`forward_shared`). Routed experts keep
MXFP4.

The unit block scales are still loaded by TMA, as MXFP8 loads its real block
scales, which is about 3% of the shared kernel's bytes (about 1.7 MB per call).
A later change could skip these loads for fp8_nv and use constant scales in
shared memory. The shared kernel is off the critical path, so the expected gain
is only a slightly lower HBM contention with the routed kernel.

### W8A8 kernel changes and rollback

All changes are uncommitted on top of `e17318f2`. Paths below are relative to
`src/ext/megamoe/` unless they start with `python/`.

| File | Change |
| --- | --- |
| `megamoe_w8a8.cu` (new) | The `megaMoeW8A8Local` kernel and its entry point. |
| `include/megamoe_collective.cuh` | `W4A8CollectiveTypesT` gains `bool Mxfp8Weights_ = false, bool Local_ = false`, which select the weight type and set `LocalExpert`. Adds the `W8A8LocalCollectiveTypes` alias. |
| `include/megamoe_device.cuh` | Local types get no dispatch shared storage. Adds `W8A8LocalParameters` and the entry declaration. |
| `include/megamoe_w4a8_roles.cuh` | When local, skips expert-weight caching and the wait for dispatched input. Passes the token count to `taskAt`. |
| `include/megamoe_mma.cuh` | Passes the token count to `taskAt`. |
| `include/megamoe_epilogue.cuh` | `LocalMode == 2` uses route weight 1 and scales the BF16 output by the activation scale. The FP8 packing path requires `LocalMode == 0`. |
| `megamoe_launch.cu`, `runtime.cc`, `include/megamoe_kernel.hpp`, `CMakeLists.txt` | Workspace layout, MXFP8 weight packing, plan and launch, the `forward_shared` input path, build and JIT install lists. |
| `python/mscclpp/ext/megamoe/api.py` | Accepts `weight_dtype="mxfp8"` with `activation_dtype="fp8_static"` for shared contexts only (`is_w8a8`). |
| `python/mscclpp/ext/megamoe/benchmark.py` | Adds `--shared-activation-dtype` and `--shared-activation-scale`, the W8A8 reference, and recognition of the W8A8 kernel in traces. |
| `python/test/test_megamoe.py`, `python/mscclpp/ext/megamoe/README.md` | Tests and documentation. |

The fp8_nv changes come on top of these:

| File | Change |
| --- | --- |
| `include/megamoe_kernel.hpp` | Adds `WeightDtype::Fp8Nv = 2`, and `PackedWeights::fc1TensorScale` and `fc2TensorScale` (default 1). |
| `include/megamoe_device.cuh` | `W8A8LocalParameters` becomes a struct that derives from the local `KernelParameters` and adds `fc1WeightScale` and `fc2WeightScale`. Routed parameter layouts are unchanged. |
| `include/megamoe_epilogue.cuh` | `staticDequantScale<Fc>` folds the weight scale into the FC1 static and local FC2 scales, for `W8A8LocalParameters` only. |
| `megamoe_launch.cu` | Accepts `Fp8Nv` for local W8A8 only, packs it with `UnitScales`, and copies the scales into the plan. |
| `runtime.cc` | Reads and validates the two scales once, at construction. `forwardShared` accepts `Fp8Nv`. |
| `megamoe_w8a8.cu` | Header comment only. |
| `python/csrc/ext/megamoe.cpp` | Adds the binding enum value `Fp8Nv`. This needs a rebuilt `_mscclpp` module. |
| `python/mscclpp/ext/megamoe/api.py` | Accepts `weight_dtype="fp8_nv"` with static FP8 activations and one local expert. Validates float32 `(e,)` scales. |
| `python/mscclpp/ext/megamoe/benchmark.py` | Adds `--shared-weight-dtype {mxfp8,fp8_nv}`, fp8_nv weights (amax / 448) and their reference, and `shared.weight_dtype` in the JSON. |
| `python/test/test_megamoe.py`, `python/mscclpp/ext/megamoe/README.md` | fp8_nv configuration tests, the GPU test run for both weight formats (it also checks that zero and NaN scales are rejected), and documentation. |

The defaults keep the routed kernel unchanged. Its SASS is identical to the
original build, and routed only measured 99.1 µs with both packages. After the
fp8_nv change, the routed SASS and every routed `ptxas -v` line are again
identical. Only the packing kernel's symbol name changed, because of the new
template parameter.

The routed-path edits compile away for routed kernels: they are
`if constexpr (LocalExpert)` branches, `LocalMode` checks and a defaulted token
count argument.

A first version used a defaulted type parameter (`class Weight_ =
mx_float4_t<…>`) instead of the bool.

- Although the type was the same, it changed `ptxas` register allocation for
  the routed static-FP8 kernel: stack grew from 8 to 40 bytes and spills from
  12 to 44–136 bytes.
- Routed only slowed from 99.1 to 100.5 µs.
- The bool parameter avoids this.
- After any header change, compare the routed kernel's `ptxas -v` lines in
  `<JIT_CACHE>/<key>.build.log`.

To roll back the source changes, use one of the following. The two options are
alternatives, so run only one of them.

```bash
cd /home/caiorocha/test/mscclpp-dg
# First, in both cases: the --allow-sm-oversubscription diagnostic flag.
git apply -R $RUN/oversub-flag.patch   # benchmark.py, README.md

# Option 1: fp8_nv only (keeps the MXFP8 W8A8 shared expert).
git apply -R $RUN/fp8nv-native-incremental.patch   # src/ext/megamoe and python/csrc/ext/megamoe.cpp
git apply -R $RUN/fp8nv-python-incremental.patch   # api.py, benchmark.py, README.md, test_megamoe.py

# Option 2: everything (W8A8 and fp8_nv).
git apply -R $RUN/w8a8-native.patch   # src/ext/megamoe (including megamoe_w8a8.cu) and python/csrc/ext/megamoe.cpp
git apply -R $RUN/w8a8-python.patch   # api.py, benchmark.py, README.md, test_megamoe.py
git rm -q --cached src/ext/megamoe/megamoe_w8a8.cu
```

`$RUN/w8a8-mxfp8-before-fp8nv.patch` is the full diff before fp8_nv.

To roll back the benchmark environment, use `$RUN/package-w8a8` and
`$RUN/jit-cache-w8a8` with key `10a95c41…` (MXFP8 W8A8 only), or `$RUN/package`
and `$RUN/jit-cache` with key `1955dc4f…` (original).
