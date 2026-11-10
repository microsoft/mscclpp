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

### Merged senior small-batch optimization (`origin/binyli/megamoe-opt`)

`origin/binyli/megamoe-opt` (commit `823fc021`, "Optimize W4A8 MegaMoE routing, pipelining, and combine for EP32") is merged into this branch. It changes the routed W4A8 kernel only:

- Kernel prologue: pipeline and TMEM setup move before routing, and the end-of-kernel cluster sync becomes a resident-grid barrier.
- Token-block metadata (first 64 blocks, 32 experts) is cached in shared memory.
- The FC2 route weights are staged by one dedicated CTA.
- Remote-route receive is single-pass when segments ≤ CTAs.
- Task order is staggered per wave (`waveTaskOrdinal` / `stepClusterOffset`), and the load warp prefetches readiness counters.
- When `dispatch_warps × dispatch_stages > 2`, the dispatch buffers share shared memory with the epilogue.
- `load_stages` 9/10 use smaller B stages; 10 is promoted to 11.
- Combine uses a transposed warp-task mapping.
- The BF16 → MXFP8 quantization is fused into routing. It only applies to `mxfp8` activations, **not** `fp8_static`.

Merge conflicts were only with the W8A8 local shared expert (`LocalExpert`):

- `W4A8SharedStorageT` keeps `tmemReady` for the local kernel only (last member), and the local kernel carries no dispatch buffers or state.
- The epilogue `dispatchDone` wait is skipped for local experts.
- The load and MMA roles take the new wave order and pass `tokens` to `taskAt`. Local FC1 skips readiness waits and prefetches.
- Both 4096/4352 and 4096/4608 W4A8 specializations are registered.

Known issues with the merged code:

- **tile_k 256 is broken** on the senior's branch itself: it gives NaN with both `fp8_static` and `mxfp8`, while tile_k 128 passes `--check`. The likely cause is in `megamoe_w4a8_collective.cuh`: `BStageBytes` and `largeBStride` are hard-coded for tile_k 128 (8 KB B stage), so TMA writes and MMA reads use different stage strides. No fix applied; tile_k 256 policies are excluded below.
- `dispatch_warps 4` + `dispatch_stages 2` exceeds shared memory.

Build and plans:

- Source: `$RUN/source-merged` (source-w8a8 + `merge-binyli-incremental.patch`, plus a `nullptr` input argument for the prepared launch). Built with `build-merged.sh` and assembled as `package-merged`.
- Policies: `policies-merged.json` / `policy-keys-merged.txt`. The new ones are `k128-ls10*` and `k128-e16w8-ls10` (e16w8 with load_stages 10).
- Plans come from `make_plan_merged.py`:
  - MA: 14 policies at margin 2.
  - MB: the top 3 at margins 0 and 4.
  - MC/OC: the best new vs the best old routed-only config, 3 interleaved repeats.
  - MS: the top 2 policies × shared 18/20 (8/12 at 64).
  - MSF/MRS: the best overlap config and routed-only at the same CTAs, 3 repeats.
- All values are µs medians, max across ranks, mean of 3 runs.

| Tokens | Pattern | Old routed-only best | Old shared+routed (C2) | New routed-only best | New shared+routed (policy, split) | New routed-only same CTAs | Shared overhead vs new best | New vs old shared+routed |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 8 | balanced | 97.03 | 106.32 | 91.01 | 100.36 (e16w8-ls10, s18) | 92.64 | +9.35 | -5.96 (-5.6%) |
| 8 | hot_experts | 77.82 | 88.57 | 72.67 | 84.58 (e16w8-ls10, s18) | 77.07 | +11.91 | -3.99 (-4.5%) |
| 8 | lognormal | 96.77 | 106.29 | 91.03 | 100.39 (e16w8-ls10, s18) | 92.64 | +9.36 | -5.90 (-5.6%) |
| 8 | zipf | 82.75 | 92.97 | 78.30 | 88.26 (e16w8-ls10, s20) | 80.65 | +9.97 | -4.71 (-5.1%) |
| 8 | zipf_skewed | 81.48 | 91.43 | 75.43 | 86.24 (epilogue16-warps8, s18) | 78.67 | +10.82 | -5.19 (-5.7%) |
| 16 | balanced | 99.09 | 108.36 | 91.81 | 101.40 (e16w8-ls10, s18) | 93.46 | +9.59 | -6.97 (-6.4%) |
| 16 | hot_experts | 81.04 | 91.44 | 76.21 | 87.30 (epilogue16-warps8, s18) | 78.92 | +11.09 | -4.13 (-4.5%) |
| 16 | lognormal | 99.40 | 108.72 | 92.52 | 102.31 (e16w8-ls10, s18) | 95.26 | +9.78 | -6.41 (-5.9%) |
| 16 | zipf | 85.16 | 95.46 | 81.37 | 91.36 (e16w8-ls10, s20) | 83.83 | +9.98 | -4.10 (-4.3%) |
| 16 | zipf_skewed | 86.18 | 95.39 | 82.94 | 92.54 (epilogue-reg192, s20) | 86.43 | +9.59 | -2.86 (-3.0%) |
| 32 | balanced | 100.56 | 109.49 | 92.90 | 102.48 (e16w8-ls10, s18) | 93.99 | +9.58 | -7.01 (-6.4%) |
| 32 | hot_experts | 85.63 | 95.02 | 82.83 | 92.56 (adaptive-cache, s20) | 85.85 | +9.73 | -2.46 (-2.6%) |
| 32 | lognormal | 102.05 | 111.78 | 96.73 | 106.75 (epilogue16-warps8, s18) | 99.48 | +10.01 | -5.03 (-4.5%) |
| 32 | zipf | 87.27 | 97.75 | 82.70 | 93.59 (epilogue16-warps8, s20) | 85.92 | +10.89 | -4.17 (-4.3%) |
| 32 | zipf_skewed | 99.94 | 107.27 | 96.56 | 107.96 (adaptive-expert-major, s18) | 101.24 | +11.40 | +0.69 (+0.6%) |
| 64 | balanced | 102.44 | 110.18 | 98.33 | 107.21 (epilogue-reg192, s12) | 102.48 | +8.87 | -2.97 (-2.7%) |
| 64 | hot_experts | 90.81 | 100.49 | 88.73 | 96.34 (adaptive-expert-major, s8) | 89.32 | +7.61 | -4.16 (-4.1%) |
| 64 | lognormal | 104.79 | 115.24 | 101.40 | 110.48 (adaptive-cache, s8) | 102.81 | +9.08 | -4.76 (-4.1%) |
| 64 | zipf | 92.82 | 100.67 | 89.56 | 96.48 (epilogue-reg192, s12) | 89.60 | +6.93 | -4.18 (-4.2%) |
| 64 | zipf_skewed | 126.81 | 137.92 | 123.66 | 135.79 (epilogue-reg192, s12) | 128.77 | +12.13 | -2.13 (-1.5%) |

- Routed-only: −6 to −7% at 8–16 tokens/rank, −3 to −8% at 32, −2 to −4% at 64. The new `k128-e16w8-ls10` wins most cases at ≤16 tokens/rank.
- Routed + shared: −3 to −7% at ≤32 tokens/rank (except zipf_skewed at 32, whose old best used tile_k 256) and −1.5 to −4% at 64. Balanced reaches 100–102 µs at 8–32 tokens/rank with the shared expert overlapped.
- The cost of adding the shared expert is unchanged (+7 to +12 µs vs the best routed-only).

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

## Fused v2 (shared expert inside the MegaMoE kernel)

`--fused-shared-expert` runs the shared expert inside the routed W4A8 kernel (`runFusedW4A8` in
`src/ext/megamoe/megamoe_w4a8.cu`). Goal: routed + shared ≈ routed-only, so the shared expert is fully hidden.

### Result

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

### Approaches tried (summary)

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

### Detailed final design and why each step helped

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

### Files touched (fused path only)

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

The full diff against the base (including the fused v1 work) is `fused-analysis/fused-v2-final.patch` in the run
directory.


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
