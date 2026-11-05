# Native MegaMoE Shared-Expert Benchmark

The native benchmark is located at:

```text
/home/caiorocha/test/mscclpp-dg/python/mscclpp/ext/megamoe/benchmark.py
```

Run it with `--overlap-shared-expert`. The sections below record the exact
configuration, commands, environment and results of the routed-only versus
routed + shared comparison, so that it can be reproduced.

## Shared expert

- One local world-size-one MSCCL++ W8A16 MegaMoE context per rank.
- BF16 activations and MXFP8 weights.
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

Append `--json-output <RESULT_FILE>` to every run. Optional flags:

- `--check` compares the routed and shared outputs with a Torch reference. It
  passed for both overlap runs.
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
  `/home/caiorocha/test/run-dg-mscclpp/native-megamoe-shared-tune-20261105T095309Z/`.
  The results and traces of the measured runs are in its `results/` and
  `traces/` directories.
  - `launch.sh <NAME> <PORT> <RUN_ARGUMENTS>` adds the common arguments and
    writes `results/<NAME>.json` and `logs/<NAME>.log`.
  - `rank.sh` is the per-rank wrapper.
  - `deploy.sh` copies the package and JIT cache to the worker nodes.
  - `gpu-idle.sh` checks that all GPUs are idle.

## Software

- CUDA 13.0 (driver, runtime and `nvcc`), PyTorch 2.11.0 and Python 3.12.
- The MSCCL++ package at `<PACKAGE_DIR>` consists of the prebuilt libraries from
  `/home/caiorocha/test/run-dg-mscclpp/kernel-only-tuning-n8-64-20261103T224544Z/packages/mscclpp-prepared`
  with this repository's `python/mscclpp` copied over them, including the
  updated `benchmark.py`.
  - The libraries were built as Release with `MSCCLPP_BUILD_EXT_MEGAMOE=ON`,
    `MSCCLPP_BUILD_PYTHON_BINDINGS=ON` and `MSCCLPP_BUILD_EXT_COLLECTIVES=ON`.
  - The MegaMoE kernels are unmodified.

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
`routed_control_latency_us_max_across_ranks` at the same routed CTA count. Kernel
times come from `--trace-dir` traces of the same configurations.

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
- No single split is best for both routing patterns. Use 136 + 16 for balanced
  routing and 128 + 24 for hot experts.

